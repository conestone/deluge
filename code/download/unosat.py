"""
UNOSAT Flood Downloader
- Tests all download URLs in parallel (20 workers)
- Downloads in parallel (20 workers) and for each event:
    * Extracts all shapefiles  → shapefiles/{id}_{country}[_{layer}].*
    * Converts flood layers    → geopackage/{id}_{country}.gpkg
    * Writes metadata          → metadata/{id}_{country}.json
- Clears geopackage/, shapefiles/, metadata/ on every run
- Compresses shapefiles/ into shapefiles.zip when done
"""

import json
import re
import shutil
import sys
import time
import zipfile
import pandas as pd
import requests
import geopandas as gpd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

# Configuration
BASE_URL = "https://unosat.org"
INPUT_FILE = CFG["unosat_events_json"]
SHP_DIR = Path(CFG["unosat_shp_dir"])
GPKG_DIR = Path(CFG["unosat_gpkg_dir_rel"])
META_DIR = Path(CFG["unosat_meta_dir_rel"])
MAX_WORKERS = 20
DOWNLOAD_TIMEOUT = 120
CHECK_TIMEOUT = 10

SHP_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qpj"}
DOWNLOAD_RETRIES = 3


def load_events(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def sanitize_name(name):
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    name = re.sub(r"[\s]+", "_", name)
    return name.strip("_") or "unknown"


def extract_fs_id(image_file):
    if not image_file:
        return None
    parts = image_file.split("/")
    if len(parts) > 3 and parts[2] == "unosat_filesystem":
        return parts[3]
    return None


def url_candidates(fs_id, glide):
    patterns = [f"{glide}_SHP.zip", f"{glide}_shp.zip", f"{glide}.zip"]
    return [f"{BASE_URL}/static/unosat_filesystem/{fs_id}/{p}" for p in patterns]


def check_event_url(event_info):
    fs_id = event_info.get("fs_id")
    glide = event_info.get("glide")
    if not fs_id or not glide:
        return None
    for url in url_candidates(fs_id, glide):
        try:
            r = requests.head(url, timeout=CHECK_TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                return {**event_info, "url": url}
        except Exception:
            pass
    return None


def is_flood_layer(layer_name):
    """True only for actual flood extent layers."""
    n = layer_name.lower()
    if "flood" not in n:
        return False
    for excl in ("preflood", "analysis_extent", "analysisextent", "cloud", "damage"):
        if excl in n:
            return False
    return True


def find_date_column(gdf):
    """Return the first column that holds dates (by dtype, then by name)."""
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            return col
    for col in gdf.columns:
        if re.search(r"date|_dat|dat_|^dat$", col, re.IGNORECASE):
            return col
    return None


def format_date(val):
    """Format a date value as YYYYMMDD for use in filenames."""
    try:
        if pd.isna(val):
            return "nodate"
    except Exception:
        pass
    try:
        return pd.Timestamp(val).strftime("%Y%m%d")
    except Exception:
        return re.sub(r"[^\w]", "", str(val))[:10] or "nodate"


def dissolve_duplicates(gdf):
    """
    Merge features that share identical non-geometry attributes into one entity
    by dissolving their geometries. Returns the original GDF if no duplicates
    are found or if dissolve fails.
    """
    geom_col = gdf.geometry.name
    attr_cols = [c for c in gdf.columns if c != geom_col]
    if not attr_cols or not gdf.duplicated(subset=attr_cols).any():
        return gdf
    try:
        return gdf.dissolve(by=attr_cols, as_index=False)
    except Exception:
        return gdf


def download_event(url_info):
    """
    Download ZIP, then:
      1. Extract every .shp (+ companions) into SHP_DIR
      2. For each flood layer: split by date → one GPKG per unique date
         named {base_name}_{date}.gpkg; if no date column, {base_name}.gpkg
    Returns (success: bool, base_name: str, status: str).
    """
    event_id = url_info["event_id"]
    base_name = f"{event_id}_{sanitize_name(url_info['country'])}"
    zip_path = SHP_DIR / f"{base_name}_tmp.zip"
    created_gpkgs = []  # track for cleanup on error

    try:
        # ── Download (with retries on broken connections) ─────────────────────
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                r = requests.get(url_info["url"], stream=True, timeout=DOWNLOAD_TIMEOUT)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                break
            except Exception:
                if zip_path.exists():
                    zip_path.unlink()
                if attempt == DOWNLOAD_RETRIES:
                    raise
                time.sleep(2 * attempt)

        with zipfile.ZipFile(zip_path, "r") as zf:
            all_entries = zf.namelist()

        all_shp = [e for e in all_entries if e.lower().endswith(".shp")]
        if not all_shp:
            return False, base_name, "no .shp found in ZIP"

        multi = len(all_shp) > 1

        # ── Extract shapefiles ────────────────────────────────────────────────
        with zipfile.ZipFile(zip_path, "r") as zf:
            for shp_entry in all_shp:
                shp_stem = Path(shp_entry).stem
                file_base = f"{base_name}_{shp_stem}" if multi else base_name
                for entry in all_entries:
                    ep = Path(entry)
                    if ep.stem == shp_stem and ep.suffix.lower() in SHP_EXTENSIONS:
                        (SHP_DIR / f"{file_base}{ep.suffix.lower()}").write_bytes(
                            zf.read(entry)
                        )

        # ── Build GeoPackages (flood layers only, split by date) ──────────────
        flood_shp = [e for e in all_shp if is_flood_layer(Path(e).stem)]
        if not flood_shp:
            return True, base_name, f"{len(all_shp)} shp layers, no flood layers → gpkg skipped"

        # gpkg_name → first-write flag (to set mode="w" vs "a")
        gpkg_written: dict[str, bool] = {}

        for shp_entry in flood_shp:
            layer_name = Path(shp_entry).stem
            try:
                gdf = gpd.read_file(f"/vsizip/{zip_path}/{shp_entry}")
                date_col = find_date_column(gdf)

                if date_col:
                    unique_dates = gdf[date_col].dropna().unique()
                else:
                    unique_dates = []

                if len(unique_dates) > 1:
                    # Split into one GPKG per date
                    for date_val in unique_dates:
                        date_str = format_date(date_val)
                        gpkg_name = f"{base_name}_{date_str}"
                        gpkg_path = GPKG_DIR / f"{gpkg_name}.gpkg"
                        subset = dissolve_duplicates(gdf[gdf[date_col] == date_val].copy())
                        mode = "w" if gpkg_name not in gpkg_written else "a"
                        subset.to_file(gpkg_path, driver="GPKG", layer=layer_name, mode=mode)
                        if gpkg_path not in created_gpkgs:
                            created_gpkgs.append(gpkg_path)
                        gpkg_written[gpkg_name] = True
                else:
                    # Single date or no date column
                    if len(unique_dates) == 1:
                        gpkg_name = f"{base_name}_{format_date(unique_dates[0])}"
                    else:
                        gpkg_name = base_name
                    gpkg_path = GPKG_DIR / f"{gpkg_name}.gpkg"
                    mode = "w" if gpkg_name not in gpkg_written else "a"
                    dissolve_duplicates(gdf).to_file(
                        gpkg_path, driver="GPKG", layer=layer_name, mode=mode
                    )
                    if gpkg_path not in created_gpkgs:
                        created_gpkgs.append(gpkg_path)
                    gpkg_written[gpkg_name] = True

            except Exception:
                pass

        n_gpkgs = len(gpkg_written)
        status = f"{len(all_shp)} shp, {len(flood_shp)} flood → {n_gpkgs} gpkg"
        return True, base_name, status

    except Exception as e:
        for p in created_gpkgs:
            if p.exists():
                p.unlink()
        return False, base_name, str(e)
    finally:
        if zip_path.exists():
            zip_path.unlink()


def save_metadata(entry, base_name):
    with open(META_DIR / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main():
    print("=" * 72)
    print("UNOSAT FLOOD DOWNLOADER")
    print("=" * 72)

    # Clear all output folders
    for d in (SHP_DIR, GPKG_DIR, META_DIR):
        print(f"Clearing '{d}/'...")
        clear_dir(d)

    events = load_events(INPUT_FILE)
    print(f"Loaded {len(events)} entries from {INPUT_FILE}")

    event_infos = []
    raw_entries = {}

    for entry in events:
        if "map_event" not in entry:
            continue
        ev = entry["map_event"]
        event_id = ev.get("id")
        if event_id is None:
            continue
        raw_entries[event_id] = entry
        event_infos.append({
            "event_id": event_id,
            "glide": ev.get("glide", ""),
            "country": entry.get("area_event_name", "unknown"),
            "fs_id": extract_fs_id(ev.get("image_file", "")),
        })

    print(f"Events with IDs: {len(event_infos)}")

    # ── Step 1: Test all URLs in parallel ────────────────────────────────────
    print(f"\nTesting URLs for {len(event_infos)} events ({MAX_WORKERS} workers)...")
    available = []
    not_found_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_info = {pool.submit(check_event_url, info): info for info in event_infos}
        for i, future in enumerate(as_completed(future_to_info), 1):
            info = future_to_info[future]
            result = future.result()
            status = "OK  " if result else "MISS"
            print(
                f"  [{i:>4}/{len(event_infos)}] {status}  "
                f"id={info['event_id']}  {info['country']}",
                flush=True,
            )
            if result:
                available.append(result)
            else:
                not_found_count += 1

    print(f"\nURL check — available: {len(available)}, not found: {not_found_count}")
    if not available:
        print("Nothing to download.")
        return

    # ── Step 2: Download, extract shapefiles & build GeoPackages ─────────────
    print(f"\nDownloading {len(available)} events ({MAX_WORKERS} workers)...")
    successful = 0
    retry_queue = []  # url_infos that failed — retried sequentially afterwards

    def download_task(url_info):
        ok, base_name, status = download_event(url_info)
        if ok:
            save_metadata(raw_entries.get(url_info["event_id"], {}), base_name)
        return ok, base_name, status

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_url = {pool.submit(download_task, ui): ui for ui in available}
        for i, future in enumerate(as_completed(future_to_url), 1):
            ok, base_name, status = future.result()
            tag = "OK  " if ok else "FAIL"
            print(f"  [{i:>4}/{len(available)}] {tag}  {base_name}  ({status})", flush=True)
            if ok:
                successful += 1
            else:
                retry_queue.append(future_to_url[future])

    # ── Step 2b: Retry failures one by one ───────────────────────────────────
    if retry_queue:
        print(f"\nRetrying {len(retry_queue)} failed downloads sequentially...")
        still_failed = 0
        for i, url_info in enumerate(retry_queue, 1):
            print(f"  [{i:>3}/{len(retry_queue)}] Retrying {url_info['event_id']} {url_info['country']}...",
                  flush=True)
            ok, base_name, status = download_event(url_info)
            if ok:
                save_metadata(raw_entries.get(url_info["event_id"], {}), base_name)
                print(f"    OK    {base_name}  ({status})", flush=True)
                successful += 1
            else:
                print(f"    FAIL  {base_name}  ({status})", flush=True)
                still_failed += 1
        failed = still_failed
    else:
        failed = 0

    # ── Step 3: Compress and delete shapefiles/ folder ───────────────────────
    print(f"\nCompressing '{SHP_DIR}/' → shapefiles.zip ...")
    shutil.make_archive("shapefiles", "zip", root_dir=".", base_dir=str(SHP_DIR))
    shutil.rmtree(SHP_DIR)
    print(f"Deleted '{SHP_DIR}/'.")

    print("\n" + "=" * 72)
    print(f"Downloaded:  {successful}")
    print(f"Failed:      {failed}")
    print(f"Archive:     {Path('shapefiles.zip').resolve()}")
    print(f"GeoPackages: {GPKG_DIR.resolve()}")
    print(f"Metadata:    {META_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
