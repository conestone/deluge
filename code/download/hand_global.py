#!/usr/bin/env python3
"""
Download the GlobalHAND/90m-global/hand-1000 image from Google Earth Engine
for the entire world, tile by tile, then merge all tiles into one GeoTIFF.

Requirements
------------
    pip install earthengine-api geemap gdal

Authentication (run once)
-------------------------
    earthengine authenticate

Notes
-----
- At 90 m, the world is ~444 000 × 222 000 pixels (~98 Gpx).
- GEE's direct-download API limits each request to ~32 MB, so we split the
  globe into 2×2° tiles (≈2 467×2 467 px each, ~24 MB Float32).
- That yields up to 16 200 tiles.  Only tiles that contain actual data are
  saved (empty / ocean tiles are skipped automatically by GEE).
- After downloading, a GDAL VRT is built and translated to a single
  Cloud-Optimised GeoTIFF (COG).  The VRT step avoids loading all tiles
  into RAM at once.
"""

import ee
import os
import sys
import glob
import time
import zipfile
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── optional GDAL import (fall back to CLI tools) ──────────────────────────
try:
    from osgeo import gdal
    gdal.UseExceptions()
    GDAL_PYTHON = True
except ImportError:
    import subprocess
    GDAL_PYTHON = False

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  paths / GEE project pulled from _run/config.yml
# ═══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

GEE_IMAGE    = CFG["gee_image"]
OUTPUT_DIR   = CFG["hand_tiles_dir"]        # directory that receives *.tif tiles
OUTPUT_FILE  = CFG["hand_output_file"]      # final merged output
SCALE        = 90                    # spatial resolution in metres
CRS          = "EPSG:4326"
LON_STEP     = 2                     # tile width  in degrees  (keep ≤ 4)
LAT_STEP     = 2                     # tile height in degrees  (keep ≤ 4)
MAX_WORKERS  = 8                     # parallel download threads
MAX_RETRIES  = 3                     # retries per tile on error
GEE_PROJECT  = CFG["gee_project"]
# ═══════════════════════════════════════════════════════════════════════════


# ── print lock so parallel threads don't garble output ─────────────────────
_print_lock = threading.Lock()

def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ── Earth Engine initialisation ─────────────────────────────────────────────
def _resolve_project() -> str | None:
    """Return a GEE project ID from (in priority order):
    1. The GEE_PROJECT constant set at the top of this file.
    2. The GOOGLE_CLOUD_PROJECT or EE_PROJECT environment variables.
    3. Interactive prompt.
    """
    if GEE_PROJECT:
        return GEE_PROJECT
    for env_var in ("GOOGLE_CLOUD_PROJECT", "EE_PROJECT", "EARTHENGINE_PROJECT"):
        val = os.environ.get(env_var)
        if val:
            log(f"Using GEE project from env {env_var}: {val}")
            return val
    print("\nNo GEE project ID found in config or environment variables.")
    print("You can find your project at https://console.cloud.google.com/")
    project = input("Enter your Google Cloud / Earth Engine project ID: ").strip()
    if not project:
        sys.exit("A project ID is required. Exiting.")
    return project


def init_ee() -> None:
    project = _resolve_project()
    try:
        ee.Initialize(project=project)
        log(f"Earth Engine initialised (project: {project}).")
    except Exception:
        log("Credentials not found – running authentication …")
        ee.Authenticate()
        ee.Initialize(project=project)
        log(f"Earth Engine initialised (project: {project}).")


# ── tile grid ───────────────────────────────────────────────────────────────
def build_tile_list(lon_step: int = LON_STEP, lat_step: int = LAT_STEP) -> list[dict]:
    """Return a list of bounding-box dicts covering the whole world."""
    tiles = []
    for lon in range(-180, 180, lon_step):
        for lat in range(-90, 90, lat_step):
            tiles.append({
                "lon_min": lon,
                "lat_min": lat,
                "lon_max": min(lon + lon_step, 180),
                "lat_max": min(lat + lat_step, 90),
            })
    return tiles


def tile_filename(t: dict) -> str:
    return (
        f"hand_lon{t['lon_min']:+04d}_lat{t['lat_min']:+03d}"
        f"_lon{t['lon_max']:+04d}_lat{t['lat_max']:+03d}.tif"
    )


# ── single-tile download ─────────────────────────────────────────────────────
def download_tile(image: ee.Image, tile: dict, output_dir: str) -> str | None:
    """
    Download one tile.  Returns the local file path on success, None on failure.
    Skips tiles that already exist on disk.
    """
    fname   = tile_filename(tile)
    fpath   = os.path.join(output_dir, fname)

    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        log(f"  [skip] {fname}")
        return fpath

    region = ee.Geometry.Rectangle(
        [tile["lon_min"], tile["lat_min"], tile["lon_max"], tile["lat_max"]]
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            url = image.getDownloadURL({
                "scale":       SCALE,
                "crs":         CRS,
                "region":      region,
                "format":      "GEO_TIFF",
                "filePerBand": False,
            })

            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()

            tmp_path = fpath + ".tmp"
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65_536):
                    f.write(chunk)

            # GEE sometimes returns a zip archive – unwrap it
            if zipfile.is_zipfile(tmp_path):
                with zipfile.ZipFile(tmp_path) as z:
                    tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
                    if not tifs:
                        os.remove(tmp_path)
                        log(f"  [empty] {fname} – no TIF inside zip, skipping")
                        return None
                    z.extract(tifs[0], output_dir)
                    os.replace(os.path.join(output_dir, tifs[0]), fpath)
                os.remove(tmp_path)
            else:
                os.replace(tmp_path, fpath)

            log(f"  [ok]   {fname}")
            return fpath

        except Exception as exc:
            log(f"  [err]  {fname}  attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

    log(f"  [fail] {fname} – giving up after {MAX_RETRIES} attempts")
    return None


# ── parallel tile download ───────────────────────────────────────────────────
def download_all_tiles(image: ee.Image, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    tiles = build_tile_list()
    total = len(tiles)
    log(f"\nTile grid: {total} tiles  ({LON_STEP}°×{LAT_STEP}°, scale={SCALE} m)")
    log(f"Output dir: {output_dir}\n")

    done_paths: list[str] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_tile, image, t, output_dir): t
            for t in tiles
        }
        for fut in as_completed(futures):
            done_count += 1
            result = fut.result()
            if result:
                done_paths.append(result)
            # progress line every 50 tiles
            if done_count % 50 == 0 or done_count == total:
                log(f"  Progress: {done_count}/{total} tiles processed "
                    f"({len(done_paths)} downloaded)")

    log(f"\nDownload complete: {len(done_paths)} tiles saved.")
    return done_paths


# ── merge tiles ──────────────────────────────────────────────────────────────
def merge_tiles(input_dir: str, output_file: str) -> None:
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        log("No TIF files found – nothing to merge.")
        return

    log(f"\nMerging {len(tif_files)} tiles → {output_file}")

    vrt_file = str(Path(output_file).with_suffix(".vrt"))

    if GDAL_PYTHON:
        # Build an in-memory VRT from all tiles
        log("  Building VRT …")
        vrt = gdal.BuildVRT(vrt_file, tif_files)
        vrt.FlushCache()
        vrt = None

        # Translate VRT → Cloud-Optimised GeoTIFF
        log("  Translating to COG GeoTIFF (may take a while) …")
        gdal.Translate(
            output_file,
            vrt_file,
            options=gdal.TranslateOptions(
                format="GTiff",
                creationOptions=[
                    "COMPRESS=LZW",
                    "PREDICTOR=2",
                    "TILED=YES",
                    "BLOCKXSIZE=512",
                    "BLOCKYSIZE=512",
                    "BIGTIFF=YES",
                ],
            ),
        )
    else:
        # Fall back to command-line GDAL tools
        log("  Building VRT (CLI) …")
        subprocess.run(
            ["gdalbuildvrt", vrt_file, *tif_files], check=True
        )
        log("  Translating to GeoTIFF (CLI) …")
        subprocess.run(
            [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "COMPRESS=LZW",
                "-co", "PREDICTOR=2",
                "-co", "TILED=YES",
                "-co", "BIGTIFF=YES",
                vrt_file,
                output_file,
            ],
            check=True,
        )

    size_gb = os.path.getsize(output_file) / 1e9
    log(f"  Saved: {output_file}  ({size_gb:.2f} GB)")
    log(f"  VRT kept at: {vrt_file}  (useful for quick re-merges)")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    init_ee()

    log(f"\nLoading image: {GEE_IMAGE}")
    hand = ee.Image(GEE_IMAGE)

    # Download all tiles
    download_all_tiles(hand, OUTPUT_DIR)

    # Merge into a single GeoTIFF
    merge_tiles(OUTPUT_DIR, OUTPUT_FILE)

    log("\nAll done.")
    log(f"  Tiles  : {OUTPUT_DIR}/")
    log(f"  Merged : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
