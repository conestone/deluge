"""
InundationBasinLinker — WorldFloods
=====================================
Links WorldFloods inundated-area GeoJSON files to HydroBASINS catchments.

Workflow (per GeoJSON)
----------------------
1. Load the GeoJSON and extract features with class == 'flood'.
2. Reproject flood geometries to WGS84 (EPSG:4326) and union into one geometry.
3. Find all HydroBASINS level-9 basins that intersect the inundated area (in WGS84).
4. For each flooded basin, collect its complete upstream catchment via BFS
   on the reversed NEXT_DOWN graph and union into a lumped catchment geometry.
5. Clip the inundated area to each lumped catchment.
6. Optionally match each lumped catchment to a GRDC watershed by IoU.
7. Save flood_inundation and lumped_basins as GeoPackages in WGS84, one per GeoJSON.

Parallelism
-----------
Each GeoJSON is processed in its own worker process. Static data (HydroBASINS,
GRDC) is loaded once per worker via ProcessPoolExecutor's initializer.

Usage
-----
    linker = InundationBasinLinker(
        basins_path  = "/path/to/hybas_global_lvl9.gpkg",
        geojson_dir  = "/path/to/WorldFloods/floodmaps",
        output_dir   = "/path/to/DELUGE/A_basins_total_upstrm",
        grdc_path    = "/path/to/GRDC_Watersheds.shp",   # optional
        n_workers    = 16,
    )
    results = linker.run()
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

_GEOD = Geod(ellps="WGS84")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

BASINS_PATH  = CFG["hydrosheds_lvl9_gpkg"]
GEOJSON_DIR  = CFG["worldfloods_geojson_dir"]
OUTPUT_DIR   = CFG["deluge_preproc_dir"]
GRDC_PATH    = CFG["grdc_shp"]
METADATA_DIR = CFG["worldfloods_metadata_dir"]


# ---------------------------------------------------------------------------
# Pure helper functions (stateless)
# ---------------------------------------------------------------------------

# Control characters that never appear in any text encoding (excludes tab/LF/CR).
# Their presence in a file almost certainly indicates embedded binary data.
_BINARY_CTRL_RE = re.compile(rb"[\x01-\x08\x0e-\x1f]")


def _detect_encoding(path: Path) -> str:
    """Return the character encoding of *path*, or ``'binary'`` if the file
    contains control bytes that indicate embedded binary/corrupted data."""
    raw = path.read_bytes()
    if _BINARY_CTRL_RE.search(raw):
        return "binary"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        import chardet
        result = chardet.detect(raw[:16_384])
        if result.get("encoding") and (result.get("confidence") or 0) >= 0.7:
            return result["encoding"]
    except ImportError:
        pass
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _read_geojson(path: Path) -> gpd.GeoDataFrame:
    """Read a GeoJSON file, re-encoding to plain UTF-8 first if necessary.

    Raises ValueError for files that contain embedded binary data (detected
    via control-character scan), so corrupted files fail fast without writing
    a temp file.  For non-UTF-8 text files the content is decoded and rewritten
    to a temp file so GDAL/fiona always receive clean UTF-8 (works with both
    fiona and pyogrio backends).
    """
    import os, tempfile
    enc = _detect_encoding(path)
    if enc == "binary":
        raise ValueError(
            f"file contains embedded binary data and cannot be parsed as GeoJSON; "
            f"the source file is likely corrupted: {path.name}"
        )
    if enc.lower() in ("utf-8", "ascii"):
        return gpd.read_file(path)
    text = path.read_bytes().decode(enc)  # strips BOM when enc == "utf-8-sig"
    fd, tmp_path = tempfile.mkstemp(suffix=".geojson")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        return gpd.read_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_event_type_stems(metadata_dir: Path, event_types: set[str]) -> set[str]:
    """Return stems of metadata files whose event type is in *event_types*."""
    stems: set[str] = set()
    for f in metadata_dir.rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding=_detect_encoding(f)))
        except Exception:
            continue
        if data.get("event type") in event_types:
            stems.add(f.stem.removesuffix("_metadata"))
    return stems

def _build_upstream_index(basins: gpd.GeoDataFrame) -> dict[int, set[int]]:
    upstream: dict[int, set[int]] = defaultdict(set)
    for row in basins[["HYBAS_ID", "NEXT_DOWN"]].itertuples(index=False):
        if row.NEXT_DOWN != 0:
            upstream[row.NEXT_DOWN].add(row.HYBAS_ID)
    return dict(upstream)


def _get_all_upstream(basin_id: int, upstream_direct: dict) -> set[int]:
    visited: set[int] = set()
    queue = deque([basin_id])
    while queue:
        bid = queue.popleft()
        if bid in visited:
            continue
        visited.add(bid)
        for up in upstream_direct.get(bid, ()):
            queue.append(up)
    return visited


def _match_grdc(
    catchment_geom,
    grdc_local: gpd.GeoDataFrame | None,
    threshold: float,
) -> tuple[float | None, float | None]:
    if grdc_local is None or grdc_local.empty:
        return None, None
    candidates = grdc_local[grdc_local.intersects(catchment_geom)]
    if candidates.empty:
        return None, None
    best_no, best_iou = None, 0.0
    for row in candidates.itertuples(index=False):
        inter = row.geometry.intersection(catchment_geom).area
        union = row.geometry.union(catchment_geom).area
        iou   = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_no = iou, row.grdc_no
    if best_iou >= threshold:
        return best_no, round(best_iou, 4)
    return None, None


# ---------------------------------------------------------------------------
# Worker-process state — populated once per process by the initializer
# ---------------------------------------------------------------------------

_W_BASINS:              gpd.GeoDataFrame | None = None
_W_UPSTREAM:            dict | None             = None
_W_GRDC:                gpd.GeoDataFrame | None = None
_W_THRESHOLD:           float                   = 0.99
_W_COVERAGE_THRESHOLD:  float                   = 0.95
_W_FLOOD_DIR:           Path | None             = None
_W_BASINS_DIR:          Path | None             = None
_W_METADATA_DIR:        Path | None             = None


def _worker_init(
    basins_path:        str,
    grdc_path:          str | None,
    threshold:          float,
    coverage_threshold: float,
    flood_dir:          str,
    basins_dir:         str,
    metadata_dir:       str | None,
) -> None:
    """Load static data once per worker process."""
    global _W_BASINS, _W_UPSTREAM, _W_GRDC, _W_THRESHOLD, _W_COVERAGE_THRESHOLD, _W_FLOOD_DIR, _W_BASINS_DIR, _W_METADATA_DIR

    _W_BASINS              = gpd.read_file(basins_path)
    _W_UPSTREAM            = _build_upstream_index(_W_BASINS)
    if grdc_path:
        grdc = gpd.read_file(grdc_path)[["grdc_no", "geometry"]]
        if grdc.crs and grdc.crs.to_epsg() != 4326:
            grdc = grdc.to_crs("EPSG:4326")
        _W_GRDC = grdc
    else:
        _W_GRDC = None
    _W_THRESHOLD           = threshold
    _W_COVERAGE_THRESHOLD  = coverage_threshold
    _W_FLOOD_DIR           = Path(flood_dir)
    _W_BASINS_DIR          = Path(basins_dir)
    _W_METADATA_DIR        = Path(metadata_dir) if metadata_dir else None


def _worker_process_geojson(geojson_path_str: str) -> dict:
    """Process one WorldFloods GeoJSON file. Runs inside a worker process."""
    geojson_path = Path(geojson_path_str)
    stem         = geojson_path.stem

    # --- Load GeoJSON, extract flood features, reproject to WGS84 ----------
    gdf = _read_geojson(geojson_path)

    flood_gdf = gdf[gdf["w_class"] == "Flooded area"]
    if flood_gdf.empty:
        return {"geojson": geojson_path_str, "used": [], "not_used": [], "flood_path": None, "basins_path": None}

    # Reproject to WGS84 so we can intersect directly with HydroBASINS
    flood_4326     = flood_gdf.to_crs("EPSG:4326")
    inundated_geom = unary_union([
        g if g.is_valid else make_valid(g) for g in flood_4326.geometry.values
    ])

    # --- Derive extent box from metadata bounding box, fallback to flood bounds
    extent_box       = None
    flood_date       = None
    satellite_source = None
    if _W_METADATA_DIR is not None:
        base_stem = stem.removesuffix("_floodmap")
        meta_path = _W_METADATA_DIR / f"{base_stem}_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding=_detect_encoding(meta_path)))
            bb = meta.get("bounding box", {})
            if all(k in bb for k in ("west", "east", "north", "south")):
                extent_box = box(bb["west"], bb["south"], bb["east"], bb["north"])
            raw_date         = meta.get("satellite date")
            flood_date       = raw_date.split("T")[0] if raw_date else None
            satellite_source = meta.get("satellite")
    if extent_box is None:
        extent_box = box(*flood_4326.total_bounds)

    # --- Clip GRDC to extent (both already in WGS84) -----------------------
    grdc_local = None
    if _W_GRDC is not None:
        grdc_local = _W_GRDC[_W_GRDC.intersects(extent_box)].copy()

    # --- Find flooded basins in native WGS84 (no reprojection needed) ------
    candidates = _W_BASINS[_W_BASINS.intersects(extent_box)]
    flooded    = candidates[candidates.intersects(inundated_geom)]

    basin_geom = _W_BASINS.set_index("HYBAS_ID")["geometry"]

    used:           list[int]  = []
    not_used:       list[int]  = []
    flood_features: list[dict] = []
    basin_features: list[dict] = []

    for basin_row in flooded.itertuples(index=False):
        bid = basin_row.HYBAS_ID

        upstream_ids   = _get_all_upstream(bid, _W_UPSTREAM)
        is_headwater   = upstream_ids == {bid}
        n_upstream     = len(upstream_ids) - 1

        valid_ids      = [i for i in upstream_ids if i in basin_geom.index]
        catchment_geom = unary_union(basin_geom.loc[valid_ids].values)

        covered = extent_box.intersection(catchment_geom).area / catchment_geom.area if catchment_geom.area > 0 else 0.0
        if covered < _W_COVERAGE_THRESHOLD:
            not_used.append(bid)
            continue

        used.append(bid)

        grdc_no, grdc_iou = _match_grdc(catchment_geom, grdc_local, _W_THRESHOLD)
        basin_area_m2 = abs(_GEOD.geometry_area_perimeter(catchment_geom)[0])

        basin_features.append({
            "HYBAS_ID":       bid,
            "NEXT_DOWN":      basin_row.NEXT_DOWN,
            "headwater":      is_headwater,
            "n_upstream":     n_upstream,
            "basin_area_m2":  int(basin_area_m2),
            "basin_area_km2": round(basin_area_m2 / 1_000_000, 2),
            "grdc_no":        grdc_no,
            "grdc_iou":       grdc_iou,
            "geometry":       catchment_geom,
        })

        clipped = inundated_geom.intersection(catchment_geom)
        if clipped.is_empty:
            continue

        area_m2 = abs(_GEOD.geometry_area_perimeter(clipped)[0])
        pct = round((area_m2 * 100) / basin_area_m2, 2) if basin_area_m2 > 0 else None

        flood_features.append({
            "HYBAS_ID":                   bid,
            "headwater":                  is_headwater,
            "n_upstream":                 n_upstream,
            "area_m2":                    area_m2,
            "area_km2":                   round(area_m2 / 1_000_000, 2),
            "inundation_basin_percentage": pct,
            "flood_date":                 flood_date,
            "source_data":                "worldfloods",
            "satellite_source":           satellite_source,
            "geometry":                   clipped,
        })

    # --- Save outputs in WGS84 (only when at least one basin was used) ------
    flood_path = basins_path_out = None

    if used and flood_features:
        flood_path = _W_FLOOD_DIR / f"{stem}.gpkg"
        gpd.GeoDataFrame(flood_features, crs="EPSG:4326").to_file(flood_path, driver="GPKG")

    if used and basin_features:
        basins_path_out = _W_BASINS_DIR / f"{stem}.gpkg"
        gpd.GeoDataFrame(basin_features, crs="EPSG:4326").to_file(basins_path_out, driver="GPKG")

    return {
        "geojson":          geojson_path_str,
        "used":             used,
        "not_used":         not_used,
        "n_flood_features": len(flood_features),
        "flood_path":       str(flood_path) if flood_path else None,
        "basins_path":      str(basins_path_out) if basins_path_out else None,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InundationBasinLinker:

    def __init__(
        self,
        basins_path:          str | Path,
        geojson_dir:          str | Path,
        output_dir:           str | Path,
        grdc_path:            str | Path | None = None,
        metadata_dir:         str | Path | None = None,
        event_types:          set[str] | None   = None,
        grdc_iou_threshold:   float = 0.99,
        coverage_threshold:   float = 0.95,
        n_workers:            int   = 16,
    ) -> None:
        self.basins_path          = str(basins_path)
        self.geojson_dir          = Path(geojson_dir)
        self.output_dir           = Path(output_dir)
        self.grdc_path            = str(grdc_path) if grdc_path else None
        self.metadata_dir         = Path(metadata_dir) if metadata_dir else None
        self.event_types          = event_types if event_types is not None else {"Riverine flood"}
        self.grdc_iou_threshold   = grdc_iou_threshold
        self.coverage_threshold   = coverage_threshold
        self.n_workers            = n_workers

    def run(self) -> list[dict]:
        """
        Process all GeoJSON files in geojson_dir in parallel.

        Returns a list of per-file result dicts with keys:
            "geojson", "used", "not_used", "flood_path", "basins_path"
        """
        geojson_files = sorted(self.geojson_dir.glob("*.geojson"))
        if not geojson_files:
            print(f"No GeoJSON files found in {self.geojson_dir}")
            return []

        if self.metadata_dir is not None:
            valid_stems   = _load_event_type_stems(self.metadata_dir, self.event_types)
            before        = len(geojson_files)
            geojson_files = [f for f in geojson_files if f.stem.removesuffix("_floodmap") in valid_stems]
            print(
                f"Metadata filter ({', '.join(sorted(self.event_types))}): "
                f"{len(geojson_files)}/{before} files retained."
            )

        flood_dir  = self.output_dir / "flood_inundation" / "WorldFloods"
        basins_dir = self.output_dir / "lumped_basins" / "WorldFloods"
        flood_dir.mkdir(parents=True, exist_ok=True)
        basins_dir.mkdir(parents=True, exist_ok=True)

        n = len(geojson_files)
        print(f"Processing {n} GeoJSON file(s) with {self.n_workers} worker(s) ...")

        init_args = (
            self.basins_path,
            self.grdc_path,
            self.grdc_iou_threshold,
            self.coverage_threshold,
            str(flood_dir),
            str(basins_dir),
            str(self.metadata_dir) if self.metadata_dir else None,
        )

        results = []
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as executor:
            futures = {
                executor.submit(_worker_process_geojson, str(f)): f.name
                for f in geojson_files
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                    name   = Path(result["geojson"]).name
                    print(
                        f"[{done:4d}/{n}] {name} — "
                        f"used: {len(result['used'])}, not used: {len(result['not_used'])}"
                    )
                    results.append(result)
                except Exception as exc:
                    name = futures[future]
                    print(f"[{done:4d}/{n}] ERROR {name}: {exc}")
                    results.append({"geojson": name, "error": str(exc)})

        ok     = sum(1 for r in results if "error" not in r)
        errors = sum(1 for r in results if "error" in r)
        print(f"\nDone — {ok} succeeded, {errors} failed.")
        saved_flood  = sum(1 for r in results if r.get("flood_path"))
        saved_basins = sum(1 for r in results if r.get("basins_path"))
        print(f"Saved files — flood inundation: {saved_flood}, lumped basins: {saved_basins}.")
        total_basins     = sum(len(r.get("used", [])) for r in results)
        total_inundation = sum(r.get("n_flood_features", 0) for r in results)
        print(f"Total basins used: {total_basins} | Total inundation areas generated: {total_inundation}.")

        # --- Write CSV summary ----------------------------------------------
        import csv
        csv_path = self.output_dir / "flood_inundation" / "basin_usage_WorldFloods.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["geojson", "HYBAS_ID", "status"])
            for r in sorted(results, key=lambda x: x["geojson"]):
                if "error" in r:
                    continue
                name = Path(r["geojson"]).stem
                for bid in r["used"]:
                    writer.writerow([name, bid, "used"])
                for bid in r["not_used"]:
                    writer.writerow([name, bid, "not_used"])
        print(f"Saved CSV: {csv_path}")

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    linker = InundationBasinLinker(
        basins_path         = BASINS_PATH,
        geojson_dir         = GEOJSON_DIR,
        output_dir          = OUTPUT_DIR,
        grdc_path           = GRDC_PATH,
        metadata_dir        = METADATA_DIR,
        event_types         = {"Riverine flood", "Flash flood", "Flood"},
        coverage_threshold  = 0.95,
        n_workers           = 16,
    )
    linker.run()
