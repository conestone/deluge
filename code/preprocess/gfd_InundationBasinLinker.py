"""
InundationBasinLinker - Global Flood Database
=====================
Links GFD inundated-area TIF files to HydroBASINS catchments.

Workflow (per TIF)
------------------
1. Vectorise inundated pixels (value=1) from the TIF.
2. Find all basins whose geometry intersects those pixels.
3. For each flooded basin, collect its complete upstream catchment via BFS
   on the reversed NEXT_DOWN graph.
4. Extent check: flag catchments whose bounding box exceeds the TIF extent.
5. Clip the inundated area to each valid catchment.
6. Optionally match each lumped catchment to a GRDC watershed by IoU.
7. Save flood_inundation and lumped_basins as GeoPackages, one per TIF.

Parallelism
-----------
Each TIF is processed in its own worker process.  Static data (HydroBASINS,
GRDC) is loaded once per worker via ProcessPoolExecutor's initializer, so
large DataFrames are never serialised between processes.

Usage
-----
    linker = InundationBasinLinker(
        basins_path = "testdata/hybas_af_lev06_v1c/hybas_af_lev06_v1c.shp",
        tif_dir     = "testdata/tifs",
        output_dir  = "testdata/output",
        grdc_path   = "/path/to/GRDC_Watersheds.shp",   # optional
        n_workers   = 16,
    )
    results = linker.run()
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from pyproj import CRS, Geod, Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform, unary_union

_GEOD = Geod(ellps="WGS84")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

BASINS_PATH   = CFG["hydrosheds_lvl6_gpkg"]
TIF_DIR       = CFG["gfd_inundated_tif_dir"]
OUTPUT_DIR    = CFG["deluge_preproc_dir"]
GRDC_PATH     = CFG["grdc_shp"]
METADATA_DIR  = CFG["gfd_metadata_dir"]


# ---------------------------------------------------------------------------
# Pure helper functions (stateless, used both in workers and tests)
# ---------------------------------------------------------------------------

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

_W_BASINS:        gpd.GeoDataFrame | None = None
_W_BASIN_GEOM:    gpd.GeoSeries   | None = None   # pre-indexed by HYBAS_ID
_W_UPSTREAM:      dict | None             = None
_W_GRDC:          gpd.GeoDataFrame | None = None
_W_THRESHOLD:     float                   = 0.99
_W_FLOOD_DIR:     Path | None             = None
_W_BASINS_DIR:    Path | None             = None
_W_METADATA_DIR:  Path | None             = None


def _worker_init(
    basins_path:  str,
    grdc_path:    str | None,
    threshold:    float,
    flood_dir:    str,
    basins_dir:   str,
    metadata_dir: str | None,
) -> None:
    """Load static data once per worker process."""
    global _W_BASINS, _W_BASIN_GEOM, _W_UPSTREAM, _W_GRDC, _W_THRESHOLD, _W_FLOOD_DIR, _W_BASINS_DIR, _W_METADATA_DIR

    # Load only the two attribute columns needed — geometry is always included.
    # This drops ~15 numeric columns per row and halves per-worker RAM for
    # the global level-9 file (~1.4 M basins).
    _W_BASINS     = gpd.read_file(basins_path, columns=["HYBAS_ID", "NEXT_DOWN"])
    _W_UPSTREAM   = _build_upstream_index(_W_BASINS)
    # Pre-build the indexed geometry Series and warm the spatial index so the
    # first TIF processed by this worker pays no extra overhead.
    _W_BASIN_GEOM = _W_BASINS.set_index("HYBAS_ID")["geometry"]
    _ = _W_BASINS.sindex  # warm spatial index

    if grdc_path:
        grdc = gpd.read_file(grdc_path)[["grdc_no", "geometry"]]
        if grdc.crs and grdc.crs.to_epsg() != 4326:
            grdc = grdc.to_crs("EPSG:4326")
        _W_GRDC = grdc
    else:
        _W_GRDC = None
    _W_THRESHOLD    = threshold
    _W_FLOOD_DIR    = Path(flood_dir)
    _W_BASINS_DIR   = Path(basins_dir)
    _W_METADATA_DIR = Path(metadata_dir) if metadata_dir else None


def _worker_process_tif(tif_path_str: str) -> dict:
    """Process one TIF file. Runs inside a worker process."""
    tif_path = Path(tif_path_str)
    stem     = tif_path.stem

    # --- Load and vectorise TIF -----------------------------------------
    with rasterio.open(tif_path) as src:
        tif_bounds = src.bounds
        tif_crs    = src.crs
        data       = src.read(1)
        raster_tf  = src.transform

    flood_mask = (data == 1).astype(np.uint8)
    del data
    polys = [
        shape(geom)
        for geom, val in shapes(flood_mask, mask=flood_mask, transform=raster_tf)
        if val == 1
    ]
    inundated_geom = unary_union(polys)
    del polys

    # Bring flood geometry and TIF bounding box into WGS84 so that all
    # subsequent work happens in the same CRS as HydroBASINS (_W_BASINS).
    # For the vast majority of GFD TIFs this is a no-op.
    basins_crs = _W_BASINS.crs
    if tif_crs != basins_crs:
        _t = Transformer.from_crs(tif_crs, basins_crs, always_xy=True)
        inundated_geom = shapely_transform(_t.transform, inundated_geom)
        bounds_in_basins_crs = transform_bounds(tif_crs, basins_crs, *tif_bounds)
    else:
        bounds_in_basins_crs = tif_bounds
    tif_box = box(*bounds_in_basins_crs)

    # --- Load flood date from metadata (keyed by DFO ID before "_From_") --
    flood_date = None
    if _W_METADATA_DIR is not None:
        dfo_id    = stem.split("_From_")[0]
        meta_path = _W_METADATA_DIR / f"{dfo_id}_properties.json"
        if meta_path.exists():
            flood_date = json.loads(meta_path.read_text()).get("began")
    satellite_source = "MODIS"

    # --- Clip GRDC to TIF extent (GRDC is already in WGS84) ------------
    grdc_local = None
    if _W_GRDC is not None:
        grdc_local = _W_GRDC[_W_GRDC.intersects(tif_box)].copy()

    # --- Find flooded basins — clip first, skip full reprojection ------
    # Filter in native WGS84 (cheap spatial index lookup on ~dozens of rows)
    # instead of reprojecting all 1.4 M basins to TIF CRS first.
    candidates = _W_BASINS[_W_BASINS.intersects(tif_box)]
    flooded    = candidates[candidates.intersects(inundated_geom)]

    used:           list[int]  = []
    not_used:       list[int]  = []
    flood_features: list[dict] = []
    basin_features: list[dict] = []

    for basin_row in flooded.itertuples(index=False):
        bid = basin_row.HYBAS_ID

        upstream_ids   = _get_all_upstream(bid, _W_UPSTREAM)
        is_headwater   = upstream_ids == {bid}
        n_upstream     = len(upstream_ids) - 1

        # Use the pre-built indexed GeoSeries — no per-TIF reconstruction.
        valid_ids      = [i for i in upstream_ids if i in _W_BASIN_GEOM.index]
        catchment_geom = unary_union(_W_BASIN_GEOM.loc[valid_ids].values)

        if not tif_box.contains(catchment_geom.envelope):
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
            "HYBAS_ID":                    bid,
            "headwater":                   is_headwater,
            "n_upstream":                  n_upstream,
            "area_m2":                     area_m2,
            "area_km2":                    round(area_m2 / 1_000_000, 2),
            "inundation_basin_percentage": pct,
            "flood_date":                  flood_date,
            "source_data":                 "GFD",
            "satellite_source":            satellite_source,
            "geometry":                    clipped,
        })

    # --- Save outputs in WGS84 (only when at least one basin was used) -----
    flood_path = basins_path_out = None

    if used and flood_features:
        flood_path = _W_FLOOD_DIR / f"{stem}.gpkg"
        gpd.GeoDataFrame(flood_features, crs=basins_crs).to_file(flood_path, driver="GPKG")

    if used and basin_features:
        basins_path_out = _W_BASINS_DIR / f"{stem}.gpkg"
        gpd.GeoDataFrame(basin_features, crs=basins_crs).to_file(basins_path_out, driver="GPKG")

    return {
        "tif":              tif_path_str,
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
        basins_path:        str | Path,
        tif_dir:            str | Path,
        output_dir:         str | Path,
        grdc_path:          str | Path | None = None,
        metadata_dir:       str | Path | None = None,
        grdc_iou_threshold: float = 0.99,
        n_workers:          int   = 4,
    ) -> None:
        self.basins_path        = str(basins_path)
        self.tif_dir            = Path(tif_dir)
        self.output_dir         = Path(output_dir)
        self.grdc_path          = str(grdc_path) if grdc_path else None
        self.metadata_dir       = Path(metadata_dir) if metadata_dir else None
        self.grdc_iou_threshold = grdc_iou_threshold
        self.n_workers          = n_workers

    def run(self) -> list[dict]:
        """
        Process all TIF files in tif_dir in parallel.

        Returns a list of per-TIF result dicts with keys:
            "tif", "used", "not_used", "flood_path", "basins_path"
        """
        tif_files = sorted(self.tif_dir.glob("*.tif"))
        if not tif_files:
            print(f"No TIF files found in {self.tif_dir}")
            return []

        flood_dir  = self.output_dir / "flood_inundation" / "GFD"
        basins_dir = self.output_dir / "lumped_basins" / "GFD"
        flood_dir.mkdir(parents=True, exist_ok=True)
        basins_dir.mkdir(parents=True, exist_ok=True)

        n = len(tif_files)
        print(f"Processing {n} TIF file(s) with {self.n_workers} worker(s) ...")

        init_args = (
            self.basins_path,
            self.grdc_path,
            self.grdc_iou_threshold,
            str(flood_dir),
            str(basins_dir),
            str(self.metadata_dir) if self.metadata_dir else None,
        )

        results = []
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_init,
            initargs=init_args,
            max_tasks_per_child=500,
        ) as executor:
            futures = {
                executor.submit(_worker_process_tif, str(f)): f.name
                for f in tif_files
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                    name   = Path(result["tif"]).name
                    print(
                        f"[{done:4d}/{n}] {name} — "
                        f"used: {len(result['used'])}, not used: {len(result['not_used'])}"
                    )
                    results.append(result)
                except Exception as exc:
                    name = futures[future]
                    print(f"[{done:4d}/{n}] ERROR {name}: {exc}")
                    results.append({"tif": name, "error": str(exc)})

        ok     = sum(1 for r in results if "error" not in r)
        errors = sum(1 for r in results if "error" in r)
        print(f"\nDone — {ok} succeeded, {errors} failed.")
        saved_flood  = sum(1 for r in results if r.get("flood_path"))
        saved_basins = sum(1 for r in results if r.get("basins_path"))
        print(f"Saved files — flood inundation: {saved_flood}, lumped basins: {saved_basins}.")
        total_basins     = sum(len(r.get("used", [])) for r in results)
        total_inundation = sum(r.get("n_flood_features", 0) for r in results)
        print(f"Total basins used: {total_basins} | Total inundation areas generated: {total_inundation}.")

        # --- Write CSV summary ------------------------------------------
        import csv
        csv_path = self.output_dir / "flood_inundation" / "basin_usage_GFD.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tif", "HYBAS_ID", "status"])
            for r in sorted(results, key=lambda x: x["tif"]):
                if "error" in r:
                    continue
                tif_name = Path(r["tif"]).stem
                for bid in r["used"]:
                    writer.writerow([tif_name, bid, "used"])
                for bid in r["not_used"]:
                    writer.writerow([tif_name, bid, "not_used"])
        print(f"Saved CSV: {csv_path}")

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    linker = InundationBasinLinker(
        basins_path  = BASINS_PATH,
        tif_dir      = TIF_DIR,
        output_dir   = OUTPUT_DIR,
        grdc_path    = GRDC_PATH,
        metadata_dir = METADATA_DIR,
        n_workers    = 8,
    )
    linker.run()
