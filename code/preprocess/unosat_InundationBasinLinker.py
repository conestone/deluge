"""
InundationBasinLinker - UNOSAT
==============================
Links UNOSAT flood-extent GeoPackage files to HydroBASINS catchments.

Workflow (per GeoPackage)
-------------------------
1. Read all layers; extract flood-water features via Water_Clas / d_Water_Cl.
2. Union flood geometries (already WGS84) into one inundated geometry.
3. Find all HydroBASINS level-9 basins that intersect the inundated area.
4. For each flooded basin, collect its complete upstream catchment via BFS
   on the reversed NEXT_DOWN graph and union into a lumped catchment geometry.
5. Extent check: discard catchments whose bounding box exceeds the data extent.
6. Clip the inundated area to each valid catchment.
7. Optionally match each lumped catchment to a GRDC watershed by IoU.
8. Save flood_inundation and lumped_basins as GeoPackages in WGS84, one per input file.

Parallelism
-----------
Each GeoPackage is processed in its own worker process. Static data (HydroBASINS,
GRDC) is loaded once per worker via ProcessPoolExecutor's initializer.

Usage
-----
    linker = InundationBasinLinker(
        basins_path = "/path/to/hybas_global_lvl9.gpkg",
        gpkg_dir    = "/path/to/UNOSAT/geopackage",
        output_dir  = "/path/to/DELUGE/A_basins_total_upstrm",
        grdc_path   = "/path/to/GRDC_Watersheds.shp",   # optional
        n_workers   = 16,
    )
    results = linker.run()
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from pyogrio import list_layers
from pyproj import Geod
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

_GEOD = Geod(ellps="WGS84")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

BASINS_PATH = CFG["hydrosheds_lvl9_gpkg"]
GPKG_DIR    = CFG["unosat_gpkg_dir"]
OUTPUT_DIR  = CFG["deluge_preproc_dir"]
GRDC_PATH   = CFG["grdc_shp"]

# Flood-water class values across all known UNOSAT column schemas
_FLOOD_CLASSES = {
    # "Satellite Detected Water",
    "Flood Water",
    # "Satellite Detected Water / Possible Saturated Soil",
}


# ---------------------------------------------------------------------------
# Pure helper functions (stateless)
# ---------------------------------------------------------------------------

def _extract_flood_geometries(gpkg_path: Path) -> dict[str | None, dict]:
    """
    Read all layers in a GeoPackage and return flood-water geometries grouped by date.

    Returns {date_str: {"geoms": [...], "sensors": set(...)}}
    where date_str is the ISO date from Sensor_Dat (or None if absent).

    UNOSAT files use two column schemas:
      - Water_Clas (str): 'Satellite Detected Water', 'Flood Water', …
      - d_Water_Cl (str): same values, used when Water_Clas is stored as int
    Geometry type may be Polygon or MultiPolygon; Unknown layers are skipped.
    """
    groups: dict[str | None, dict] = defaultdict(lambda: {"geoms": [], "sensors": set()})

    for layer_name, geom_type in list_layers(str(gpkg_path)):
        if geom_type not in ("Polygon", "MultiPolygon"):
            continue
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        if gdf.empty:
            continue

        flood_gdf = None
        if "Water_Clas" in gdf.columns and gdf["Water_Clas"].dtype == object:
            subset = gdf[gdf["Water_Clas"].isin(_FLOOD_CLASSES)]
            if not subset.empty:
                flood_gdf = subset
        if flood_gdf is None and "d_Water_Cl" in gdf.columns:
            subset = gdf[gdf["d_Water_Cl"].isin(_FLOOD_CLASSES)]
            if not subset.empty:
                flood_gdf = subset
        if flood_gdf is None:
            continue

        if flood_gdf.crs and flood_gdf.crs.to_epsg() != 4326:
            flood_gdf = flood_gdf.to_crs("EPSG:4326")

        has_date   = "Sensor_Dat" in flood_gdf.columns
        has_sensor = "Sensor_ID"  in flood_gdf.columns

        for row in flood_gdf.itertuples(index=False):
            g = row.geometry if row.geometry.is_valid else make_valid(row.geometry)

            date_str = None
            if has_date:
                val = getattr(row, "Sensor_Dat", None)
                if val is not None and str(val) != "NaT":
                    date_str = str(val)[:10]

            sensor_str = None
            if has_sensor:
                val = getattr(row, "Sensor_ID", None)
                if val is not None:
                    sensor_str = str(val)

            groups[date_str]["geoms"].append(g)
            if sensor_str:
                groups[date_str]["sensors"].add(sensor_str)

    return dict(groups)


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

_W_BASINS:    gpd.GeoDataFrame | None = None
_W_UPSTREAM:  dict | None             = None
_W_GRDC:      gpd.GeoDataFrame | None = None
_W_THRESHOLD: float                   = 0.99
_W_FLOOD_DIR: Path | None             = None
_W_BASINS_DIR: Path | None            = None


def _worker_init(
    basins_path: str,
    grdc_path:   str | None,
    threshold:   float,
    flood_dir:   str,
    basins_dir:  str,
) -> None:
    """Load static data once per worker process."""
    global _W_BASINS, _W_UPSTREAM, _W_GRDC, _W_THRESHOLD, _W_FLOOD_DIR, _W_BASINS_DIR

    _W_BASINS    = gpd.read_file(basins_path)
    _W_UPSTREAM  = _build_upstream_index(_W_BASINS)
    if grdc_path:
        grdc = gpd.read_file(grdc_path)[["grdc_no", "geometry"]]
        if grdc.crs and grdc.crs.to_epsg() != 4326:
            grdc = grdc.to_crs("EPSG:4326")
        _W_GRDC = grdc
    else:
        _W_GRDC = None
    _W_THRESHOLD  = threshold
    _W_FLOOD_DIR  = Path(flood_dir)
    _W_BASINS_DIR = Path(basins_dir)


def _worker_process_gpkg(gpkg_path_str: str) -> dict:
    """Process one UNOSAT GeoPackage file. Runs inside a worker process."""
    gpkg_path = Path(gpkg_path_str)
    stem      = gpkg_path.stem

    # --- Extract flood geometries grouped by date --------------------------
    raw_groups = _extract_flood_geometries(gpkg_path)
    if not raw_groups:
        return {"gpkg": gpkg_path_str, "used": [], "not_used": [], "n_flood_features": 0, "flood_path": None, "basins_path": None}

    # Union per date; collect sorted sensor list per date
    date_inundated: dict[str | None, tuple] = {}  # date → (geom, sensor_str)
    for date_str, group in raw_groups.items():
        geom       = unary_union(group["geoms"])
        sensor_str = ", ".join(sorted(group["sensors"])) if group["sensors"] else None
        date_inundated[date_str] = (geom, sensor_str)

    inundated_all = unary_union([g for g, _ in date_inundated.values()])

    # --- Derive extent box from total bounds of all flood features ----------
    extent_box = box(*inundated_all.bounds)

    # --- Clip GRDC to extent -----------------------------------------------
    grdc_local = None
    if _W_GRDC is not None:
        grdc_local = _W_GRDC[_W_GRDC.intersects(extent_box)].copy()

    # --- Find flooded basins ------------------------------------------------
    candidates = _W_BASINS[_W_BASINS.intersects(extent_box)]
    flooded    = candidates[candidates.intersects(inundated_all)]

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
        catchment_geom = unary_union(basin_geom.loc[valid_ids])

        if not extent_box.contains(catchment_geom.envelope):
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

        # One flood feature row per date
        for date_str, (date_geom, sensor_str) in date_inundated.items():
            clipped = date_geom.intersection(catchment_geom)
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
                "flood_date":                  date_str,
                "source_data":                 "unosat",
                "satellite_source":            sensor_str,
                "geometry":                    clipped,
            })

    # --- Save outputs in WGS84 (only when at least one basin was used) -----
    flood_path = basins_path_out = None

    if used and flood_features:
        flood_path = _W_FLOOD_DIR / f"{stem}.gpkg"
        gpd.GeoDataFrame(flood_features, crs="EPSG:4326").to_file(flood_path, driver="GPKG")

    if used and basin_features:
        basins_path_out = _W_BASINS_DIR / f"{stem}.gpkg"
        gpd.GeoDataFrame(basin_features, crs="EPSG:4326").to_file(basins_path_out, driver="GPKG")

    return {
        "gpkg":             gpkg_path_str,
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
        gpkg_dir:           str | Path,
        output_dir:         str | Path,
        grdc_path:          str | Path | None = None,
        grdc_iou_threshold: float = 0.99,
        n_workers:          int   = 16,
    ) -> None:
        self.basins_path        = str(basins_path)
        self.gpkg_dir           = Path(gpkg_dir)
        self.output_dir         = Path(output_dir)
        self.grdc_path          = str(grdc_path) if grdc_path else None
        self.grdc_iou_threshold = grdc_iou_threshold
        self.n_workers          = n_workers

    def run(self) -> list[dict]:
        """
        Process all GeoPackage files in gpkg_dir in parallel.

        Returns a list of per-file result dicts with keys:
            "gpkg", "used", "not_used", "flood_path", "basins_path"
        """
        gpkg_files = sorted(self.gpkg_dir.glob("*.gpkg"))
        if not gpkg_files:
            print(f"No GeoPackage files found in {self.gpkg_dir}")
            return []

        flood_dir  = self.output_dir / "flood_inundation" / "Unosat"
        basins_dir = self.output_dir / "lumped_basins" / "Unosat"
        flood_dir.mkdir(parents=True, exist_ok=True)
        basins_dir.mkdir(parents=True, exist_ok=True)

        n = len(gpkg_files)
        print(f"Processing {n} GeoPackage file(s) with {self.n_workers} worker(s) ...")

        init_args = (
            self.basins_path,
            self.grdc_path,
            self.grdc_iou_threshold,
            str(flood_dir),
            str(basins_dir),
        )

        results = []
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as executor:
            futures = {
                executor.submit(_worker_process_gpkg, str(f)): f.name
                for f in gpkg_files
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                    name   = Path(result["gpkg"]).name
                    print(
                        f"[{done:4d}/{n}] {name} — "
                        f"used: {len(result['used'])}, not used: {len(result['not_used'])}"
                    )
                    results.append(result)
                except Exception as exc:
                    name = futures[future]
                    print(f"[{done:4d}/{n}] ERROR {name}: {exc}")
                    results.append({"gpkg": name, "error": str(exc)})

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
        csv_path = self.output_dir / "flood_inundation" / "basin_usage_Unosat.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["gpkg", "HYBAS_ID", "status"])
            for r in sorted(results, key=lambda x: x["gpkg"]):
                if "error" in r:
                    continue
                name = Path(r["gpkg"]).stem
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
        basins_path = BASINS_PATH,
        gpkg_dir    = GPKG_DIR,
        output_dir  = OUTPUT_DIR,
        grdc_path   = GRDC_PATH,
        n_workers   = 12,
    )
    linker.run()
