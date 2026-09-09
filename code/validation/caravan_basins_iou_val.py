"""
Match CARAVAN basin polygons to HydroSHEDS basins in basins.geoparquet by polygon
overlap (IoU >= MIN_OVERLAP). Output:
  - caravan_hybas_mapping.csv  : HYBAS_ID <-> gauge_id for all matches
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
CARAVAN_SHAPEFILES = Path(CFG["caravan_shapefiles_dir"])
BASINS_PATH        = Path(CFG["deluge_basins_geoparquet"])
OUT_DIR            = Path(CFG["validation_output_dir"])

SUBDATASETS = ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"]
MIN_OVERLAP = 0.99       # minimum IoU to consider two polygons the same basin
PROJ_CRS    = "ESRI:54009"  # Mollweide equal-area for area calculations

# ── Step 1: load all CARAVAN basin shapefiles ─────────────────────────────────
print("Loading CARAVAN basin shapefiles...")
parts = []
for sub in SUBDATASETS:
    shp = CARAVAN_SHAPEFILES / sub / f"{sub}_basin_shapes.shp"
    gdf = gpd.read_file(shp)[["gauge_id", "geometry"]]
    parts.append(gdf)
    print(f"  {sub}: {len(gdf)} basins")

caravan = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
print(f"Total CARAVAN basins: {len(caravan)}")

# ── Step 2: load HydroSHEDS basins ────────────────────────────────────────────
print("\nLoading basins.geoparquet...")
basins = gpd.read_parquet(BASINS_PATH)
print(f"Total HydroSHEDS basins: {len(basins)}")

# ── Step 3: project both to equal-area CRS for area computations ──────────────
print(f"\nReprojecting to {PROJ_CRS}...")
caravan_proj = caravan.to_crs(PROJ_CRS).copy()
caravan_proj["caravan_area"] = caravan_proj.geometry.area

basins_proj = basins.reset_index()[["HYBAS_ID", "geometry"]].to_crs(PROJ_CRS).copy()
basins_proj["hybas_area"] = basins_proj.geometry.area
basins_proj = basins_proj.set_index("HYBAS_ID")

# ── Step 4: spatial index join to find candidate overlapping pairs ─────────────
print("\nRunning spatial index join to find candidate pairs...")
candidates = gpd.sjoin(
    caravan_proj[["gauge_id", "caravan_area", "geometry"]],
    basins_proj[["hybas_area", "geometry"]].reset_index(),
    how="inner",
    predicate="intersects",
)
print(f"Candidate pairs: {len(candidates)}")

# ── Step 5: compute exact IoU for each candidate pair ─────────────────────────
print("\nComputing intersection-over-union for candidate pairs...")

basins_geom_map = basins_proj["geometry"]  # indexed by HYBAS_ID

ious, int_areas = [], []
# candidates.geometry is the caravan polygon (left side of the join)
for c_geom, row in tqdm(
    zip(candidates.geometry, candidates.itertuples()), total=len(candidates)
):
    b_geom = basins_geom_map.loc[row.HYBAS_ID]
    intersection = c_geom.intersection(b_geom).area
    union = row.caravan_area + row.hybas_area - intersection
    ious.append(intersection / union if union > 0 else 0.0)
    int_areas.append(intersection)

candidates["iou"]               = ious
candidates["intersection_area"] = int_areas

# ── Step 6: filter by IoU threshold ───────────────────────────────────────────
matches = candidates[candidates["iou"] >= MIN_OVERLAP].copy()
print(f"\nMatches with IoU >= {MIN_OVERLAP}: {len(matches)}")

# Diagnostics: duplicates
dup_hybas   = matches["HYBAS_ID"].duplicated(keep=False).sum()
dup_caravan = matches["gauge_id"].duplicated(keep=False).sum()
if dup_hybas:
    print(f"  WARNING: {dup_hybas} rows share a HYBAS_ID with another match (one-to-many)")
if dup_caravan:
    print(f"  WARNING: {dup_caravan} rows share a gauge_id with another match (many-to-one)")

# Breakdown by subdataset
matches["subdataset"] = matches["gauge_id"].str.split("_").str[0]
print("\nMatches per subdataset:")
print(matches.groupby("subdataset").size().to_string())

# IoU distribution for matches
print(f"\nIoU stats for matches:\n{matches['iou'].describe()}")

# ── Step 7: save mapping CSV ───────────────────────────────────────────────────
mapping = matches[["HYBAS_ID", "gauge_id", "iou", "caravan_area", "hybas_area"]].copy()
mapping = mapping.sort_values("HYBAS_ID").reset_index(drop=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
mapping_path = OUT_DIR / "caravan_hybas_mapping.csv"
mapping.to_csv(mapping_path, index=False)
print(f"\nSaved mapping: {mapping_path} ({len(mapping)} rows)")
print("Done.")
