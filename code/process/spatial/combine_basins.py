import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402


def _extract_event_id(stem: str, source: str) -> str:
    if source == "GFD":
        m = re.match(r"^(DFO_\d+)_From", stem)
        return m.group(1) if m else stem
    elif source == "Unosat":
        m = re.match(r"^(\d+)_", stem)
        return f"UNOSAT_{m.group(1)}" if m else stem
    else:  # WorldFloods
        m = re.match(r"^EMSR(\d+)_", stem)
        return f"EMSR_{m.group(1)}" if m else stem

BASIN_DIR = Path(CFG["deluge_preproc_dir"])
OUTPUT_PATH = Path(CFG["deluge_basins_geoparquet"])
LUMPED_BASINS = BASIN_DIR / "lumped_basins"
FLOOD_INUNDATION = BASIN_DIR / "flood_inundation"

CARAVAN_SHAPEFILES = Path(CFG["caravan_shapefiles_dir"])
CARAVAN_SUBDATASETS = ["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"]
CARAVAN_MIN_IOU = 0.99
CARAVAN_PROJ_CRS = "ESRI:54009"  # Mollweide equal-area for area calculations

SOURCES = ["GFD", "Unosat", "WorldFloods"]
SOURCE_EVENT_COL = {"GFD": "tif", "Unosat": "gpkg", "WorldFloods": "geojson"}


# ── Step 1: load basin_usage CSVs, filter to 'used' ──────────────────────────
print("Loading basin usage data...")
usage_parts = []
for src in SOURCES:
    col = SOURCE_EVENT_COL[src]
    df = pd.read_csv(FLOOD_INUNDATION / f"basin_usage_{src}.csv")
    df = df[df["status"] == "used"][["HYBAS_ID", col]].copy()
    df = df.rename(columns={col: "event_name"})
    df["source"] = src
    usage_parts.append(df)

usage_df = pd.concat(usage_parts, ignore_index=True)

# ── Step 2: map each event to a flood date (always from inundation geopackage) ─
print("Building event → date mapping...")
event_dates: dict[str, str | None] = {}

for src in SOURCES:
    events = usage_df.loc[usage_df["source"] == src, "event_name"].unique()
    for event in tqdm(events, desc=f"  {src} event dates"):
        path = FLOOD_INUNDATION / src / f"{event}.gpkg"
        if path.exists():
            gdf = gpd.read_file(path)
            valid = gdf["flood_date"].dropna() if "flood_date" in gdf.columns else pd.Series([], dtype=str)
            event_dates[event] = str(valid.iloc[0]) if not valid.empty else None
        else:
            event_dates[event] = None

usage_df["flood_date"] = usage_df["event_name"].map(event_dates)
usage_df["event_id"]   = usage_df.apply(
    lambda row: _extract_event_id(row["event_name"], row["source"]), axis=1
)

# ── Step 3: aggregate flood event metadata per basin ─────────────────────────
print("Aggregating flood events per basin...")

# Unique flood dates per basin (ISO strings sort chronologically)
flood_dates_agg = (
    usage_df.groupby("HYBAS_ID")["flood_date"]
    .apply(lambda x: sorted({d for d in x.dropna() if d}))
    .rename("flood_event_dates")
    .reset_index()
)
flood_dates_agg["n_flood_events"] = flood_dates_agg["flood_event_dates"].apply(len)
flood_dates_agg["flood_event_dates"] = flood_dates_agg["flood_event_dates"].apply(", ".join)

# Unique (event_id, flood_date) pairs per basin — more precise than date alone
n_floodmaps_agg = (
    usage_df.dropna(subset=["flood_date"])
    .drop_duplicates(subset=["HYBAS_ID", "event_id", "flood_date"])
    .groupby("HYBAS_ID")
    .size()
    .rename("n_floodmaps")
    .reset_index()
)

sources_agg = (
    usage_df.groupby("HYBAS_ID")["source"]
    .apply(lambda x: ", ".join(sorted(set(x.tolist()))))
    .rename("sources")
    .reset_index()
)

basin_events = flood_dates_agg.merge(n_floodmaps_agg, on="HYBAS_ID", how="left")
basin_events = basin_events.merge(sources_agg, on="HYBAS_ID")
used_hybas_ids = set(basin_events["HYBAS_ID"])

# ── Step 4: read basin geometries, deduplicate by HYBAS_ID ───────────────────
# Each basin (polygon + static attributes) only needs to be stored once even
# though the same basin appears across many event files.
print("Reading basin geometries...")
seen_ids: set = set()
basin_rows = []

for src in SOURCES:
    src_dir = LUMPED_BASINS / src
    files = sorted(src_dir.glob("*.gpkg"))
    for fpath in tqdm(files, desc=f"  {src}"):
        gdf = gpd.read_file(fpath)
        gdf = gdf[gdf["HYBAS_ID"].isin(used_hybas_ids) & ~gdf["HYBAS_ID"].isin(seen_ids)]
        if not gdf.empty:
            basin_rows.append(gdf)
            seen_ids.update(gdf["HYBAS_ID"].tolist())

basins_gdf = pd.concat(
    [df.dropna(axis=1, how="all") for df in basin_rows if not df.empty],
    ignore_index=True,
)
basins_gdf = gpd.GeoDataFrame(basins_gdf, geometry="geometry", crs="EPSG:4326")

# ── Step 5: merge geometry with flood event aggregation ───────────────────────
print("Merging geometry with flood event data...")
basins_gdf = basins_gdf.merge(basin_events, on="HYBAS_ID", how="inner")

# ── Step 6: drop basins with no parseable flood date ─────────────────────────
before = len(basins_gdf)
basins_gdf = basins_gdf[basins_gdf["n_flood_events"] >= 1].copy()
basins_gdf["n_floodmaps"] = basins_gdf["n_floodmaps"].astype(int)
print(f"Dropped {before - len(basins_gdf)} basins with no flood date; {len(basins_gdf):,} remaining.")

# ── Step 6b: fix invalid geometries with buffer(0) ───────────────────────────
invalid_mask = ~basins_gdf.geometry.is_valid
n_invalid = invalid_mask.sum()
if n_invalid > 0:
    basins_gdf.loc[invalid_mask, "geometry"] = basins_gdf.loc[invalid_mask, "geometry"].buffer(0)
    print(f"Fixed {n_invalid} invalid geometries with buffer(0).")

# ── Step 7: add deluge_id as sequential integer ───────────────────────────────
basins_gdf = basins_gdf.sort_values("HYBAS_ID").reset_index(drop=True)
basins_gdf.insert(0, "deluge_id", range(1, len(basins_gdf) + 1))

# ── Step 8: match basins to CARAVAN gauges by polygon IoU ──────────────────────
# Assign caravan_id := CARAVAN gauge_id when the DELUGE lumped catchment overlaps
# a CARAVAN basin with IoU >= CARAVAN_MIN_IOU; NaN otherwise.
print("Matching basins to CARAVAN gauges...")
caravan_parts = []
for sub in CARAVAN_SUBDATASETS:
    shp = CARAVAN_SHAPEFILES / sub / f"{sub}_basin_shapes.shp"
    gdf = gpd.read_file(shp)[["gauge_id", "geometry"]]
    caravan_parts.append(gdf)
    print(f"  {sub}: {len(gdf)} basins")
caravan = gpd.GeoDataFrame(
    pd.concat(caravan_parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
)

# Project both to equal-area CRS for area-based IoU
caravan_proj = caravan.to_crs(CARAVAN_PROJ_CRS).copy()
caravan_proj["caravan_area"] = caravan_proj.geometry.area

basins_proj = (
    basins_gdf[["HYBAS_ID", "geometry"]].to_crs(CARAVAN_PROJ_CRS).copy()
)
basins_proj["hybas_area"] = basins_proj.geometry.area
basins_proj_indexed = basins_proj.set_index("HYBAS_ID")

# Candidate pairs via spatial-index join, then exact IoU per pair
candidates = gpd.sjoin(
    caravan_proj[["gauge_id", "caravan_area", "geometry"]],
    basins_proj_indexed[["hybas_area", "geometry"]].reset_index(),
    how="inner",
    predicate="intersects",
)
print(f"  Candidate pairs: {len(candidates)}")

basins_geom_map = basins_proj_indexed["geometry"]
ious = []
for c_geom, row in tqdm(
    zip(candidates.geometry, candidates.itertuples()),
    total=len(candidates),
    desc="  computing IoU",
):
    b_geom = basins_geom_map.loc[row.HYBAS_ID]
    intersection = c_geom.intersection(b_geom).area
    union = row.caravan_area + row.hybas_area - intersection
    ious.append(intersection / union if union > 0 else 0.0)
candidates["iou"] = ious

matches = candidates[candidates["iou"] >= CARAVAN_MIN_IOU]
# Keep the best match per HYBAS_ID if any one-to-many ambiguity remains
matches = (
    matches.sort_values("iou", ascending=False)
    .drop_duplicates("HYBAS_ID", keep="first")[["HYBAS_ID", "gauge_id"]]
)
print(f"  Matches at IoU >= {CARAVAN_MIN_IOU}: {len(matches)}")

basins_gdf = basins_gdf.merge(
    matches.rename(columns={"gauge_id": "caravan_id"}), on="HYBAS_ID", how="left"
)

# ── Step 9: set HYBAS_ID as index and fix column order ────────────────────────
basins_gdf = basins_gdf.rename(columns={"grdc_no": "grdc_id"}).drop(columns=["grdc_iou"])
cols = [
    "deluge_id", "caravan_id", "grdc_id", "NEXT_DOWN", "headwater", "n_upstream",
    "basin_area_m2", "basin_area_km2",
    "n_flood_events", "n_floodmaps", "flood_event_dates", "sources", "geometry",
]
basins_gdf = basins_gdf.set_index("HYBAS_ID")[cols]
basins_gdf.index.name = "HYBAS_ID"

# ── Step 10: save as GeoParquet ───────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving {len(basins_gdf):,} basins → {OUTPUT_PATH}")
basins_gdf.to_parquet(OUTPUT_PATH)
print("Done.")
