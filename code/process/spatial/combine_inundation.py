import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402

INUNDATION_DIR = Path(CFG["deluge_preproc_dir"]) / "flood_inundation"
BASINS_PATH = Path(CFG["deluge_basins_geoparquet"])
OUTPUT_PATH = Path(CFG["deluge_inundation_geoparquet"])

SOURCES = ["GFD", "Unosat", "WorldFloods"]

# Static basin attributes already stored in basins.geoparquet — drop here to avoid redundancy
DROP_COLS = ["headwater", "n_upstream"]


def extract_event_id(stem: str, source: str) -> str:
    if source == "GFD":
        m = re.match(r"^(DFO_\d+)_From", stem)
        return m.group(1) if m else stem
    elif source == "Unosat":
        m = re.match(r"^(\d+)_", stem)
        return f"UNOSAT_{m.group(1)}" if m else stem
    else:  # WorldFloods
        m = re.match(r"^EMSR(\d+)_", stem)
        return f"EMSR_{m.group(1)}" if m else stem


# ── Step 1: load valid HYBAS_IDs from combined basin file ────────────────────
print("Loading valid basin IDs...")
basins = gpd.read_parquet(BASINS_PATH)
valid_ids = set(basins.index)
print(f"  {len(valid_ids):,} valid basins")

# ── Step 2: read all inundation files, filter to valid basins ─────────────────
print("Reading inundation files...")
parts = []

for src in SOURCES:
    src_dir = INUNDATION_DIR / src
    files = sorted(src_dir.glob("*.gpkg"))
    for fpath in tqdm(files, desc=f"  {src}"):
        gdf = gpd.read_file(fpath)
        gdf = gdf[gdf["HYBAS_ID"].isin(valid_ids)]
        if not gdf.empty:
            gdf = gdf.drop(columns=[c for c in DROP_COLS if c in gdf.columns])
            gdf["event_id"] = extract_event_id(fpath.stem, src)
            parts.append(gdf)

# ── Step 3: concatenate ───────────────────────────────────────────────────────
print("Combining...")
inundation_gdf = pd.concat(parts, ignore_index=True)
inundation_gdf = gpd.GeoDataFrame(inundation_gdf, geometry="geometry", crs="EPSG:4326")
print(f"  {len(inundation_gdf):,} rows before deduplication")

# Drop rows with missing flood_date or satellite_source: they contribute nothing
# usable downstream, and NaN in the dedup key silently breaks drop_duplicates.
before = len(inundation_gdf)
inundation_gdf = inundation_gdf.dropna(subset=["flood_date", "satellite_source"])
dropped = before - len(inundation_gdf)
if dropped:
    print(f"  Dropped {dropped} rows with missing flood_date or satellite_source")

# ── Step 4: deduplicate within the same (HYBAS_ID, flood_date, event_id) ──────
# A basin can appear in multiple tiled/AOI files that map to the same event_id
# (e.g. EMSR_664 from AOI01 and AOI07, or DFO_1725 tiles). Keep the row with
# the highest inundation percentage.
key = ["HYBAS_ID", "flood_date", "event_id"]
inundation_gdf = (
    inundation_gdf
    .sort_values("inundation_basin_percentage", ascending=False)
    .drop_duplicates(subset=key, keep="first")
)
print(f"  {len(inundation_gdf):,} rows after deduplication")

# ── Step 4b: normalise satellite names ───────────────────────────────────────
_NOAA_VIIRS = re.compile(r"VIIRS-NOAA|VIIRS_NOAA|NOO-VIIRS|NOAA/VIIRS|NOAA_VIIRS", re.IGNORECASE)
inundation_gdf["satellite_source"] = inundation_gdf["satellite_source"].str.replace(
    _NOAA_VIIRS, "NOAA-VIIRS", regex=True
)

inundation_gdf["satellite_source"] = inundation_gdf["satellite_source"].str.replace(
    "COSMO-SkyMed Second Generation", "COSMO-SkyMed SG", regex=False
)

_OTHER_TOKEN = re.compile(r"^\s*(35|42|44|58|multi[\s_-]*sensors?)\s*$", re.IGNORECASE)

def _map_satellite(value):
    if not isinstance(value, str):
        return value
    tokens = [_OTHER_TOKEN.sub("Other", t.strip()) for t in value.split(",")]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)

inundation_gdf["satellite_source"] = inundation_gdf["satellite_source"].apply(_map_satellite)

# ── Step 5: fix invalid geometries ───────────────────────────────────────────
invalid_mask = ~inundation_gdf.geometry.is_valid
n_invalid = invalid_mask.sum()
if n_invalid > 0:
    inundation_gdf.loc[invalid_mask, "geometry"] = (
        inundation_gdf.loc[invalid_mask, "geometry"].buffer(0)
    )
    print(f"Fixed {n_invalid} invalid geometries with buffer(0).")

# ── Step 6: set MultiIndex (HYBAS_ID, flood_date, event_id) and sort ─────────
inundation_gdf = inundation_gdf.set_index(key).sort_index()

# ── Step 7: save as GeoParquet ────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving {len(inundation_gdf):,} records → {OUTPUT_PATH}")
inundation_gdf.to_parquet(OUTPUT_PATH)
print("Done.")
