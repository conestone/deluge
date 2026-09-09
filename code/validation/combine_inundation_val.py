import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

INPUT_PATH = Path(CFG["deluge_inundation_geoparquet"])
BASINS_PATH = Path(CFG["deluge_basins_geoparquet"])

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"

EXPECTED_SOURCES = {"GFD", "unosat", "worldfloods"}
EXPECTED_INDEX = ["HYBAS_ID", "flood_date", "event_id"]

issues = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        issues.append(label)


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\nLoading {INPUT_PATH} ...")
gdf = gpd.read_parquet(INPUT_PATH)
print(f"{INFO} {len(gdf):,} rows, {len(gdf.columns)} columns")
print(f"{INFO} columns: {list(gdf.columns)}\n")

print(f"Loading {BASINS_PATH} ...")
basins = gpd.read_parquet(BASINS_PATH)
valid_ids = set(basins.index)
print(f"{INFO} {len(valid_ids):,} valid basins\n")

# ── 1. Index ──────────────────────────────────────────────────────────────────
print("── 1. Index ──")
check(f"Index names are {tuple(EXPECTED_INDEX)}",
      list(gdf.index.names) == EXPECTED_INDEX,
      f"actual: {list(gdf.index.names)}")
check("No duplicate index rows", not gdf.index.duplicated().any(),
      f"{gdf.index.duplicated().sum()} duplicates")

hybas_in_file = set(gdf.index.get_level_values("HYBAS_ID").unique())
not_in_basins = hybas_in_file - valid_ids
check("All HYBAS_IDs exist in basins.geoparquet", len(not_in_basins) == 0,
      f"{len(not_in_basins)} unknown IDs")

n_covered = len(hybas_in_file)
print(f"{INFO} {n_covered:,} of {len(valid_ids):,} basins "
      f"({n_covered/len(valid_ids)*100:.1f}%) have at least one inundation record")

# ── 2. Geometry ───────────────────────────────────────────────────────────────
print("\n── 2. Geometry ──")
null_geom = gdf.geometry.isna().sum()
check("No null geometries", null_geom == 0, f"{null_geom} null")

empty_geom = gdf.geometry.is_empty.sum()
check("No empty geometries", empty_geom == 0, f"{empty_geom} empty")

invalid_geom = (~gdf.geometry.is_valid).sum()
check("All geometries valid", invalid_geom == 0, f"{invalid_geom} invalid")

check("CRS is EPSG:4326", gdf.crs.to_epsg() == 4326, f"EPSG:{gdf.crs.to_epsg()}")

bounds = gdf.geometry.total_bounds
check("Longitudes within ±180", bounds[0] >= -180 and bounds[2] <= 180,
      f"lon [{bounds[0]:.2f}, {bounds[2]:.2f}]")
check("Latitudes within ±90", bounds[1] >= -90 and bounds[3] <= 90,
      f"lat [{bounds[1]:.2f}, {bounds[3]:.2f}]")

# ── 3. Area consistency ───────────────────────────────────────────────────────
print("\n── 3. Area consistency ──")
check("area_m2 > 0 for all rows", (gdf["area_m2"] > 0).all(),
      f"{(gdf['area_m2'] <= 0).sum()} non-positive values")

# area_km2 is stored with 2 decimal places in the source files, so values
# smaller than ~5000 m² round to 0.00 — use area_m2 as ground truth
zero_km2 = (gdf["area_km2"] == 0).sum()
if zero_km2:
    max_m2 = gdf.loc[gdf['area_km2'] == 0, 'area_m2'].max()
    print(f"{INFO} area_km2 == 0: {zero_km2} rows (source precision: 2 dp, "
          f"max area_m2 for these: {max_m2:.0f} m²)")

# area_km2 should approximate area_m2 / 1e6. Source values are stored at 2 dp,
# so allow max(10% relative, 0.005 km² absolute) — the absolute term dominates
# for small areas where 2-dp rounding blows up the relative error.
mask = gdf["area_km2"] > 0
km2_from_m2 = gdf.loc[mask, "area_m2"] / 1e6
abs_dev = (gdf.loc[mask, "area_km2"] - km2_from_m2).abs()
rel_dev = abs_dev / km2_from_m2
bad_unit = ((abs_dev > 0.005) & (rel_dev > 0.10)).sum()
check("area_km2 ≈ area_m2 / 1e6 (max 10% relative or 0.005 km² absolute)",
      bad_unit == 0, f"{bad_unit} inconsistent rows")

# Flood polygon area should be <= basin area (inundation cannot exceed 100%)
over_100 = (gdf["inundation_basin_percentage"] > 100).sum()
check("inundation_basin_percentage <= 100 for all rows", over_100 == 0,
      f"{over_100} rows exceed 100%")

zero_pct = (gdf["inundation_basin_percentage"] == 0).sum()
if zero_pct:
    print(f"{INFO} inundation_basin_percentage == 0: {zero_pct} rows "
          f"(very small floods; rounds to 0.00% at 2 dp)")

# Cross-check inundation % against basin area from basins.geoparquet
basin_area = basins["basin_area_km2"].rename("basin_area_km2")
merged = gdf[["area_km2", "inundation_basin_percentage"]].join(basin_area, on="HYBAS_ID")
recomputed_pct = (merged["area_km2"] / merged["basin_area_km2"]) * 100
deviation = (recomputed_pct - merged["inundation_basin_percentage"]).abs()
large_dev = (deviation > 1.0).sum()
check("inundation_basin_percentage consistent with area_km2 / basin_area (within 1 pp)",
      large_dev == 0, f"{large_dev} rows with >1 percentage point deviation")

# ── 4. Date validity ──────────────────────────────────────────────────────────
print("\n── 4. Date validity ──")
flood_date = gdf.index.get_level_values("flood_date").to_series(index=gdf.index)
null_dates = flood_date.isna().sum()
check("No null flood_date values", null_dates == 0,
      f"{null_dates} null dates in index")

non_null = flood_date.dropna()
parsed = pd.to_datetime(non_null, format="%Y-%m-%d", errors="coerce")
bad_dates = parsed.isna().sum()
check("All flood_dates parse as YYYY-MM-DD", bad_dates == 0,
      f"{bad_dates} unparseable")

valid = parsed.dropna()
if len(valid):
    min_date, max_date = valid.min(), valid.max()
    check("Earliest flood_date >= 1990-01-01",
          min_date >= pd.Timestamp("1990-01-01"),
          f"earliest: {min_date.date()}")
    check("Latest flood_date <= today",
          max_date <= pd.Timestamp.today(),
          f"latest: {max_date.date()}")
    print(f"{INFO} Date range: {min_date.date()} → {max_date.date()}")

# ── 5. Source integrity ───────────────────────────────────────────────────────
print("\n── 5. Source integrity ──")
found_sources = set(gdf["source_data"].dropna().unique())
unexpected = found_sources - EXPECTED_SOURCES
check("source_data contains only expected values", len(unexpected) == 0,
      f"unexpected: {unexpected}")
missing = EXPECTED_SOURCES - found_sources
if missing:
    print(f"{INFO} Expected sources absent from data: {missing}")
print(f"{INFO} Sources present: {sorted(found_sources)}")

print(f"{INFO} Satellite sources: {sorted(gdf['satellite_source'].dropna().unique())}")

# ── 6. Null / missing values ──────────────────────────────────────────────────
print("\n── 6. Null / missing values ──")
for col in ["area_m2", "area_km2", "inundation_basin_percentage",
            "source_data", "satellite_source", "geometry"]:
    null_n = gdf[col].isna().sum()
    check(f"No nulls in '{col}'", null_n == 0, f"{null_n} nulls")

if null_dates > 0:
    null_slice = gdf[flood_date.isna()]
    null_events = null_slice.index.get_level_values("event_id").unique().tolist()
    null_sources = null_slice["source_data"].dropna().unique().tolist()
    print(f"  [WARN] flood_date null in {null_dates} rows from "
          f"{len(null_events)} event(s) ({null_sources}) — no date in source file: "
          f"{null_events}")

# ── 7. Summary statistics ─────────────────────────────────────────────────────
print("\n── 7. Summary statistics ──")
events_per_basin = gdf.groupby(level="HYBAS_ID").size()
print(f"{INFO} Records total:              {len(gdf):,}")
print(f"{INFO} Unique HYBAS_IDs:           {events_per_basin.index.nunique():,}")
print(f"{INFO} Unique event_ids:           {gdf.index.get_level_values('event_id').nunique():,}")
print(f"{INFO} Records per basin:          "
      f"min={events_per_basin.min()}, "
      f"median={events_per_basin.median():.0f}, "
      f"max={events_per_basin.max()}, "
      f"mean={events_per_basin.mean():.1f}")
print(f"{INFO} inundation_basin_%:         "
      f"min={gdf['inundation_basin_percentage'].min():.2f}, "
      f"median={gdf['inundation_basin_percentage'].median():.2f}, "
      f"max={gdf['inundation_basin_percentage'].max():.2f}")
print(f"{INFO} area_km2:                   "
      f"min={gdf['area_km2'].min():.3f}, "
      f"median={gdf['area_km2'].median():.2f}, "
      f"max={gdf['area_km2'].max():.2f}")

rows_per_source = gdf.groupby("source_data").size()
print(f"{INFO} Rows per source_data:")
for src, count in rows_per_source.items():
    print(f"       {src}: {count:,}")

sat_counts = gdf["satellite_source"].value_counts().head(10)
print(f"{INFO} Top satellite_source values:")
for name, count in sat_counts.items():
    print(f"       {name}: {count:,}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
if issues:
    print(f"FAILED {len(issues)} check(s):")
    for i in issues:
        print(f"  • {i}")
else:
    print("All checks passed.")
