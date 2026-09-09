import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

INPUT_PATH = Path(CFG["deluge_basins_geoparquet"])

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"

issues = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        issues.append(label)


def split_csv(value) -> list[str]:
    """Split a comma-separated string into a list of trimmed tokens."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str):
        return list(value)  # already a sequence
    return [tok.strip() for tok in value.split(",") if tok.strip()]


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\nLoading {INPUT_PATH} ...")
gdf = gpd.read_parquet(INPUT_PATH)
print(f"{INFO} {len(gdf):,} basins, {len(gdf.columns)} columns")
print(f"{INFO} columns: {list(gdf.columns)}\n")

dates_list = gdf["flood_event_dates"].apply(split_csv)
sources_list = gdf["sources"].apply(split_csv)

# ── 1. Index ──────────────────────────────────────────────────────────────────
print("── 1. Index ──")
check("HYBAS_ID is the index", gdf.index.name == "HYBAS_ID")
check("No duplicate HYBAS_IDs", gdf.index.is_unique,
      f"{gdf.index.duplicated().sum()} duplicates" if not gdf.index.is_unique else "")
check("deluge_id is unique", gdf["deluge_id"].is_unique)
check("deluge_id runs 1 … n",
      gdf["deluge_id"].min() == 1 and gdf["deluge_id"].max() == len(gdf),
      f"min={gdf['deluge_id'].min()}, max={gdf['deluge_id'].max()}, n={len(gdf)}")

# ── 2. Geometry ───────────────────────────────────────────────────────────────
print("\n── 2. Geometry ──")
null_geom = gdf.geometry.isna().sum()
check("No null geometries", null_geom == 0, f"{null_geom} null")

empty_geom = gdf.geometry.is_empty.sum()
check("No empty geometries", empty_geom == 0, f"{empty_geom} empty")

invalid_geom = (~gdf.geometry.is_valid).sum()
check("All geometries valid", invalid_geom == 0, f"{invalid_geom} invalid")

check("CRS is EPSG:4326", gdf.crs.to_epsg() == 4326, f"EPSG:{gdf.crs.to_epsg()}")

bounds = gdf.geometry.total_bounds  # [minx, miny, maxx, maxy]
check("Longitudes within ±180", bounds[0] >= -180 and bounds[2] <= 180,
      f"lon [{bounds[0]:.2f}, {bounds[2]:.2f}]")
check("Latitudes within ±90", bounds[1] >= -90 and bounds[3] <= 90,
      f"lat [{bounds[1]:.2f}, {bounds[3]:.2f}]")

# ── 3. Flood event consistency ────────────────────────────────────────────────
print("\n── 3. Flood event consistency ──")
len_dates = dates_list.apply(len)
mismatch = (len_dates != gdf["n_flood_events"]).sum()
check("len(flood_event_dates) == n_flood_events for all rows", mismatch == 0,
      f"{mismatch} mismatches")

zero_events = (gdf["n_flood_events"] < 1).sum()
check("n_flood_events >= 1 for all rows", zero_events == 0,
      f"{zero_events} rows with 0 events")

not_sorted = dates_list.apply(lambda d: d != sorted(d)).sum()
check("flood_event_dates are sorted chronologically", not_sorted == 0,
      f"{not_sorted} rows with unsorted dates")

empty_sources = sources_list.apply(len).eq(0).sum()
check("sources is never empty", empty_sources == 0,
      f"{empty_sources} rows with empty sources")

# ── 4. Date validity ──────────────────────────────────────────────────────────
print("\n── 4. Date validity ──")
all_dates = dates_list.explode().dropna()
parsed = pd.to_datetime(all_dates, format="%Y-%m-%d", errors="coerce")
bad_dates = parsed.isna().sum()
check("All dates parse as YYYY-MM-DD", bad_dates == 0,
      f"{bad_dates} unparseable dates")

valid = parsed.dropna()
if len(valid):
    min_date, max_date = valid.min(), valid.max()
    check("Earliest date >= 1981-01-01", min_date >= pd.Timestamp("1981-01-01"),
          f"earliest: {min_date.date()}")
    check("Latest date <= today", max_date <= pd.Timestamp.today(),
          f"latest: {max_date.date()}")
    print(f"{INFO} Date range: {min_date.date()} → {max_date.date()}")
    print(f"{INFO} Total date tokens: {len(valid):,}")

# ── 5. Area validation ────────────────────────────────────────────────────────
print("\n── 5. Area validation ──")
check("basin_area_m2 > 0 for all rows", (gdf["basin_area_m2"] > 0).all(),
      f"{(gdf['basin_area_m2'] <= 0).sum()} non-positive values")

check("basin_area_km2 ≈ basin_area_m2 / 1e6",
      ((gdf["basin_area_m2"] / 1e6 - gdf["basin_area_km2"]).abs() < 0.01).all(),
      "m2/km2 columns disagree beyond 0.01 km²")

# Reproject to equal-area and compare with stored basin_area_km2
gdf_ea = gdf.to_crs("EPSG:6933")
computed_km2 = gdf_ea.geometry.area / 1e6
ratio = computed_km2 / gdf["basin_area_km2"]
# HydroSHEDS area is pre-computed along the DEM; allow up to 2× deviation
large_deviation = ((ratio < 0.5) | (ratio > 2.0)).sum()
check("Computed area within 2× of stored basin_area_km2", large_deviation == 0,
      f"{large_deviation} basins with >2× deviation")
print(f"{INFO} Median area ratio (computed/stored): {ratio.median():.3f}")

# ── 6. Null / missing values ──────────────────────────────────────────────────
print("\n── 6. Null / missing values ──")
always_present = ["deluge_id", "NEXT_DOWN", "headwater", "n_upstream",
                  "basin_area_m2", "basin_area_km2", "n_flood_events",
                  "n_floodmaps", "flood_event_dates", "sources", "geometry"]
for col in always_present:
    null_n = gdf[col].isna().sum()
    check(f"No nulls in '{col}'", null_n == 0, f"{null_n} nulls")

for col in ["grdc_id", "caravan_id"]:
    null_rate = gdf[col].isna().mean() * 100
    n_present = gdf[col].notna().sum()
    print(f"{INFO} '{col}' null rate: {null_rate:.1f}% ({n_present:,} basins have it)")

# ── 7. NEXT_DOWN referential integrity (informational) ───────────────────────
print("\n── 7. NEXT_DOWN (informational) ──")
has_downstream = gdf["NEXT_DOWN"] != 0
n_outlets = (~has_downstream).sum()
n_with_down = int(has_downstream.sum())
in_dataset = gdf.loc[has_downstream, "NEXT_DOWN"].isin(gdf.index).sum()
print(f"{INFO} {n_outlets:,} outlet basins (NEXT_DOWN == 0)")
if n_with_down:
    print(f"{INFO} {n_with_down:,} basins with a downstream basin; "
          f"{in_dataset:,} ({in_dataset/n_with_down*100:.1f}%) of those are also in this dataset")

# ── 8. Summary statistics ─────────────────────────────────────────────────────
print("\n── 8. Summary statistics ──")
print(f"{INFO} Basins total:        {len(gdf):,}")
print(f"{INFO} Headwater basins:    {gdf['headwater'].sum():,} ({gdf['headwater'].mean()*100:.1f}%)")
print(f"{INFO} n_flood_events:      "
      f"min={gdf['n_flood_events'].min()}, "
      f"median={gdf['n_flood_events'].median():.0f}, "
      f"max={gdf['n_flood_events'].max()}, "
      f"mean={gdf['n_flood_events'].mean():.1f}")
print(f"{INFO} n_floodmaps:         "
      f"min={gdf['n_floodmaps'].min()}, "
      f"median={gdf['n_floodmaps'].median():.0f}, "
      f"max={gdf['n_floodmaps'].max()}, "
      f"mean={gdf['n_floodmaps'].mean():.1f}")
print(f"{INFO} basin_area_km2:      "
      f"min={gdf['basin_area_km2'].min():.1f}, "
      f"median={gdf['basin_area_km2'].median():.1f}, "
      f"max={gdf['basin_area_km2'].max():.1f}")

# Individual sensor occurrences (a source string can list several sensors, with repeats)
sensor_counts = sources_list.explode().value_counts()
print(f"{INFO} Sensor occurrences (across all rows, counted with repeats):")
for sensor, count in sensor_counts.items():
    print(f"       {sensor}: {count:,}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
if issues:
    print(f"FAILED {len(issues)} check(s):")
    for i in issues:
        print(f"  • {i}")
else:
    print("All checks passed.")
