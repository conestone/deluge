import sys
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402

# Paths
BASINS_DIR = CFG["deluge_basins_geoparquet"]
INUNDATION_DIR = CFG["deluge_inundation_geoparquet"]

# Filter value: Percentage of Inundated Area in a Basin
P_AREA = 0.1 

print("Reading Basins and Inundation files")

# Read geoparquet files
basins = gpd.read_parquet(BASINS_DIR)
inundation = gpd.read_parquet(INUNDATION_DIR)

# apply a hard inundation area filter to reduce the number of basins
inundation_sf = inundation.loc[inundation["inundation_basin_percentage"] >= P_AREA]
basins_sf_list = inundation_sf.index.get_level_values("HYBAS_ID").unique()
basins_sf = basins[basins.index.isin(basins_sf_list)].copy()
n_basins = basins_sf.shape[0]

# Recompute per-basin flood attributes from the filtered inundation table
inundation_reset = inundation_sf.reset_index()

flood_dates_agg = (
    inundation_reset.groupby("HYBAS_ID")["flood_date"]
    .apply(lambda x: sorted({d for d in x.dropna() if d}))
)
n_flood_events = flood_dates_agg.apply(len)
flood_event_dates = flood_dates_agg.apply(lambda dates: ", ".join(str(d) for d in dates))

n_floodmaps = (
    inundation_reset.dropna(subset=["flood_date"])
    .drop_duplicates(subset=["HYBAS_ID", "event_id", "flood_date"])
    .groupby("HYBAS_ID")
    .size()
)

sources = (
    inundation_reset.groupby("HYBAS_ID")["satellite_source"]
    .apply(lambda x: ", ".join(sorted(set(x.dropna().tolist()))))
)

basins_sf["n_flood_events"] = n_flood_events.reindex(basins_sf.index).fillna(0).astype(int)
basins_sf["n_floodmaps"] = n_floodmaps.reindex(basins_sf.index).fillna(0).astype(int)
basins_sf["flood_event_dates"] = flood_event_dates.reindex(basins_sf.index).fillna("")
basins_sf["sources"] = sources.reindex(basins_sf.index).fillna("")

# Drop basins whose inundation rows had missing flood_date or satellite_source,
# so we don't ship rows with n_flood_events == 0 or empty sources.
bad = (basins_sf["n_flood_events"] == 0) | (basins_sf["sources"] == "")
if bad.any():
    print(f"Dropping {int(bad.sum())} basins with no flood events or no source")
    basins_sf = basins_sf[~bad]
    inundation_sf = inundation_sf[
        inundation_sf.index.get_level_values("HYBAS_ID").isin(basins_sf.index)
    ]

# Recompute n_upstream within the filtered basin set via NEXT_DOWN topology
kept = set(basins_sf.index)
children_of: dict = defaultdict(list)
parent_of: dict = {}
for bid, nd in basins_sf["NEXT_DOWN"].items():
    if nd in kept:
        parent_of[bid] = nd
        children_of[nd].append(bid)

in_degree = {bid: len(children_of[bid]) for bid in basins_sf.index}
upstream_counts = {bid: 0 for bid in basins_sf.index}
queue = deque(bid for bid, deg in in_degree.items() if deg == 0)
while queue:
    bid = queue.popleft()
    parent = parent_of.get(bid)
    if parent is None:
        continue
    upstream_counts[parent] += 1 + upstream_counts[bid]
    in_degree[parent] -= 1
    if in_degree[parent] == 0:
        queue.append(parent)

basins_sf["n_upstream"] = basins_sf.index.map(upstream_counts).astype(int)

# Reassign deluge_id as a dense 1..N sequence over the filtered basins
basins_sf = basins_sf.sort_index()
basins_sf["deluge_id"] = range(1, len(basins_sf) + 1)

n_gauges_sf = basins_sf.loc[basins_sf["caravan_id"]
                            .isna() == False].sort_values("basin_area_km2", ascending=False).shape[0]
n_gauges = basins.loc[basins["caravan_id"]
                      .isna() == False].sort_values("basin_area_km2", ascending=False).shape[0]

n_grdc_sf = basins_sf.loc[basins_sf["grdc_id"].isna()==False].shape[0]
n_grdc = basins.loc[basins["grdc_id"].isna()==False].shape[0]

src_sf = inundation_sf.groupby("source_data").size()
src_all = inundation.groupby("source_data").size()

def pct_kept(label: str) -> str:
    total = int(src_all.get(label, 0))
    kept = int(src_sf.get(label, 0))
    if total == 0:
        return "n/a (0 in original)"
    return f"{kept:,} / {total:,} ({100 * kept / total:.0f} %)"

inundation_sf.to_parquet(INUNDATION_DIR)
basins_sf.to_parquet(BASINS_DIR)

print(
    f"Original Number of Basins: {basins.shape[0]}",
    f"\nFiltered Number of Basins: {n_basins}",
    f"\nSize of filtered Basins: {100* n_basins/basins.shape[0]:.0f} %",
    f"\n\nOriginal Number of Inundation Maps:{inundation.index.shape[0]}",
    f"\nFiltered Number of Inundation Maps:{inundation_sf.index.shape[0]}",
    f"\nSize of filtered Inundation Maps: {100* inundation_sf.index.shape[0]/inundation.index.shape[0]:.0f} %",
    f"\n\nOriginal Number of CARAVAN Gauges:{n_gauges}",
    f"\nFiltered Number of CARAVAN Gauges:{n_gauges_sf}",
    f"\nSize of filtered CARAVAN Gauges: {100* n_gauges_sf/n_gauges:.0f} %",
    f"\n\nOriginal Number of GRDC Gauges:{n_grdc}",
    f"\nFiltered Number of GRDC Gauges:{n_grdc_sf}",
    f"\nSize of filtered GRDC Gauges: {100* n_grdc_sf/n_grdc:.0f} %",
    f"\n\nAmount of filtered GFD:         {pct_kept('GFD')}",
    f"\nAmount of filtered UNOSAT:      {pct_kept('unosat')}",
    f"\nAmount of filtered WorldFloods: {pct_kept('worldfloods')}")