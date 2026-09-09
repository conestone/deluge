import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
BASINS_PATH     = CFG["deluge_basins_geoparquet"]
INUNDATION_PATH = CFG["deluge_inundation_geoparquet"]

# ── Load data ─────────────────────────────────────────────────────────────────
basins     = gpd.read_parquet(BASINS_PATH)
inundation = gpd.read_parquet(INUNDATION_PATH)

# ── Derive outlet basins ──────────────────────────────────────────────────────
sorted_basins = basins.sort_values("basin_area_km2", ascending=False)
geoms = list(sorted_basins.geometry.values)
ids   = list(sorted_basins.index.values)
tree  = STRtree(geoms)

dominated = set()
for i, (hid, geom) in enumerate(zip(ids, geoms)):
    if hid in dominated:
        continue
    for j in tree.query(geom):
        if j <= i or ids[j] in dominated:
            continue
        try:
            ratio = geoms[j].intersection(geom).area / geoms[j].area
            if ratio > 0.5:
                dominated.add(ids[j])
        except Exception:
            pass

outlets = basins[~basins.index.isin(dominated)].copy()
print(f"{len(outlets)} non-overlapping outlet basins")

# ── Flood event statistics ────────────────────────────────────────────────────
unique_events = inundation.index.get_level_values("event_id").nunique()
unique_dates  = inundation.index.get_level_values("flood_date").nunique()
unique_maps   = inundation.index.droplevel("HYBAS_ID").unique().__len__()

print(f"\nUnique flood events (event_id)       : {unique_events:,}")
print(f"Unique flood dates                   : {unique_dates:,}")
print(f"Unique flood maps (event_id + date)  : {unique_maps:,}")
print(f"Total basin-event records            : {len(inundation):,}")

print("\nBreakdown by source:")
df_all = inundation.reset_index()
summary = pd.concat([
    df_all.groupby("source_data")["event_id"].count().rename("total_records"),
    df_all.groupby("source_data")["event_id"].nunique().rename("unique_events"),
    df_all.groupby("source_data")["HYBAS_ID"].nunique().rename("unique_basins"),
], axis=1)
summary["records_per_event"] = (summary["total_records"] / summary["unique_events"]).round(1)
print(summary.to_string())

print("\nUnique events per year:")
df_year = df_all[["event_id", "flood_date"]].drop_duplicates("event_id").copy()
df_year["year"] = pd.to_datetime(df_year["flood_date"], errors="coerce").dt.year
print(df_year.groupby("year")["event_id"].count().rename("n_events").to_string())

# ── Plot: flood maps per outlet basin ─────────────────────────────────────────
world = gpd.read_file(
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)
world = world[world["SOVEREIGNT"] != "Antarctica"]

vmin = outlets["n_floodmaps"].min()
vmax = outlets["n_floodmaps"].max()
cmap = cm.YlOrRd
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(16, 8))

world.plot(ax=ax, color="#ebebeb", edgecolor="#bbbbbb", linewidth=0.3, zorder=1)

outlets.plot(
    ax=ax,
    column="n_floodmaps",
    cmap=cmap,
    norm=norm,
    alpha=0.6,
    edgecolor="grey",
    zorder=2,
)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                    fraction=0.025, pad=0.02, shrink=0.5)
cbar.set_label("Number of flood maps per basin", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.set_axis_off()
ax.set_title(
    "Global Coverage Of DELUGE",
    # f"Global Coverage Of DELUGE — outlet basins  (n={len(outlets):,} | unique flood maps: {unique_maps:,})",
    fontsize=18, weight = "bold"
)
plt.tight_layout()
plt.savefig("../../plots/outlets_map.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Plot 2×2: inundation statistics ──────────────────────────────────────────
import numpy as np

SOURCE_COLORS = {"GFD": "#00AA5A", "unosat": "#E6641E", "worldfloods": "#4B288C"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DELUGE — Inundation Statistics", fontsize=14, weight="bold", y=1.01)

# ── Top-left: violin — inundation area by data source ────────────────────────
ax = axes[0, 0]
sources     = sorted(inundation["source_data"].unique())
source_data = [np.log10(inundation.loc[(inundation["source_data"] == s) & (inundation["area_km2"] > 0), "area_km2"].values)
               for s in sources]

parts = ax.violinplot(source_data, positions=range(len(sources)),
                      showmedians=True, showextrema=True)
for pc, src in zip(parts["bodies"], sources):
    pc.set_facecolor(SOURCE_COLORS.get(src, "steelblue"))
    pc.set_alpha(1.0)
for key in ("cmedians", "cmaxes", "cmins", "cbars"):
    parts[key].set_color("black")

ax.set_xticks(range(len(sources)))
ax.set_xticklabels(sources, fontsize=10)
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f"$10^{{{t:.0f}}}$" for t in yticks], fontsize=9)
ax.set_xlabel("Data source", fontsize=10)
ax.set_ylabel("Inundation area (km²)", fontsize=10)
ax.set_title("Inundation area by data source", fontsize=11, weight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# ── Top-right: pie — inundated basins per data source ────────────────────────
ax = axes[0, 1]
source_counts = inundation["source_data"].value_counts()
ax.pie(
    source_counts.values,
    labels=source_counts.index,
    colors=[SOURCE_COLORS.get(s, "grey") for s in source_counts.index],
    autopct="%1.1f%%",
    startangle=90,
)
ax.set_title("Inundated basins per data source", fontsize=11, weight="bold")

# ── Bottom-left: stacked bar timeseries — unique events per year by source ────
ax = axes[1, 0]
df_year = (
    inundation.reset_index()[["event_id", "flood_date", "source_data"]]
    .drop_duplicates(subset=["event_id", "flood_date", "source_data"])
)
df_year["year"] = pd.to_datetime(df_year["flood_date"], errors="coerce").dt.year.astype("Int64")
pivot_year = df_year.groupby(["year", "source_data"]).size().unstack(fill_value=0)
all_years = range(int(pivot_year.index.min()), int(pivot_year.index.max()) + 1)
pivot_year = pivot_year.reindex(all_years, fill_value=0)
pivot_year.plot(
    kind="bar", stacked=True, ax=ax,
    color=[SOURCE_COLORS.get(c, "grey") for c in pivot_year.columns],
)
ax.set_xticklabels([str(int(y)) for y in pivot_year.index], rotation=90, fontsize=9)
ax.set_title("Unique flood events per year by source", fontsize=11, weight="bold")
ax.set_ylabel("Number of events", fontsize=10)
ax.set_xlabel("Year", fontsize=10)
ax.legend(title="Data source", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# ── Bottom-right: stacked bar — basins per satellite, sorted high→low ─────────
ax = axes[1, 1]
pivot = (
    inundation
    .groupby(["satellite_source", "source_data"])
    .size()
    .unstack(fill_value=0)
)
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
pivot.plot(
    kind="bar", stacked=True, ax=ax,
    color=[SOURCE_COLORS.get(c, "grey") for c in pivot.columns],
)
ax.set_title("Inundated basins per satellite by data source", fontsize=11, weight="bold")
ax.set_ylabel("Number of basins", fontsize=10)
ax.set_xlabel("Satellite", fontsize=10)
ax.tick_params(axis="x", rotation=90)
ax.set_yscale("log")
ax.legend(title="Data source", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("../../plots/inundation_stats.png", dpi=300, bbox_inches="tight")
plt.show()
