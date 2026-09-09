"""
Validate DELUGE meteorology (ERA5-Land aggregated by aggregate_ERA5Land.py) against
the CARAVAN reference on the shared basins.

Basin selection
---------------
Uses basins.geoparquet rows with a non-null `caravan_id` column — i.e. the same
subset produced by `aggregate_ERA5Land.py --limit-basins caravan`. Each such row
carries the CARAVAN gauge id (e.g. "lamah_207407"), which is used to locate the
matching CARAVAN netCDF at
    /home/ok2907/Documents/Data/Caravan/Caravan-nc/timeseries/netcdf/<subdataset>/<gauge_id>.nc

Variables
---------
DELUGE uses raw ERA5-Land variable names (single daily aggregate per variable).
CARAVAN exposes daily min/mean/max (or *_sum for accumulations). We compare each
DELUGE variable against the CARAVAN aggregate most likely to be equivalent
(_mean for state variables, _sum for accumulated fluxes). See VAR_MAP.

Metrics
-------
Per (basin, variable), on the intersection of finite-valued days within the
shared date range:
    RMSE  = sqrt(mean((deluge - caravan)^2))
    PBIAS = 100 * sum(deluge - caravan) / sum(caravan)      [%]
    n     = number of paired finite days
    mean_deluge, mean_caravan

Outputs
-------
- per-basin, per-variable long table:
      /home/ok2907/Documents/Data/DELUGE/C_Validation/meteorology_val.csv
- summary per variable (median RMSE, median PBIAS, n basins):
      /home/ok2907/Documents/Data/DELUGE/C_Validation/meteorology_val_summary.csv
- combined nRMSE-vs-PBIAS scatter + per-variable |PBIAS| spider grid:
      /home/ok2907/Documents/Data/DELUGE/C_Validation/meteorology_val_nrmse_pbias_spider.png

Notes
-----
- No unit conversion is applied. DELUGE and CARAVAN both derive from ERA5-Land
  and are expected to share units; if a variable shows a systematic ~1000x or
  sign offset, that indicates a mm-vs-m or convention mismatch to investigate.
- DELUGE aggregate_ERA5Land.py sign-flips `total_evaporation` and
  `potential_evaporation` (positive out of surface). CARAVAN
  `potential_evaporation_sum_ERA5_LAND` keeps the raw ERA5-Land sign (negative
  for evaporative loss). Expect a large PBIAS for potential_evaporation unless
  this is reconciled downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402


# ── Paths ─────────────────────────────────────────────────────────────────────
BASINS_PATH   = Path(CFG["deluge_basins_geoparquet"])
DELUGE_ZARR   = Path(CFG["deluge_meteorology_zarr"])
CARAVAN_NC    = Path(CFG["caravan_nc_dir"])
OUT_DIR       = Path(CFG["validation_output_dir"])

DATE_EPOCH = pd.Timestamp("1951-01-01")  # CARAVAN nc `date` variable epoch

# Variables whose values are in millimetres (either flagged in
# download_6hourly.MM_VARS or converted to mm via aggregate_ERA5Land.UNIT_SCALE).
# Everything expressed in mm is rounded to whole millimetres in the validation
# outputs; all other variables (including unitless correlation, %, K, kPa, m/s,
# W m^-2) get 2 decimals.
MM_VARIABLES = frozenset({
    "snowfall",
    "snowmelt",
    "potential_evaporation",
    "total_evaporation",
    "total_precipitation",
    "snow_depth_water_equivalent",
})


def _decimals_for(var: str) -> int:
    """0 for mm-valued variables, 2 otherwise."""
    return 0 if var in MM_VARIABLES else 2


# Rounding threshold — matches N_DECIMALS=2 in aggregate_ERA5Land.py. Daily
# pairs whose absolute difference is at or below this value are treated as
# rounding noise and excluded from metric computation.
NEGLIGIBLE_DIFF = 0.01

# Per-variable signal threshold for PBIAS. When a value is set, PBIAS is
# computed only over days where max(DELUGE, CARAVAN) >= threshold — i.e. days
# with a physically meaningful signal in at least one of the two products.
# Rationale for SWE: for most CARAVAN basins there is no snow for years at a
# time, and the tiny residual differences around zero (fractions of a mm)
# used to dominate the aggregate PBIAS. Requiring at least 1 mm of SWE on
# either side restricts the metric to actual snow days. RMSE, correlation
# and std_ratio still use the full paired sample.
SIGNAL_THRESHOLD: dict[str, float] = {
    "snow_depth_water_equivalent": 1.0,   # mm
}


# DELUGE var  →  CARAVAN var
VAR_MAP = {
    "10m_u_component_of_wind":      "u_component_of_wind_10m_mean",
    "10m_v_component_of_wind":      "v_component_of_wind_10m_mean",
    "2m_dewpoint_temperature":      "dewpoint_temperature_2m_mean",
    "2m_temperature":               "temperature_2m_mean",
    "potential_evaporation":        "potential_evaporation_sum_ERA5_LAND",
    "snow_depth_water_equivalent":  "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation":  "surface_net_solar_radiation_mean",
    "surface_net_thermal_radiation": "surface_net_thermal_radiation_mean",
    "surface_pressure":             "surface_pressure_mean",
    "total_precipitation":          "total_precipitation_sum",
    "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1_mean",
    "volumetric_soil_water_layer_2": "volumetric_soil_water_layer_2_mean",
    "volumetric_soil_water_layer_3": "volumetric_soil_water_layer_3_mean",
    "volumetric_soil_water_layer_4": "volumetric_soil_water_layer_4_mean",
}


def load_caravan_series(gauge_id: str, caravan_vars: list[str]) -> pd.DataFrame:
    """Read a CARAVAN gauge nc file into a date-indexed DataFrame of requested vars.

    Returns an empty DataFrame if the file is missing or none of the requested
    variables exist.
    """
    subdataset = gauge_id.split("_")[0]
    nc_path = CARAVAN_NC / subdataset / f"{gauge_id}.nc"
    if not nc_path.exists():
        return pd.DataFrame()

    ds = netCDF4.Dataset(str(nc_path), "r")
    try:
        date_days = ds.variables["date"][:].filled(np.nan).astype(np.float64)
        dates = DATE_EPOCH + pd.to_timedelta(date_days, unit="D")
        cols = {}
        for v in caravan_vars:
            if v in ds.variables:
                cols[v] = ds.variables[v][:].filled(np.nan).astype(np.float64)
    finally:
        ds.close()

    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))


def compute_metrics(
    deluge: np.ndarray,
    caravan: np.ndarray,
    pbias_threshold: float | None = None,
) -> dict:
    """Pairwise RMSE, PBIAS, correlation, and std ratio over finite entries only.

    `corr` and `std_ratio` are added so a Taylor diagram can be built directly
    from per-basin, per-variable rows without a second pass through the data.

    On days where |DELUGE − CARAVAN| ≤ NEGLIGIBLE_DIFF the pointwise difference
    is treated as exactly zero when forming RMSE and PBIAS. DELUGE values are
    written by aggregate_ERA5Land.py with N_DECIMALS=2, so disagreement inside
    that rounding threshold is noise — for SWE in tropical basins those days
    used to dominate the sums and produced PBIAS of −60 to −70 % on essentially
    identical (near-zero) series. `n` is the full paired-day count (nothing is
    dropped), and correlation / std_ratio use the original values.

    When `pbias_threshold` is provided, PBIAS is restricted to days where
    max(DELUGE, CARAVAN) ≥ threshold — i.e. only days with a physically
    meaningful signal contribute. `n_pbias` in the result is the number of
    days that survived the threshold. RMSE, correlation and std_ratio are
    still computed over the full paired sample.
    """
    m = np.isfinite(deluge) & np.isfinite(caravan)
    n = int(m.sum())
    if n == 0:
        return dict(n=0, n_pbias=0, rmse=np.nan, pbias=np.nan,
                    corr=np.nan, std_ratio=np.nan,
                    mean_deluge=np.nan, mean_caravan=np.nan)
    d = deluge[m]
    c = caravan[m]
    diff = d - c
    diff = np.where(np.abs(diff) <= NEGLIGIBLE_DIFF, 0.0, diff)
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if pbias_threshold is not None:
        signal_mask = np.maximum(d, c) >= float(pbias_threshold)
        n_pbias = int(signal_mask.sum())
        if n_pbias == 0:
            # Neither product ever exceeds the threshold — both series agree
            # at ~0 throughout the window. That's perfect agreement, not
            # missing data, so record it as PBIAS = 0 rather than NaN.
            pbias = 0.0
        else:
            c_sig = c[signal_mask]
            denom = float(np.sum(c_sig))
            pbias = (float(100.0 * np.sum(diff[signal_mask]) / denom)
                     if denom != 0.0 else np.nan)
    else:
        n_pbias = n
        denom = float(np.sum(c))
        pbias = float(100.0 * np.sum(diff) / denom) if denom != 0.0 else np.nan

    std_d = float(np.std(d))
    std_c = float(np.std(c))
    if std_d > 0 and std_c > 0:
        corr = float(np.corrcoef(d, c)[0, 1])
        std_ratio = std_d / std_c
    else:
        corr = np.nan
        std_ratio = np.nan
    return dict(
        n=n,
        n_pbias=n_pbias,
        rmse=rmse,
        pbias=pbias,
        corr=corr,
        std_ratio=std_ratio,
        mean_deluge=float(np.mean(d)),
        mean_caravan=float(np.mean(c)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Basins with a caravan match ──────────────────────────────────────────
    print(f"Loading basins from {BASINS_PATH} ...")
    basins = gpd.read_parquet(BASINS_PATH)
    matched = basins[basins["caravan_id"].notna()].copy()
    print(f"  {len(matched)} basins with a caravan_id")

    # ── DELUGE meteorology zarr ──────────────────────────────────────────────
    print(f"Opening DELUGE meteorology zarr {DELUGE_ZARR} ...")
    dds = xr.open_zarr(str(DELUGE_ZARR), consolidated=None, chunks={})
    deluge_basins = set(map(int, dds.basin.values.tolist()))

    # Restrict to variables actually present in both sides of the mapping.
    var_pairs = [(dv, cv) for dv, cv in VAR_MAP.items() if dv in dds.data_vars]
    missing_deluge = [dv for dv in VAR_MAP if dv not in dds.data_vars]
    if missing_deluge:
        print(f"  WARN: DELUGE zarr is missing mapped vars: {missing_deluge}")
    print(f"  shared DELUGE↔CARAVAN variables to compare: {len(var_pairs)}")

    # Restrict basins to those present in the DELUGE output (in case aggregation
    # was run on a smaller subset than the full caravan set).
    matched = matched[matched.index.isin(deluge_basins)]
    print(f"  {len(matched)} basins are also present in the DELUGE zarr")
    if len(matched) == 0:
        print("Nothing to compare. Did aggregate_ERA5Land.py --limit-basins caravan "
              "run and write meteorology.zarr for these basins?")
        return

    deluge_date_min = pd.Timestamp(dds.date.values[0])
    deluge_date_max = pd.Timestamp(dds.date.values[-1])
    print(f"  DELUGE date span: {deluge_date_min.date()} → {deluge_date_max.date()}")

    caravan_vars = [cv for _, cv in var_pairs]

    # ── Per basin, per variable metrics ──────────────────────────────────────
    rows = []
    for hybas_id, brow in tqdm(matched.iterrows(), total=len(matched), desc="basins"):
        gauge_id = str(brow["caravan_id"])
        cdf = load_caravan_series(gauge_id, caravan_vars)
        if cdf.empty:
            print(f"  WARN: no CARAVAN data for {gauge_id} (HYBAS_ID {hybas_id})")
            continue

        # Restrict to intersecting date range.
        lo = max(deluge_date_min, cdf.index.min())
        hi = min(deluge_date_max, cdf.index.max())
        if lo > hi:
            continue

        cdf = cdf.loc[lo:hi]
        ddf = (
            dds.sel(basin=int(hybas_id), date=slice(lo, hi))
               .to_dataframe()[list(dict.fromkeys(dv for dv, _ in var_pairs))]
        )
        # Align on shared dates (inner join).
        joined = ddf.join(cdf, how="inner", lsuffix="_deluge", rsuffix="_caravan")

        for dv, cv in var_pairs:
            if cv not in cdf.columns:
                continue
            d_arr = joined[dv].to_numpy(dtype=np.float64)
            c_arr = joined[cv].to_numpy(dtype=np.float64)
            m = compute_metrics(
                d_arr, c_arr,
                pbias_threshold=SIGNAL_THRESHOLD.get(dv),
            )
            rows.append({
                "HYBAS_ID":     int(hybas_id),
                "caravan_id":   gauge_id,
                "variable":     dv,
                "caravan_var":  cv,
                "date_start":   lo.date().isoformat(),
                "date_end":     hi.date().isoformat(),
                **m,
            })

    if not rows:
        print("No paired series to score. Exiting.")
        return

    per_basin = pd.DataFrame(rows)
    # Round metric columns per variable: 0 decimals for mm-valued variables,
    # 2 decimals otherwise. `pbias`, `corr` and `std_ratio` are unitless and
    # always get 2 decimals.
    per_basin = _round_per_basin(per_basin)
    per_basin_path = OUT_DIR / "meteorology_val.csv"
    per_basin.to_csv(per_basin_path, index=False)
    print(f"\nWrote per-basin metrics: {per_basin_path}  ({len(per_basin)} rows)")

    # ── Summary per variable ─────────────────────────────────────────────────
    summary = (
        per_basin
        .groupby("variable", as_index=False)
        .agg(
            n_basins=("HYBAS_ID", "nunique"),
            median_rmse=("rmse", "median"),
            median_pbias=("pbias", "median"),
            mean_pbias=("pbias", "mean"),
            median_n_days=("n", "median"),
        )
        .sort_values("variable")
    )
    summary = _round_summary(summary)
    summary_path = OUT_DIR / "meteorology_val_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    combined_path = OUT_DIR / "meteorology_val_nrmse_pbias_spider.png"
    plot_nrmse_pbias_and_spider(per_basin, combined_path)
    print(f"Wrote combined scatter+spider: {combined_path}")

    dds.close()


def _round_per_basin(df: pd.DataFrame) -> pd.DataFrame:
    """Round `rmse`, `mean_deluge`, `mean_caravan` to the variable's native
    precision (0 dp for mm, 2 dp otherwise) and round unitless columns to 2 dp.
    """
    df = df.copy()
    dp = df["variable"].map(_decimals_for).astype(int)
    for col in ("rmse", "mean_deluge", "mean_caravan"):
        if col in df.columns:
            df[col] = [
                round(v, d) if pd.notna(v) else v
                for v, d in zip(df[col].to_numpy(), dp.to_numpy())
            ]
    for col in ("pbias", "corr", "std_ratio"):
        if col in df.columns:
            df[col] = df[col].round(2)
    return df


def _round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Round summary metrics per variable — median_rmse follows the variable's
    unit convention; percent / count columns get 2 dp / integer as appropriate.
    """
    df = df.copy()
    if "median_rmse" in df.columns:
        df["median_rmse"] = [
            round(v, _decimals_for(var)) if pd.notna(v) else v
            for v, var in zip(df["median_rmse"].to_numpy(), df["variable"].to_numpy())
        ]
    for col in ("median_pbias", "mean_pbias"):
        if col in df.columns:
            df[col] = df[col].round(2)
    if "median_n_days" in df.columns:
        # `median` can produce half-integers (e.g. 8950.5) that Int64 refuses
        # to cast as "non-equivalent" — round to nearest whole day first.
        df["median_n_days"] = df["median_n_days"].round().astype("Int64")
    return df


# Ordered category rows for the spider plot. Each inner list becomes one row
# of subplots (variables that don't appear in the validation data are silently
# skipped). Keeping this at module scope so it's easy to tweak.
SPIDER_ROWS: list[list[str]] = [
    # Water fluxes and stores
    ["total_precipitation", "potential_evaporation", "snow_depth_water_equivalent"],
    # Near-surface state
    ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure"],
    # Radiation
    ["surface_net_solar_radiation", "surface_net_thermal_radiation"],
    # Wind
    ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    # Soil moisture
    ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
     "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4"],
]


def _variable_colors(per_basin: pd.DataFrame) -> dict:
    """Shared variable→color mapping ordered by SPIDER_ROWS so left and right
    panels of the combined plot always agree on colour per variable."""
    available = list(dict.fromkeys(per_basin["variable"].tolist()))
    ordered: list[str] = []
    for row in SPIDER_ROWS:
        for v in row:
            if v in available and v not in ordered:
                ordered.append(v)
    for v in available:
        if v not in ordered:
            ordered.append(v)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(ordered), 2)))
    return {v: palette[i] for i, v in enumerate(ordered)}


def _spider_layout(per_basin: pd.DataFrame):
    """Return (rows, nrows, ncols) for the spider grid, or (None, 0, 0) if
    there is nothing to draw."""
    available = set(per_basin["variable"].unique())
    rows = [[v for v in row if v in available] for row in SPIDER_ROWS]
    rows = [r for r in rows if r]
    if not rows:
        return None, 0, 0
    return rows, len(rows), max(len(r) for r in rows)


def _draw_pbias_spider_into_axes(
    axes_2d,
    rows: list[list[str]],
    ncols: int,
    per_basin: pd.DataFrame,
    var_color: dict,
    text_scale: float,
) -> None:
    """Draw the |PBIAS| spider into an existing 2D array of polar axes."""
    for r, row_vars in enumerate(rows):
        for c in range(ncols):
            ax = axes_2d[r, c]
            if c >= len(row_vars):
                ax.set_visible(False)
                continue
            var = row_vars[c]
            sub = (per_basin[per_basin["variable"] == var]
                   .sort_values("HYBAS_ID")
                   .reset_index(drop=True))
            if sub.empty:
                ax.set_visible(False)
                continue

            color = var_color[var]
            pbias_abs = np.abs(sub["pbias"].to_numpy(dtype=float))
            r_display = np.clip(100.0 - pbias_abs, 0.0, 100.0)
            theta = np.linspace(0.0, 2.0 * np.pi, len(sub), endpoint=False)
            theta_c = np.concatenate([theta, theta[:1]])
            r_c = np.concatenate([r_display, r_display[:1]])

            ax.plot(theta_c, r_c, color=color, linewidth=1.1)
            ax.fill(theta_c, r_c, color=color, alpha=0.3)

            ax.set_ylim(0.0, 100.0)
            ax.set_rticks([0, 25, 50, 75, 100])
            ax.set_yticklabels([""] * 5)
            ax.set_rlabel_position(90)
            ax.set_xticks([])
            ax.set_title(var, fontsize=9 * text_scale, pad=6)
            ax.grid(True, alpha=0.4, linewidth=0.5)


def _draw_nrmse_vs_pbias(
    ax,
    per_basin: pd.DataFrame,
    *,
    var_color: dict | None = None,
    text_scale: float = 1.0,
    point_size: float = 60.0,
    square: bool = False,
    title: str | None = "DELUGE vs CARAVAN — normalized RMSE vs PBIAS per variable",
    legend: bool = True,
) -> bool:
    """Draw the median-nRMSE vs median-PBIAS scatter into `ax`.

    Returns True if anything was drawn.
    """
    df = per_basin.copy()
    denom = df["mean_caravan"].abs()
    nrmse = df["rmse"] / denom.where(denom > 0)
    df = df.assign(nrmse=nrmse).dropna(subset=["nrmse", "pbias"])
    if df.empty:
        return False

    agg = (df.groupby("variable")
             .agg(nrmse=("nrmse", "median"), pbias=("pbias", "median"))
             .reset_index()
             .sort_values("variable"))

    if var_color is None:
        var_color = _variable_colors(per_basin)

    ax.axvline(0, color="0.6", linewidth=0.8)
    ax.axhline(0, color="0.6", linewidth=0.8)
    for _, row in agg.iterrows():
        ax.scatter(row["pbias"], row["nrmse"], s=point_size,
                   color=var_color[row["variable"]],
                   edgecolor="k", linewidth=0.4, label=row["variable"])

    # Pad axis limits so annotations placed toward the interior stay inside.
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_pad = 0.08 * (x_hi - x_lo)
    y_pad = 0.08 * (y_hi - y_lo)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_mid = 0.5 * (x_lo + x_hi)
    y_mid = 0.5 * (y_lo + y_hi)

    for _, row in agg.iterrows():
        if not (row["nrmse"] >= 0.1 or abs(row["pbias"]) >= 1):
            continue
        ha = "left" if row["pbias"] < x_mid else "right"
        va = "bottom" if row["nrmse"] < y_mid else "top"
        dx = 4 if ha == "left" else -4
        dy = 4 if va == "bottom" else -4
        ax.annotate(row["variable"], (row["pbias"], row["nrmse"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9 * text_scale)

    ax.set_xlabel("median pBIAS across basins (%)", fontsize=11 * text_scale)
    ax.set_ylabel("Median nRMSE across basins",
                  fontsize=11 * text_scale)
    if title:
        ax.set_title(title, fontsize=13 * text_scale)
    ax.tick_params(axis="both", labelsize=10 * text_scale)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="best", fontsize=11 * text_scale, framealpha=0.9,
                  markerscale=1.2)
    if square:
        ax.set_box_aspect(1)
    return True


def plot_nrmse_pbias_and_spider(per_basin: pd.DataFrame, out_path: Path) -> None:
    """Combined figure — nRMSE vs PBIAS scatter on the left, PBIAS spider grid
    on the right. Uses a nested gridspec directly on the top-level figure
    (subfigures were dropping the absolute-figure text used for the subplot
    titles). Colors per variable are shared between left and right panels."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    rows, spider_nrows, spider_ncols = _spider_layout(per_basin)
    if rows is None:
        print("  no data — skipping combined plot")
        return

    text_scale = 1.0
    fig_h = 7.5
    scatter_w = fig_h  # square left panel
    spider_w = max(7.5, 2.1 * spider_ncols)
    fig_w = scatter_w + spider_w

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Reserve a top strip for the main and subplot titles (top=0.86).
    outer = GridSpec(
        1, 2, figure=fig,
        width_ratios=[scatter_w, spider_w],
        left=0.06, right=0.98, top=0.86, bottom=0.06, wspace=0.05,
    )

    var_color = _variable_colors(per_basin)

    ax_left = fig.add_subplot(outer[0, 0])
    _draw_nrmse_vs_pbias(
        ax_left, per_basin,
        var_color=var_color, text_scale=text_scale,
        point_size=220.0, square=True,
        title=None, legend=False,
    )

    inner = GridSpecFromSubplotSpec(
        spider_nrows, spider_ncols,
        subplot_spec=outer[0, 1],
        wspace=0.10, hspace=0.40,
    )
    spider_axes = np.empty((spider_nrows, spider_ncols), dtype=object)
    for r in range(spider_nrows):
        for c in range(spider_ncols):
            spider_axes[r, c] = fig.add_subplot(inner[r, c], projection="polar")
    _draw_pbias_spider_into_axes(spider_axes, rows, spider_ncols, per_basin,
                                 var_color, text_scale)

    # Nudge the whole spider grid slightly downward so it sits a bit lower
    # under the subplot title without touching the title itself.
    shift = 0.025
    for r in range(spider_nrows):
        for c in range(spider_ncols):
            ax = spider_axes[r, c]
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])

    # Compute x-centers of the two panels in figure coordinates so the subplot
    # titles sit exactly above their respective columns.
    left_pos = outer[0, 0].get_position(fig)
    right_pos = outer[0, 1].get_position(fig)
    x_left = 0.5 * (left_pos.x0 + left_pos.x1)
    x_right = 0.5 * (right_pos.x0 + right_pos.x1)

    fig.text(x_left, 0.90, "nRMSE vs. pBIAS across basins",
             ha="center", va="center", fontsize=13 * text_scale)
    fig.text(x_right, 0.90, "pBIAS per variable per basin",
             ha="center", va="center", fontsize=13 * text_scale)
    fig.suptitle("Validation of Meteorological Variables",
                 fontsize=15 * text_scale, y=0.96, weight = "bold")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
