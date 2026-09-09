"""
Spatially aggregate ERA5Land daily raster data into per-basin time series.

Output structure follows the Caravan timeseries.zarr convention
(dims: basin, date; basin-chunked; Blosc-lz4 float32), but variable names
are kept identical to the ERA5Land source (no renaming).

Inputs
------
- Basins:    /home/ok2907/Documents/Data/DELUGE/A_Spatial/basins.geoparquet
             (index name = HYBAS_ID, int64; CRS EPSG:4326)
- ERA5Land:  /home/ok2907/Documents/Data/ERA5Land/ERA5Land_daily/ERA5Land_daily.zarr
             (dims time, latitude, longitude; 0.1 deg global; 17 vars)
- Caravan:   /home/ok2907/Documents/Data/Caravan/HRES/timeseries.zarr  (structural ref only)

Output
------
- /home/ok2907/Documents/Data/DELUGE/B_Timeseries/meteorology.zarr
- Fallback-log CSV alongside output:
  /home/ok2907/Documents/Data/DELUGE/B_Timeseries/meteorology.zarr.fallback_basins.csv

Approach
--------
1. Rasterize each basin individually into its bbox window on the ERA5Land grid,
   producing a sparse (basin_idx, pixel_idx) membership list. Nested basins may
   share pixels — each membership contributes independently to its basin's mean.
   The `basin_idx` is a positional index into `hybas_ids` (the HYBAS_ID column of
   `basins.geoparquet`, which is the public basin identifier in the output).
   Basins with zero pixels under all_touched=False re-rasterize with
   all_touched=True (logged as fallback).
2. For each ERA5Land variable, use dask.array.blockwise over 31-day time chunks
   (native chunk length) to compute per-basin means with np.bincount over the
   sparse memberships:
       sum_i = bincount(basin_idx[finite], weights=values[finite])
       cnt_i = bincount(basin_idx[finite])
       mean_i = sum_i / cnt_i     (NaN cells excluded)
3. Write to Zarr with basin-chunked, date=full, Blosc-lz4-shuffle float32.

Parallelization: dask LocalCluster with 16 single-thread workers. The sparse
membership arrays are captured by closure and shipped once per task graph.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Tell glibc's malloc to return freed memory to the OS immediately instead of
# hoarding it in the allocator arena. Set BEFORE any dask worker process is
# spawned so children inherit it. Without this, large ERA5-Land source blocks
# (~670 MB each) stay resident in the worker's RSS after Python frees them —
# dask sees the mismatch as "Unmanaged memory" and repeatedly pauses/resumes
# workers. Costs a few percent CPU; eliminates the pause/resume churn.
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
import zarr
from zarr.codecs import BloscCodec

from rasterio.features import rasterize
from rasterio.transform import Affine
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths (override via CLI)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402

DEFAULT_BASINS = CFG["deluge_basins_geoparquet"]
DEFAULT_ERA5   = CFG["era5land_daily_zarr"]
DEFAULT_CARAV  = CFG["caravan_hres_zarr"]
DEFAULT_OUT    = CFG["deluge_meteorology_zarr"]

# Output chunking (matches Caravan convention)
BASIN_CHUNK = 128
# Time chunk for the reduction pass. ERA5Land's native time chunk is 31, so use
# 31 to avoid crossing chunk boundaries on the source read.
TIME_CHUNK = 31

# Output time window (inclusive). The source ERA5Land Zarr may extend further at
# either end; we slice to this range before reduction so the output has exactly
# the requested calendar span.
TIME_START = "1999-01-01"
TIME_END   = "2026-05-31"

# ERA5(-Land) reports evaporation as a negative flux (mass leaving the surface).
# For downstream hydrology use we invert the sign so these variables are positive
# in the aggregated output.
SIGN_FLIP_VARS = ("total_evaporation","potential_evaporation")

# Unit conversions applied after the spatial reduction (and after any sign flip).
# ERA5-Land daily accumulations are stored as J m^-2 (radiation) or m (snow depth
# water equivalent, precipitation); pressures are in Pa. We convert to the more
# hydrology-friendly units listed below. Any variable not in this dict is left
# in its native ERA5-Land units.
#   surface_net_solar_radiation / surface_net_thermal_radiation:
#       J m^-2 day^-1  ->  W m^-2   (divide by seconds in a day)
#   snow_depth_water_equivalent:
#       m              ->  mm       (x 1000)
#   surface_pressure:
#       Pa             ->  kPa      (/ 1000)
SECONDS_PER_DAY = 3600 * 24
UNIT_SCALE: dict[str, float] = {
    "surface_net_solar_radiation":   1.0 / SECONDS_PER_DAY,
    "surface_net_thermal_radiation": 1.0 / SECONDS_PER_DAY,
    "snow_depth_water_equivalent":   1000.0,
    "surface_pressure":              1.0 / 1000.0,
}

# All output values are rounded to this many decimal places before being written
# to the zarr store. Applied after sign flips and unit conversions.
N_DECIMALS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_affine(lats: np.ndarray, lons: np.ndarray) -> Affine:
    """Build a rasterio Affine from 1-D lat/lon coordinate arrays (pixel centers).

    Assumes regular spacing. Latitudes may be ascending or descending; the returned
    transform reflects the actual orientation in the arrays so pixel (row, col)
    corresponds to (lats[row], lons[col]).
    """
    dlon = float(lons[1] - lons[0])
    dlat = float(lats[1] - lats[0])
    x0 = float(lons[0]) - dlon / 2.0
    y0 = float(lats[0]) - dlat / 2.0
    return Affine(dlon, 0.0, x0, 0.0, dlat, y0)


def inspect_and_report(basins_path: str, era_path: str, caravan_path: str) -> None:
    print("=" * 78)
    print("INSPECTION SUMMARY")
    print("=" * 78)

    print(f"\n[basins]  {basins_path}")
    gdf = gpd.read_parquet(basins_path)
    print(f"  n basins        : {len(gdf)}")
    print(f"  crs             : {gdf.crs.to_string() if gdf.crs else None}")
    print(f"  index name      : {gdf.index.name} ({gdf.index.dtype})")
    print(f"  geometry types  : {gdf.geometry.type.value_counts().to_dict()}")
    if "basin_area_km2" in gdf.columns:
        a = gdf["basin_area_km2"]
        print(f"  area km^2       : min={a.min():.3g}  med={a.median():.3g}  max={a.max():.3g}")

    print(f"\n[ERA5Land]  {era_path}")
    ds = xr.open_zarr(era_path, consolidated=None, chunks={})
    print(f"  dims            : {dict(ds.sizes)}")
    print(f"  data_vars       : {list(ds.data_vars)}")
    print(f"  lat span        : {float(ds.latitude[0]):+.3f} -> {float(ds.latitude[-1]):+.3f}  "
          f"(step {float(ds.latitude[1]-ds.latitude[0]):+.4f})")
    print(f"  lon span        : {float(ds.longitude[0]):+.3f} -> {float(ds.longitude[-1]):+.3f}  "
          f"(step {float(ds.longitude[1]-ds.longitude[0]):+.4f})")
    v0 = next(iter(ds.data_vars))
    print(f"  chunks (native) : {ds[v0].encoding.get('chunks')}")
    print(f"  time span       : {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}")
    ds.close()

    print(f"\n[Caravan HRES]  {caravan_path}   (structural reference only)")
    dsc = xr.open_zarr(caravan_path, consolidated=None, chunks={})
    print(f"  dims            : {dict(dsc.sizes)}")
    print(f"  data_vars (first 3): {list(dsc.data_vars)[:3]}")
    vc = next(iter(dsc.data_vars))
    enc = dsc[vc].encoding
    print(f"  chunks (var)    : {enc.get('chunks')}")
    print(f"  compressors     : {enc.get('compressors')}")
    print(f"  fill value      : {enc.get('_FillValue')}")
    print(f"  basin dtype     : {dsc.basin.dtype}")
    dsc.close()
    print()


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------
def rasterize_basins_sparse(
    gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a sparse (basin, pixel) membership list on the ERA5Land grid.

    Each basin is rasterized individually into its bounding-box window (not the
    full global grid). `all_touched=False` is used first; basins that get zero
    pixels under that rule re-rasterize with `all_touched=True` and are flagged.

    Nested basins are handled correctly: a pixel that lies inside a child and its
    parent is emitted as two separate (basin, pixel) pairs and contributes to
    both means independently. This is what the previous 2D-ID design could not
    represent — a single raster cell can only hold one basin code, so nested
    children were silently dropped.

    The `basin_idx` returned is the positional index into `hybas_ids` (0..N-1),
    because `np.bincount` needs a contiguous integer domain. The public basin
    identifier — HYBAS_ID — is `hybas_ids[basin_idx]` and is used as the output
    Zarr's `basin` coordinate.

    Returns
    -------
    basin_idx     : int32 (P,) positional index into hybas_ids (0..N-1)
    pixel_idx     : int64 (P,) flat pixel index (row * W + col) into (H, W)
    hybas_ids     : int64 (N,) HYBAS_IDs in gdf order
    used_fallback : bool  (N,) True where basin required all_touched=True
    """
    H, W = len(lats), len(lons)
    N = len(gdf)
    hybas_ids = gdf.index.to_numpy().astype(np.int64)

    dlon = float(lons[1] - lons[0])
    dlat = float(lats[1] - lats[0])
    lon0 = float(lons[0]) - dlon / 2.0
    lat0 = float(lats[0]) - dlat / 2.0

    basin_chunks: list[np.ndarray] = []
    pixel_chunks: list[np.ndarray] = []
    used_fallback = np.zeros(N, dtype=bool)
    n_truly_empty = 0

    print(f"[rasterize] sparse per-basin pass, {N} basins onto {H}x{W} grid ...")
    t0 = time.time()
    bounds_all = gdf.geometry.bounds.to_numpy()  # (N, 4): minx, miny, maxx, maxy
    for i in tqdm(range(N), desc="rasterize"):
        geom = gdf.geometry.iloc[i]
        minx, miny, maxx, maxy = bounds_all[i]

        # Pixel-space bbox window with a 1-cell buffer for edge safety.
        col_lo = int(np.floor((minx - lon0) / dlon)) - 1
        col_hi = int(np.ceil((maxx - lon0) / dlon)) + 1
        if dlat < 0:
            row_lo = int(np.floor((maxy - lat0) / dlat)) - 1
            row_hi = int(np.ceil((miny - lat0) / dlat)) + 1
        else:
            row_lo = int(np.floor((miny - lat0) / dlat)) - 1
            row_hi = int(np.ceil((maxy - lat0) / dlat)) + 1
        col_lo = max(0, col_lo); col_hi = min(W, col_hi)
        row_lo = max(0, row_lo); row_hi = min(H, row_hi)
        if col_hi <= col_lo or row_hi <= row_lo:
            n_truly_empty += 1
            continue

        wh = row_hi - row_lo
        ww = col_hi - col_lo
        win_tr = Affine(dlon, 0.0, lon0 + col_lo * dlon,
                        0.0, dlat,  lat0 + row_lo * dlat)

        r = rasterize(
            shapes=[(geom, 1)],
            out_shape=(wh, ww),
            transform=win_tr,
            fill=0,
            all_touched=False,
            dtype="uint8",
        )
        if not r.any():
            r = rasterize(
                shapes=[(geom, 1)],
                out_shape=(wh, ww),
                transform=win_tr,
                fill=0,
                all_touched=True,
                dtype="uint8",
            )
            used_fallback[i] = True
            if not r.any():
                n_truly_empty += 1
                continue

        rows_local, cols_local = np.where(r == 1)
        rows = rows_local.astype(np.int64) + row_lo
        cols = cols_local.astype(np.int64) + col_lo
        flat = rows * W + cols
        basin_chunks.append(np.full(flat.size, i, dtype=np.int32))
        pixel_chunks.append(flat)

    basin_idx = (np.concatenate(basin_chunks) if basin_chunks
                 else np.zeros(0, dtype=np.int32))
    pixel_idx = (np.concatenate(pixel_chunks) if pixel_chunks
                 else np.zeros(0, dtype=np.int64))

    dt = time.time() - t0
    n_fb = int(used_fallback.sum())
    pair_bytes = basin_idx.nbytes + pixel_idx.nbytes
    print(f"[rasterize] sparse pass done in {dt:.1f}s: "
          f"{basin_idx.size} (basin, pixel) pairs ({pair_bytes/1e6:.1f} MB)")
    print(f"[rasterize] basins using all_touched=True fallback: {n_fb}")
    if n_truly_empty:
        print(f"[rasterize] WARN: {n_truly_empty} basins are off-grid or too small "
              f"to touch any pixel (will emit all-NaN time series)")

    return basin_idx, pixel_idx, hybas_ids, used_fallback


def write_fallback_log(
    log_path: Path,
    gdf: gpd.GeoDataFrame,
    used_fallback: np.ndarray,
) -> None:
    if not used_fallback.any():
        log_path.write_text("HYBAS_ID,basin_area_km2,minx,miny,maxx,maxy\n")
        print(f"[log] no fallback basins; empty log written to {log_path}")
        return
    idx = np.where(used_fallback)[0]
    sub = gdf.iloc[idx].copy()
    bounds = sub.geometry.bounds
    out = pd.DataFrame({
        "HYBAS_ID": sub.index.to_numpy(),
        "basin_area_km2": (sub["basin_area_km2"].to_numpy()
                           if "basin_area_km2" in sub.columns else np.nan),
        "minx": bounds["minx"].to_numpy(),
        "miny": bounds["miny"].to_numpy(),
        "maxx": bounds["maxx"].to_numpy(),
        "maxy": bounds["maxy"].to_numpy(),
    })
    out.to_csv(log_path, index=False)
    print(f"[log] wrote {len(out)} fallback basins to {log_path}")


# ---------------------------------------------------------------------------
# Reduction kernel
# ---------------------------------------------------------------------------
def _sparse_means_block(
    data_block: np.ndarray,   # (T, H, W)  float
    basin_idx: np.ndarray,    # (P,)       int32 in [0, n_basins)
    pixel_idx: np.ndarray,    # (P,)       int64 flat pixel index
    n_basins: int,
) -> np.ndarray:
    """Compute per-basin, per-timestep unweighted means for one time chunk.

    Uses a sparse (basin, pixel) membership list — one pair per (basin, cell)
    membership, so nested basins sharing a pixel each get that pixel counted
    once toward their own mean. Returns (T, n_basins) float32. NaN input cells
    are excluded from mean and count. Basins with zero valid cells for a given
    timestep yield NaN.
    """
    T, H, W = data_block.shape
    out = np.full((T, n_basins), np.nan, dtype=np.float32)
    if basin_idx.size == 0:
        return out
    data_2d = data_block.reshape(T, H * W)
    for t in range(T):
        vals = data_2d[t, pixel_idx]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        b_t = basin_idx[finite]
        v_t = vals[finite].astype(np.float64, copy=False)
        s = np.bincount(b_t, weights=v_t, minlength=n_basins)
        c = np.bincount(b_t, minlength=n_basins)
        with np.errstate(invalid="ignore"):
            m = np.where(c > 0, s / c, np.nan)
        out[t] = m.astype(np.float32)
    return out


def reduce_variable_blockwise(
    darr: da.Array,           # (T, H, W)  chunked (TIME_CHUNK, H, W)
    basin_idx: np.ndarray,    # (P,)
    pixel_idx: np.ndarray,    # (P,)
    n_basins: int,
    sign_flip: bool = False,
    unit_scale: float = 1.0,
    n_decimals: int | None = None,
) -> da.Array:
    """Return (T, N) float32 dask array, chunked (TIME_CHUNK, N).

    Sign flip, unit scaling and rounding are applied inside the same blockwise
    task as the reduction. Doing them as separate dask ops (e.g. `-means`,
    `means * scale`, `da.round(means, k)`) breaks task fusion with the
    downstream `.T.rechunk((BASIN_CHUNK, T))`, which then forces dask to
    materialise the whole (T, N) intermediate before the transpose. On the
    full ERA5-Land grid (~670 MB per time chunk) that blows worker memory and
    causes KilledWorker — resulting in an all-NaN output zarr because only
    the skeleton (mode="w") was ever written.
    """
    def _kernel(block,
                basin_idx=basin_idx, pixel_idx=pixel_idx,
                n_basins=n_basins,
                sign_flip=sign_flip, unit_scale=float(unit_scale),
                n_decimals=n_decimals):
        out = _sparse_means_block(block, basin_idx, pixel_idx, n_basins)
        if sign_flip:
            out = -out
        if unit_scale != 1.0:
            out = out * np.float32(unit_scale)
        if n_decimals is not None:
            out = np.round(out, n_decimals).astype(np.float32, copy=False)
        return out

    time_chunks = darr.chunks[0]
    return da.blockwise(
        _kernel, "tn",
        darr, "thw",
        dtype=np.float32,
        new_axes={"n": n_basins},
        adjust_chunks={"t": time_chunks},
        concatenate=True,
    )


# ---------------------------------------------------------------------------
# Output Zarr construction
# ---------------------------------------------------------------------------
def build_output_dataset(
    ds_src: xr.Dataset,
    hybas_ids: np.ndarray,
) -> xr.Dataset:
    """Build a lazy (dask-zeros) xarray Dataset with the target output structure."""
    N = len(hybas_ids)
    dates = ds_src["time"].values
    T = len(dates)

    coords = {
        "basin": ("basin", hybas_ids.astype(np.int64)),
        "date":  ("date",  dates),
    }

    data_vars = {}
    for v in ds_src.data_vars:
        src_attrs = dict(ds_src[v].attrs)  # source is empty; kept for future-proofing
        arr = da.zeros((N, T), chunks=(BASIN_CHUNK, T), dtype=np.float32)
        data_vars[v] = xr.DataArray(
            arr, dims=("basin", "date"), name=v, attrs=src_attrs,
        )

    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)
    ds_out.attrs = {
        **dict(ds_src.attrs),
        "aggregation": ("unweighted spatial mean over pixels selected by a "
                        "sparse per-basin rasterization "
                        "(rasterio.features.rasterize, all_touched=False; "
                        "fallback all_touched=True for basins with zero "
                        "strict-rule pixels). Nested basins may share pixels: "
                        "each membership contributes once to its own mean."),
        "basin_id_field": "HYBAS_ID (index of basins.geoparquet)",
    }
    ds_out.attrs.pop("history", None)
    return ds_out


def make_encoding(ds_out: xr.Dataset) -> dict:
    """Encoding mirroring Caravan HRES: Blosc-lz4-shuffle, float32, NaN fill."""
    compressor = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
    N = ds_out.sizes["basin"]
    T = ds_out.sizes["date"]
    enc = {}
    for v in ds_out.data_vars:
        enc[v] = {
            "chunks": (min(BASIN_CHUNK, N), T),
            "compressors": (compressor,),
            "dtype": "float32",
            "_FillValue": np.float32("nan"),
        }
    enc["basin"] = {"dtype": "int64", "chunks": (min(4096, N),)}
    enc["date"] = {
        "dtype": "int64",
        "units": "days since 1970-01-01",
        "calendar": "proleptic_gregorian",
        "chunks": (T,),
    }
    return enc


# ---------------------------------------------------------------------------
# Main aggregation pipeline
# ---------------------------------------------------------------------------
def run_aggregation(
    basins_path: str,
    era_path: str,
    out_path: str,
    n_workers: int = 8,
    memory_limit: str = "12GB",
    inspect_only: bool = False,
    limit_basins: int | str | None = None,
) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ONCE in main process ---
    print("[main] loading basins ...")
    gdf = gpd.read_parquet(basins_path)
    assert gdf.index.name == "HYBAS_ID", (
        f"expected basins index name 'HYBAS_ID', got {gdf.index.name!r}"
    )
    epsg = gdf.crs.to_epsg() if gdf.crs else None
    assert epsg == 4326, f"basins CRS must be EPSG:4326, got {gdf.crs}"

    if limit_basins is not None:
        if isinstance(limit_basins, str) and limit_basins.lower() == "caravan":
            if "caravan_id" not in gdf.columns:
                raise ValueError(
                    "--limit-basins caravan requested but 'caravan_id' column is "
                    "missing from basins.geoparquet"
                )
            gdf = gdf[gdf["caravan_id"].notna()].copy()
            print(f"[main] --limit-basins=caravan: subset to {len(gdf)} basins "
                  f"with a matched caravan_id")
        else:
            n = int(limit_basins)
            gdf = gdf.iloc[:n].copy()
            print(f"[main] --limit-basins={n}: subset to {len(gdf)} basins "
                  f"(HYBAS_ID {gdf.index.min()} .. {gdf.index.max()})")

    print("[main] opening ERA5Land zarr ...")
    ds_era = xr.open_zarr(era_path, consolidated=None, chunks={})
    ds_era = ds_era.sel(time=slice(TIME_START, TIME_END))
    lats = ds_era["latitude"].values
    lons = ds_era["longitude"].values
    H, W = len(lats), len(lons)
    N = len(gdf)
    T = ds_era.sizes["time"]
    t0_str = str(ds_era["time"].values[0])[:10]
    t1_str = str(ds_era["time"].values[-1])[:10]
    print(f"[main] grid HxW = {H}x{W}, basins N = {N}, timesteps T = {T} "
          f"({t0_str} -> {t1_str})")

    # --- Rasterize (sparse basin↔pixel mapping; handles nested basins) ---
    basin_idx, pixel_idx, hybas_ids, used_fallback = rasterize_basins_sparse(
        gdf, lats, lons,
    )

    # --- Write fallback log alongside output ---
    log_path = out_p.with_name(out_p.name + ".fallback_basins.csv")
    write_fallback_log(log_path, gdf, used_fallback)

    if inspect_only:
        print("[main] --inspect-only set; stopping before compute.")
        return

    # --- Build output template + write coord/metadata skeleton ---
    ds_out_template = build_output_dataset(ds_era, hybas_ids)
    encoding = make_encoding(ds_out_template)

    print("[main] writing output zarr skeleton (coords + empty vars) ...")
    ds_out_template.to_zarr(
        str(out_p), mode="w", encoding=encoding, compute=False, consolidated=True,
    )

    # --- Dask cluster ---
    print(f"[main] starting LocalCluster (n_workers={n_workers}) ...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
        dashboard_address=None,
    )
    client = Client(cluster)
    print(f"[main] dask client: {client}")

    try:
        # Re-open source with reduction-time chunking (aligned with native chunks).
        ds_r = (
            xr.open_zarr(era_path, consolidated=None)
              .sel(time=slice(TIME_START, TIME_END))
              .chunk({"time": TIME_CHUNK, "latitude": H, "longitude": W})
        )

        for v in ds_era.data_vars:
            print(f"\n[var] {v}")
            t0 = time.time()

            darr = ds_r[v].data  # dask (T, H, W)
            means = reduce_variable_blockwise(
                darr, basin_idx, pixel_idx, N,
                sign_flip=(v in SIGN_FLIP_VARS),
                unit_scale=UNIT_SCALE.get(v, 1.0),
                n_decimals=N_DECIMALS,
            )

            # Compute the (T, N) reduction to a plain numpy array on the client.
            # Total output size is small (T * N * 4 B; ~450 MB at full scale)
            # and materialising here sidesteps the .T.rechunk((BASIN_CHUNK, T))
            # graph explosion: at N=11 124 that would be a ~28 000-task all-to-all
            # shuffle which OOMs workers, silently leaves the NaN skeleton on
            # disk, and produces the all-NaN output the caravan-only run avoids
            # (because N=28 fits in a single basin chunk).
            print(f"[var] {v} computing (T={T}, N={N}) ...")
            means_np = means.compute()   # numpy (T, N) float32
            print(f"[var] {v} compute done in {time.time()-t0:.1f}s; writing ...")

            # Wrap for basin-chunked write. `da.from_array` on an in-memory
            # numpy source is cheap; the rechunk is a trivial split, not a
            # cross-worker shuffle.
            out_da = da.from_array(means_np.T, chunks=(BASIN_CHUNK, T))
            da_out = xr.DataArray(out_da, dims=("basin", "date"), name=v)
            ds_write = da_out.to_dataset()

            ds_write.to_zarr(
                str(out_p),
                mode="r+",
                region={"basin": slice(0, N), "date": slice(0, T)},
            )

            # Fail loudly if the write silently produced an all-NaN block.
            finite = int(np.isfinite(means_np).sum())
            total = means_np.size
            if finite == 0:
                raise RuntimeError(
                    f"[var] {v} reduction produced 0 finite values out of {total} "
                    f"— aborting to avoid leaving an all-NaN output on disk"
                )
            print(f"[var] {v} done in {time.time()-t0:.1f}s  "
                  f"finite={finite}/{total}")

        print("\n[main] all variables written.")

    finally:
        client.close()
        cluster.close()
        ds_era.close()

    try:
        zarr.consolidate_metadata(str(out_p))
        print(f"[main] consolidated metadata at {out_p}")
    except Exception as e:  # noqa: BLE001
        print(f"[main] WARN: consolidate_metadata failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_limit_basins(value: str) -> int | str:
    """Argparse type: accepts a non-negative integer or the literal 'caravan'."""
    if value.lower() == "caravan":
        return "caravan"
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--limit-basins must be an integer or 'caravan', got {value!r}"
        ) from e
    if n < 0:
        raise argparse.ArgumentTypeError(
            f"--limit-basins integer must be non-negative, got {n}"
        )
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basins",   default=DEFAULT_BASINS)
    p.add_argument("--era5land", default=DEFAULT_ERA5)
    p.add_argument("--caravan",  default=DEFAULT_CARAV,
                   help="structural reference only; not read into output")
    p.add_argument("--out",      default=DEFAULT_OUT)
    p.add_argument("--workers",  type=int, default=8,
                   help="dask worker count (default 8). Fewer, larger workers "
                        "handle the 670 MB per-time-chunk ERA5-Land reads with "
                        "enough headroom to avoid pause/resume churn.")
    p.add_argument("--memory-limit", default="12GB",
                   help="per-worker memory limit for dask LocalCluster "
                        "(default 12GB, sized for one source block + Python "
                        "overhead + allocator slack)")
    p.add_argument("--inspect-only", action="store_true",
                   help="print summaries + rasterize + write fallback log, "
                        "then stop before the heavy compute")
    p.add_argument("--limit-basins", type=_parse_limit_basins, default=None,
                   metavar="N|caravan",
                   help="TEST MODE: either an integer N (process only the first "
                        "N basins from the geoparquet in file order) or the "
                        "literal 'caravan' (process only basins that have a "
                        "non-null 'caravan_id' entry, i.e. those matched to a "
                        "CARAVAN gauge). Full aggregation runs on the subset.")
    args = p.parse_args()

    inspect_and_report(args.basins, args.era5land, args.caravan)
    run_aggregation(
        basins_path=args.basins,
        era_path=args.era5land,
        out_path=args.out,
        n_workers=args.workers,
        memory_limit=args.memory_limit,
        inspect_only=args.inspect_only,
        limit_basins=args.limit_basins,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
