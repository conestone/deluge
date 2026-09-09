#!/usr/bin/env python3
"""
Download ERA5-Land hourly data using the **new** Copernicus Data Stores client
(`ecmwf.datastores.Client`), batched: all variables for a given month in a
single GRIB request restricted to a bounding box that excludes Antarctica.

Per-month workflow (storage-efficient):
  1. ONE batched CDS request → one GRIB file with all variables for the month
     in ERA5Land_hourly/era5land_{YYYY}_{MM}.grib
  2. Open the GRIB, aggregate each variable to daily (lazy / dask).
  3. Merge variables, append along time to ERA5Land_daily/ERA5Land_daily.zarr.
  4. Delete the GRIB (and its sidecar .idx files) and move to the next month.

Geographic extent:
  Bounding box [N, W, S, E] = [90, -180, -60, 180] — global except Antarctica.
  ERA5-Land's southernmost row is at -90, but everything south of -60 is
  Antarctic ocean/ice that adds storage without scientific value here.

Unit conversion:
  - MM_VARS    : m → mm    (×1000)
  - units == K : K → °C    (−273.15)

Requirements (install once):
  pip install ecmwf-datastores-client cfgrib eccodes xarray zarr

Credentials: ~/.cdsapirc (same file as the legacy cdsapi client).
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import dask
import dask.array as darr
import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.datastores import Client
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.getLogger('ecmwf.datastores').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ── User configuration ─────────────────────────────────────────────────────────

START_DATE = '1998-12-31'  # Format: YYYY-MM-DD
END_DATE   = '2026-06-01'  # Format: YYYY-MM-DD

ERA5L_BANDS = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "snow_depth_water_equivalent",
    "snowfall",
    "snowmelt",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "potential_evaporation",
    "total_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
]

# Bounding box  [North, West, South, East]  — excludes Antarctica (south = -60)
AREA = [90, -180, -60, 180]

DATA_ROOT = Path(CFG['era5land_root'])

# ── Constants ──────────────────────────────────────────────────────────────────

DIR_HOURLY = DATA_ROOT / 'ERA5Land_hourly'
ZARR_OUT   = DATA_ROOT / 'ERA5Land_daily' / 'ERA5Land_daily.zarr'

ACCUMULATED_VARS = {
    "snowfall",
    "snowmelt",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "potential_evaporation",
    "total_evaporation",
    "total_precipitation",
}

# CDS rejects the full 17-variable request as too large, so we split:
#   - accum batch  : validity-time 00 UTC of every day → full daily totals
#                    (with one extra field from month M+1 day-01 00 UTC to
#                    cover the last day of month M, concatenated into the
#                    same GRIB)
#   - instant batch: 6-hourly samples (00, 06, 12, 18 UTC), then daily mean.
#                    4 samples/day is ~6× smaller than full hourly and a
#                    reasonable approximation of the diurnal mean.
#
# Per ERA5-Land docs: validity-time 00 UTC of date D = full 24-hour
# accumulation for the PREVIOUS day (D-1).  23 UTC would only give 23 of 24
# hours, so we use 00 UTC and shift the time axis back 1 day in aggregation.
ALL_HOURS     = [f'{h:02d}:00' for h in range(24)]
INSTANT_HOURS = ['00:00', '06:00', '12:00', '18:00']
ACCUM_VARS_LIST   = [v for v in ERA5L_BANDS if v in ACCUMULATED_VARS]
INSTANT_VARS_LIST = [v for v in ERA5L_BANDS if v not in ACCUMULATED_VARS]

# Variables delivered in metres by CDS → convert to mm (×1000)
MM_VARS = {
    "snowfall",
    "snowmelt",
    "potential_evaporation",
    "total_evaporation",
    "total_precipitation",
}

# GRIB short name (as exposed by cfgrib) → CDS long name
SHORT_TO_LONG = {
    'd2m':   '2m_dewpoint_temperature',
    't2m':   '2m_temperature',
    'sd':    'snow_depth_water_equivalent',
    'sf':    'snowfall',
    'smlt':  'snowmelt',
    'swvl1': 'volumetric_soil_water_layer_1',
    'swvl2': 'volumetric_soil_water_layer_2',
    'swvl3': 'volumetric_soil_water_layer_3',
    'swvl4': 'volumetric_soil_water_layer_4',
    'ssr':   'surface_net_solar_radiation',
    'str':   'surface_net_thermal_radiation',
    'pev':   'potential_evaporation',
    'e':     'total_evaporation',
    'u10':   '10m_u_component_of_wind',
    'v10':   '10m_v_component_of_wind',
    'sp':    'surface_pressure',
    'tp':    'total_precipitation',
}

ALL_DAYS  = [f'{d:02d}' for d in range(1, 32)]

# int16 packing for Zarr — value = stored * scale_factor + add_offset, NaN → -32768.
# Scales chosen so each variable's physical range fits comfortably inside int16
# (−32768..32767) with _FillValue mapping outside any valid physical value.
#   - temps / wind / mm vars : 0.01 precision  (≈ 1/100)
#   - soil moisture          : 0.0001 m³/m³
#   - snow depth water eq.   : 0.001 m
#   - surface pressure       : 2 Pa precision, offset 70000 Pa
#   - radiation (J/m²/day)   : 2000 J/m² precision
# Cuts Zarr footprint roughly in half versus float32.
PACK_ENCODING: dict[str, dict] = {
    '2m_temperature':                {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    '2m_dewpoint_temperature':       {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'surface_pressure':              {'dtype': 'int16', 'scale_factor': 2.0,     'add_offset': 70000.0, '_FillValue': -32768},
    '10m_u_component_of_wind':       {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    '10m_v_component_of_wind':       {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'snow_depth_water_equivalent':   {'dtype': 'int16', 'scale_factor': 0.001,   'add_offset': 0.0,     '_FillValue': -32768},
    'volumetric_soil_water_layer_1': {'dtype': 'int16', 'scale_factor': 0.0001,  'add_offset': 0.0,     '_FillValue': -32768},
    'volumetric_soil_water_layer_2': {'dtype': 'int16', 'scale_factor': 0.0001,  'add_offset': 0.0,     '_FillValue': -32768},
    'volumetric_soil_water_layer_3': {'dtype': 'int16', 'scale_factor': 0.0001,  'add_offset': 0.0,     '_FillValue': -32768},
    'volumetric_soil_water_layer_4': {'dtype': 'int16', 'scale_factor': 0.0001,  'add_offset': 0.0,     '_FillValue': -32768},
    'snowfall':                      {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'snowmelt':                      {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'potential_evaporation':         {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'total_evaporation':             {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'total_precipitation':           {'dtype': 'int16', 'scale_factor': 0.01,    'add_offset': 0.0,     '_FillValue': -32768},
    'surface_net_solar_radiation':   {'dtype': 'int16', 'scale_factor': 2000.0,  'add_offset': 0.0,     '_FillValue': -32768},
    'surface_net_thermal_radiation': {'dtype': 'int16', 'scale_factor': 2000.0,  'add_offset': 0.0,     '_FillValue': -32768},
}

CHUNKS = {'time': 31, 'latitude': 300, 'longitude': 360}

# Parallelism — Python-native ProcessPoolExecutor.  Each worker handles one
# month at a time (download + aggregate + region-write to the shared Zarr).
N_PARALLEL_WORKERS      = 8
DASK_THREADS_PER_WORKER = 4   # 32 cores / 8 workers — avoids CPU thrashing
ZARR_LOCK = Path(str(ZARR_OUT) + '.write.lock')
FLAG_DIR  = DATA_ROOT / 'ERA5Land_daily' / '_progress'


# ── Helpers ────────────────────────────────────────────────────────────────────

def months_in_range(start: str, end: str) -> list[tuple[int, int]]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out: list[tuple[int, int]] = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def completed_months_in_zarr(zarr_path: Path) -> set[tuple[int, int]]:
    if not zarr_path.exists():
        return set()
    try:
        ds = xr.open_zarr(str(zarr_path))
        times = pd.to_datetime(ds.time.values)
        return {(t.year, t.month) for t in times}
    except Exception as exc:
        log.warning('Could not inspect existing Zarr: %s', exc)
        return set()


def make_client() -> Client:
    """Build the new ecmwf.datastores Client, reading credentials from ~/.cdsapirc."""
    rc_path = Path.home() / '.cdsapirc'
    cfg: dict[str, str] = {}
    if rc_path.exists():
        for line in rc_path.read_text().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                cfg[k.strip()] = v.strip()
    return Client(url=cfg.get('url'), key=cfg.get('key'), progress=True)


# ── Download (one batched GRIB per month) ──────────────────────────────────────

def _retrieve(client: Client, collection: str, request: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.retrieve(collection, request, str(target))
    except Exception:
        target.unlink(missing_ok=True)
        raise


def download_instant_batch(
    client: Client, year: int, month: int, out_path: Path,
) -> Path:
    """
    Instantaneous vars at 6-hourly sampling (00, 06, 12, 18 UTC) → one GRIB.
    The local aggregation step turns these 4 samples/day into a daily mean.
    """
    if out_path.exists():
        log.info('Already on disk: %s', out_path.name)
        return out_path

    tmp_path = out_path.with_suffix('.tmp.grib')
    log.info('⬇  %d-%02d  instant  (%d vars × %d hours/day, area=%s)',
             year, month, len(INSTANT_VARS_LIST), len(INSTANT_HOURS), AREA)
    _retrieve(client, 'reanalysis-era5-land', {
        'variable': INSTANT_VARS_LIST,
        'year': str(year),
        'month': f'{month:02d}',
        'day': ALL_DAYS,
        'time': INSTANT_HOURS,
        'data_format': 'grib',
        'download_format': 'unarchived',
        'area': AREA,
    }, tmp_path)
    tmp_path.rename(out_path)
    log.info('✓  saved %s (%.1f GB)', out_path.name, out_path.stat().st_size / 1e9)
    return out_path


def download_accum_batch(
    client: Client, year: int, month: int, out_path: Path,
) -> Path:
    """
    Accumulated vars for one month → one GRIB.

    Combines two CDS requests into one file:
      1. month=M, day=1..N, time='00:00'   — covers M-1-last and M-1..M-(N-1)
      2. month=M+1, day=01, time='00:00'   — covers M-N (the last day)

    GRIBs are sequences of self-contained messages, so concatenating the two
    binary blobs yields a valid single GRIB readable by cfgrib.
    """
    if out_path.exists():
        log.info('Already on disk: %s', out_path.name)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_main = out_path.with_suffix('.main.tmp.grib')
    tmp_supp = out_path.with_suffix('.supp.tmp.grib')
    tmp_out  = out_path.with_suffix('.tmp.grib')

    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)

    log.info('⬇  %d-%02d  accum  main (%d vars × %d days × 00 UTC)',
             year, month, len(ACCUM_VARS_LIST), len(ALL_DAYS))
    _retrieve(client, 'reanalysis-era5-land', {
        'variable': ACCUM_VARS_LIST,
        'year': str(year),
        'month': f'{month:02d}',
        'day': ALL_DAYS,
        'time': ['00:00'],
        'data_format': 'grib',
        'download_format': 'unarchived',
        'area': AREA,
    }, tmp_main)

    log.info('⬇  %d-%02d  accum  supp (= %d-%02d-01 00 UTC, last-day total)',
             year, month, next_year, next_month)
    _retrieve(client, 'reanalysis-era5-land', {
        'variable': ACCUM_VARS_LIST,
        'year': str(next_year),
        'month': f'{next_month:02d}',
        'day': '01',
        'time': '00:00',
        'data_format': 'grib',
        'download_format': 'unarchived',
        'area': AREA,
    }, tmp_supp)

    # Concatenate the two GRIBs (binary) into the final file
    try:
        with open(tmp_out, 'wb') as out, open(tmp_main, 'rb') as a, open(tmp_supp, 'rb') as b:
            while chunk := a.read(8 * 1024 * 1024):
                out.write(chunk)
            while chunk := b.read(8 * 1024 * 1024):
                out.write(chunk)
    except Exception:
        tmp_out.unlink(missing_ok=True)
        tmp_main.unlink(missing_ok=True)
        tmp_supp.unlink(missing_ok=True)
        raise

    tmp_main.unlink(missing_ok=True)
    tmp_supp.unlink(missing_ok=True)
    tmp_out.rename(out_path)
    log.info('✓  saved %s (%.1f GB)', out_path.name, out_path.stat().st_size / 1e9)
    return out_path


# ── Aggregation (GRIB → daily Zarr) ───────────────────────────────────────────

def _open_grib_datasets(grib_path: Path) -> list[xr.Dataset]:
    """
    Open a GRIB with cfgrib and normalise so 'time' is always a flat 1D
    dimension.  ``time_dims=('valid_time',)`` forces cfgrib to collapse any
    (time, step) forecast hypercube into a single valid_time axis.
    """
    import cfgrib

    raw = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={
            'indexpath': '',                # don't write .idx sidecar files
            'time_dims': ('valid_time',),   # flatten (time, step) → valid_time
        },
        chunks={'latitude': 300, 'longitude': 360},
    )
    out = []
    for ds in raw:
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        elif 'valid_time' in ds.coords and 'time' in ds.dims:
            ds = ds.swap_dims({'time': 'valid_time'}).rename({'valid_time': 'time'})

        if 'time' in ds.coords and 'time' not in ds.dims:
            ds = ds.expand_dims('time')

        if 'time' in ds.dims:
            out.append(ds)
    return out


def _convert_units(daily: xr.DataArray, long: str, attrs: dict) -> tuple[xr.DataArray, dict]:
    attrs = dict(attrs)
    if long in MM_VARS:
        daily = daily * 1000.0
        attrs['units'] = 'mm'
    if attrs.get('units') == 'K':
        daily = daily - 273.15
        attrs['units'] = 'degC'
    return daily, attrs


def build_daily_dataset(
    accum_grib_paths: list[Path],
    instant_grib_paths: list[Path],
    year: int,
    month: int,
) -> xr.Dataset:
    """
    Produce a single daily Dataset with all variables renamed to their CDS
    long names, unit-converted, and covering exactly the days of (year, month).

    Accumulated vars (hourly GRIBs from `reanalysis-era5-land`):
      - validity-time 00 UTC of date D = full daily total for date D-1
      - shift the time axis back 1 day, then filter to month M.

    Instantaneous vars (6-hourly GRIB from `reanalysis-era5-land`):
      - daily mean over the 4 samples per day (00, 06, 12, 18 UTC).
    """
    import numpy as np

    # ---- Accumulated: collect, shift back 1 day, filter to month M ----------
    accum_per_var: dict[str, list[xr.DataArray]] = {}
    for p in accum_grib_paths:
        for ds in _open_grib_datasets(p):
            for short in list(ds.data_vars):
                accum_per_var.setdefault(short, []).append(ds[short])

    month_start = pd.Timestamp(year=year, month=month, day=1)
    next_month_start_64 = (month_start + pd.DateOffset(months=1)).to_datetime64()
    month_start_64 = month_start.to_datetime64()
    one_day = np.timedelta64(1, 'D')

    daily_parts: list[xr.Dataset] = []
    for short, arrs in accum_per_var.items():
        long = SHORT_TO_LONG.get(short, short)
        attrs = dict(arrs[0].attrs)

        da = xr.concat(arrs, dim='time') if len(arrs) > 1 else arrs[0]
        shifted_time = da['time'].values.astype('datetime64[ns]') - one_day
        da = da.assign_coords(time=('time', shifted_time)).sortby('time')
        in_month = (da['time'] >= month_start_64) & (da['time'] < next_month_start_64)
        da = da.where(in_month, drop=True)
        daily = da.resample(time='1D').last()

        daily, attrs = _convert_units(daily, long, attrs)
        out = daily.to_dataset(name=long)
        out[long].attrs = attrs
        daily_parts.append(out)

    # ---- Instantaneous: 4 samples/day → daily mean ---------------------------
    instant_per_var: dict[str, xr.DataArray] = {}
    for p in instant_grib_paths:
        for ds in _open_grib_datasets(p):
            for short in list(ds.data_vars):
                instant_per_var[short] = ds[short]

    for short, da in instant_per_var.items():
        long = SHORT_TO_LONG.get(short, short)
        attrs = dict(da.attrs)

        daily = da.resample(time='1D').mean()

        daily, attrs = _convert_units(daily, long, attrs)
        out = daily.to_dataset(name=long)
        out[long].attrs = attrs
        daily_parts.append(out)

    return xr.merge(daily_parts, compat='override')


# ── Per-month orchestration ────────────────────────────────────────────────────

def process_month(
    client: Client,
    year: int,
    month: int,
    completed: set[tuple[int, int]],
) -> None:
    if (year, month) in completed:
        log.info('⏭  skip %d-%02d (already in Zarr)', year, month)
        return

    log.info('▶  month %d-%02d  ──────────────────────────────', year, month)
    accum_path   = DIR_HOURLY / f'era5land_accum_{year}_{month:02d}.grib'
    instant_path = DIR_HOURLY / f'era5land_instant_{year}_{month:02d}.grib'

    # Accum first — smaller request, fails fast if CDS rejects
    try:
        download_accum_batch(client, year, month, accum_path)
    except Exception as exc:
        log.error('Accum download FAILED %d-%02d: %s', year, month, exc)
        return
    try:
        download_instant_batch(client, year, month, instant_path)
    except Exception as exc:
        log.error('Instant download FAILED %d-%02d: %s', year, month, exc)
        return

    log.info('🛠   building daily dataset graph for %d-%02d…', year, month)
    t0 = time.monotonic()
    try:
        merged = build_daily_dataset([accum_path], [instant_path], year, month)
    except Exception as exc:
        log.error('Aggregation FAILED %d-%02d: %s', year, month, exc)
        return
    log.info('   graph ready in %.0fs  (%d vars, %d daily steps) — now computing + writing to Zarr (this is the slow step)',
             time.monotonic() - t0, len(merged.data_vars), merged.sizes.get('time', 0))

    merged = merged.chunk(CHUNKS)
    ZARR_OUT.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.monotonic()
    if ZARR_OUT.exists():
        # Subsequent monthly appends inherit the existing schema's encoding,
        # so the packing chosen on the first write is preserved automatically.
        merged.to_zarr(str(ZARR_OUT), mode='a', append_dim='time')
        log.info('Appended %d-%02d → Zarr   (write %.0fs)',
                 year, month, time.monotonic() - t1)
    else:
        encoding = {v: PACK_ENCODING[v] for v in merged.data_vars if v in PACK_ENCODING}
        merged.to_zarr(str(ZARR_OUT), mode='w', encoding=encoding)
        log.info('Created Zarr (first month %d-%02d, %d vars packed to int16, write %.0fs)',
                 year, month, len(encoding), time.monotonic() - t1)

    # Cleanup — GRIBs and any cfgrib sidecar index files
    for gp in (accum_path, instant_path):
        gp.unlink(missing_ok=True)
        for sidecar in gp.parent.glob(f'{gp.name}.*.idx'):
            sidecar.unlink(missing_ok=True)
    log.info('🧹 cleaned %d-%02d', year, month)

    completed.add((year, month))


# ── Parallel infrastructure ────────────────────────────────────────────────────

def compute_grid() -> tuple[np.ndarray, np.ndarray]:
    """ERA5-Land 0.1° grid clipped to AREA = [N, W, S, E]."""
    north, west, south, east = AREA
    lats = np.round(np.arange(north, south - 0.001, -0.1), 1)  # 90.0 → -60.0
    lons = np.round(np.arange(west, east, 0.1), 1)             # -180.0 → 179.9
    return lats, lons


def _build_template(times: pd.DatetimeIndex) -> xr.Dataset:
    """NaN-filled lazy Dataset spanning the given time axis."""
    lats, lons = compute_grid()
    nt, nlat, nlon = len(times), len(lats), len(lons)
    chunks = (CHUNKS['time'], CHUNKS['latitude'], CHUNKS['longitude'])
    data_vars = {
        var: (('time', 'latitude', 'longitude'),
              darr.full((nt, nlat, nlon), np.nan, dtype='float32', chunks=chunks))
        for var in ERA5L_BANDS
    }
    return xr.Dataset(
        data_vars,
        coords={'time': times, 'latitude': lats, 'longitude': lons},
    )


def preallocate_or_extend_zarr(zarr_path: Path) -> None:
    """
    Create the Zarr if missing, OR extend its time axis if END_DATE goes
    beyond the existing end.  Prepending (earlier START_DATE than existing)
    is NOT supported — delete the Zarr in that case.
    """
    target_times = pd.date_range(START_DATE, END_DATE, freq='D')

    if not zarr_path.exists():
        nt = len(target_times)
        log.info('Pre-allocating Zarr: time=%d days (%s..%s)',
                 nt, target_times[0].date(), target_times[-1].date())
        template = _build_template(target_times)
        encoding = {v: PACK_ENCODING[v] for v in ERA5L_BANDS}
        template.to_zarr(str(zarr_path), mode='w', encoding=encoding, compute=False)
        return

    existing = xr.open_zarr(str(zarr_path))
    existing_times = pd.to_datetime(existing.time.values)
    existing.close()

    if target_times[0] < existing_times[0]:
        raise RuntimeError(
            f'START_DATE ({target_times[0].date()}) is earlier than the existing '
            f'Zarr start ({existing_times[0].date()}). Prepending is not supported — '
            f'delete the Zarr at {zarr_path} to start fresh.'
        )

    if target_times[-1] <= existing_times[-1]:
        log.info('Existing Zarr time axis already covers %s..%s — no extension needed',
                 target_times[0].date(), target_times[-1].date())
        return

    # Extend: append NaN-filled days from (existing_end + 1 day) .. END_DATE
    new_times = pd.date_range(
        existing_times[-1] + pd.Timedelta(days=1),
        target_times[-1],
        freq='D',
    )
    log.info('Extending Zarr time axis by %d days (%s..%s)',
             len(new_times), new_times[0].date(), new_times[-1].date())
    template = _build_template(new_times)
    # safe_chunks=False: the existing 730-day axis ends mid-chunk, so the
    # appended days straddle a Zarr chunk boundary.  Pre-allocation is
    # metadata-only (compute=False) and runs in the main process before any
    # workers start, so there's no real concurrency risk.
    template.to_zarr(
        str(zarr_path),
        mode='a',
        append_dim='time',
        compute=False,
        safe_chunks=False,
    )
    log.info('Zarr now covers %d days', len(existing_times) + len(new_times))


def flag_path_for(year: int, month: int) -> Path:
    return FLAG_DIR / f'done_{year}_{month:02d}.flag'


def is_month_done(year: int, month: int) -> bool:
    return flag_path_for(year, month).exists()


def mark_month_done(year: int, month: int) -> None:
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    flag_path_for(year, month).touch()


def process_month_parallel(year: int, month: int) -> tuple[int, int, float]:
    """
    Worker entry point.  Runs in its own Python process.
    Returns (year, month, wall_seconds).  Raises on failure.
    """
    # Limit dask thread fan-out so 8 workers don't all spawn 32 threads each.
    dask.config.set(scheduler='threads', num_workers=DASK_THREADS_PER_WORKER)
    t_start = time.monotonic()

    if is_month_done(year, month):
        log.info('⏭  skip %d-%02d (flag exists)', year, month)
        return (year, month, 0.0)

    accum_path   = DIR_HOURLY / f'era5land_accum_{year}_{month:02d}.grib'
    instant_path = DIR_HOURLY / f'era5land_instant_{year}_{month:02d}.grib'
    cds = make_client()

    log.info('▶  worker pid=%d  month %d-%02d', __import__('os').getpid(), year, month)
    download_accum_batch(cds, year, month, accum_path)
    download_instant_batch(cds, year, month, instant_path)

    log.info('🛠   building daily dataset for %d-%02d…', year, month)
    merged = build_daily_dataset([accum_path], [instant_path], year, month)
    merged = merged.chunk(CHUNKS)

    # Compute time-slice indices from the ACTUAL Zarr time axis so this stays
    # correct even if the script's START_DATE differs from the Zarr's start.
    with xr.open_zarr(str(ZARR_OUT)) as _zds:
        full_times = pd.to_datetime(_zds.time.values)
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end   = (month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1))
    start_idx   = int(full_times.get_loc(month_start))
    end_idx     = int(full_times.get_loc(month_end)) + 1

    # For region writes, xarray requires every variable in the Dataset to
    # share at least one dim with the region's dims.  cfgrib leaves scalar
    # coords (number/surface/depthBelowLandLayer) and 1-D lat/lon coord arrays
    # on the merged dataset — drop anything that doesn't have the `time` dim.
    drop = [v for v in merged.variables if 'time' not in merged[v].dims]
    merged = merged.drop_vars(drop, errors='ignore')

    log.info('   computing + writing region time=[%d:%d] for %d-%02d (dropped %d non-time vars)',
             start_idx, end_idx, year, month, len(drop))
    # Serialize Zarr writes — month boundaries straddle 31-day Zarr chunks
    # (e.g. April spans chunks 2 and 3).  FileLock prevents any two workers
    # from touching the same chunk simultaneously, so we can safely tell
    # xarray to skip its own chunk-alignment check (safe_chunks=False).
    with FileLock(str(ZARR_LOCK)):
        merged.to_zarr(
            str(ZARR_OUT),
            region={'time': slice(start_idx, end_idx)},
            safe_chunks=False,
        )

    mark_month_done(year, month)

    # Cleanup — GRIBs and any cfgrib sidecar index files
    for gp in (accum_path, instant_path):
        gp.unlink(missing_ok=True)
        for sidecar in gp.parent.glob(f'{gp.name}.*.idx'):
            sidecar.unlink(missing_ok=True)

    elapsed = time.monotonic() - t_start
    log.info('✓  %d-%02d done in %.0fs', year, month, elapsed)
    return (year, month, elapsed)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    months = months_in_range(START_DATE, END_DATE)
    preallocate_or_extend_zarr(ZARR_OUT)
    FLAG_DIR.mkdir(parents=True, exist_ok=True)

    pending = [(y, m) for y, m in months if not is_month_done(y, m)]
    log.info('Months: %d total, %d already done (flag), %d pending',
             len(months), len(months) - len(pending), len(pending))
    log.info('Running %d workers (× %d dask threads each)',
             N_PARALLEL_WORKERS, DASK_THREADS_PER_WORKER)

    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=N_PARALLEL_WORKERS) as ex:
        futures = {ex.submit(process_month_parallel, y, m): (y, m) for y, m in pending}
        for fut in as_completed(futures):
            y, m = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                log.exception('FAILED  %d-%02d: %s', y, m, exc)
    log.info('All workers finished in %.0fs', time.monotonic() - t0)


if __name__ == '__main__':
    main()
