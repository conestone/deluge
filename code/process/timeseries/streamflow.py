"""
Build streamflow.zarr from CARAVAN daily streamflow for HYBAS basins that already
carry a caravan_id in basins.geoparquet.

Output: B_Timeseries/streamflow.zarr
  dims:    HYBAS_ID, date
  vars:    streamflow(HYBAS_ID, date), caravan_id(HYBAS_ID)
  attrs:   source = "caravan"
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import netCDF4
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402

CARAVAN_NC     = Path(CFG["caravan_nc_dir"])
BASINS_PATH    = Path(CFG["deluge_basins_geoparquet"])
TIMESERIES_DIR = Path(CFG["deluge_timeseries_dir"])
DATE_EPOCH     = pd.Timestamp("1951-01-01")

print("Loading basins.geoparquet...")
basins = gpd.read_parquet(BASINS_PATH)
matched = basins[basins["caravan_id"].notna()][["caravan_id"]]
print(f"Basins with caravan_id: {len(matched)}")

series = {}
caravan_ids = {}
for hybas_id, caravan_id in tqdm(matched["caravan_id"].items(), total=len(matched)):
    subdataset = caravan_id.split("_")[0]
    nc_path = CARAVAN_NC / subdataset / f"{caravan_id}.nc"
    if not nc_path.exists():
        print(f"  WARNING: {nc_path} not found, skipping")
        continue

    ds = netCDF4.Dataset(str(nc_path), "r")
    date_days  = ds.variables["date"][:].filled(np.nan).astype(np.float32)
    streamflow = ds.variables["streamflow"][:].filled(np.nan).astype(np.float32)
    ds.close()

    dates = DATE_EPOCH + pd.to_timedelta(date_days, unit="D")
    series[int(hybas_id)] = pd.Series(streamflow, index=pd.DatetimeIndex(dates, name="date"))
    caravan_ids[int(hybas_id)] = caravan_id

df = pd.DataFrame(series)                     # rows: date, cols: HYBAS_ID
df = df.sort_index().sort_index(axis=1)
hybas_ids = df.columns.to_numpy(dtype=np.int64)

ds = xr.Dataset(
    data_vars={
        "streamflow": (("HYBAS_ID", "date"), df.to_numpy(dtype=np.float32).T),
        "caravan_id": (("HYBAS_ID",), np.array([caravan_ids[h] for h in hybas_ids], dtype=object)),
    },
    coords={
        "HYBAS_ID": hybas_ids,
        "date":     df.index.values,
    },
    attrs={"source": "caravan"},
)

TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
out_path = TIMESERIES_DIR / "streamflow.zarr"
if out_path.exists():
    import shutil
    shutil.rmtree(out_path)
ds.to_zarr(out_path, mode="w")

print(f"Saved: {out_path}  ({ds.sizes['HYBAS_ID']} basins, {ds.sizes['date']} dates)")
