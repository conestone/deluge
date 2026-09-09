import shutil
import sys
from pathlib import Path
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402

BASINS_PATH = Path(CFG["deluge_basins_geoparquet"])
OUTPUT_DIR  = Path(CFG["deluge_ee_subsets_dir"])
CHUNK_SIZE  = 5000
SIMPLIFY_M  = 1000  # tolerance in meters

basins = gpd.read_parquet(BASINS_PATH).reset_index()
n = len(basins)
source_crs = basins.crs
print(f"Loaded {n} basins (CRS: {source_crs})")

# Simplify in an equal-area metric CRS so tolerance is in meters, then reproject back.
metric_crs = "EPSG:6933"
basins_metric = basins.to_crs(metric_crs)
basins_metric["geometry"] = basins_metric.geometry.simplify(
    SIMPLIFY_M, preserve_topology=True
)
basins = basins_metric.to_crs(source_crs)
print(f"Simplified geometries with tolerance {SIMPLIFY_M} m")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

starts = range(0, n, CHUNK_SIZE)
for start in starts:
    end   = min(start + CHUNK_SIZE, n)
    chunk = basins.iloc[start:end]
    path  = OUTPUT_DIR / f"basins_{start}_{end}.shp"
    chunk.to_file(path)
    print(f"  Saved {path.name}  ({len(chunk)} basins)")

zip_path = OUTPUT_DIR.parent / OUTPUT_DIR.name
shutil.make_archive(str(zip_path), "zip", OUTPUT_DIR)
print(f"\nCompressed → {zip_path}.zip")
print("Done.")
