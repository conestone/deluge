import sys
import zipfile
import glob
import os
import tempfile
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

zip_dir = CFG["hydrosheds_zip_dir"]
output_path = CFG["hydrosheds_lvl9_gpkg"]

zip_files = glob.glob(os.path.join(zip_dir, "*.zip"))
print(f"Found {len(zip_files)} zip files")

gdfs = []
with tempfile.TemporaryDirectory() as tmpdir:
    for zf_path in zip_files:
        print(f"Processing {os.path.basename(zf_path)}...")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(tmpdir)
            shapefiles = [f for f in zf.namelist() if f.endswith(".shp")]

        for shp in shapefiles:
            shp_path = os.path.join(tmpdir, shp)
            gdf = gpd.read_file(shp_path)
            gdfs.append(gdf)
            print(f"  Read {len(gdf)} features from {shp}")

print("Merging all layers...")
merged = gpd.pd.concat(gdfs, ignore_index=True)
merged = gpd.GeoDataFrame(merged, crs=gdfs[0].crs)

print(f"Saving {len(merged)} total features to {output_path}...")
merged.to_file(output_path, driver="GPKG")
print("Done.")
