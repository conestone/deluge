"""
Check Water_Clas (and d_Water_Cl) values across all UNOSAT GeoPackages.
"""

import sys
from pathlib import Path
from collections import defaultdict

import geopandas as gpd
from pyogrio import list_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

GPKG_DIR = CFG["unosat_gpkg_dir"]

gpkg_files = sorted(Path(GPKG_DIR).glob("*.gpkg"))
print(f"Found {len(gpkg_files)} GeoPackage files\n")

# {column_name: {value: set of filenames}}
value_files: dict[str, dict] = defaultdict(lambda: defaultdict(set))
no_water_col: list[str] = []

for gpkg_path in gpkg_files:
    for layer_name, geom_type in list_layers(str(gpkg_path)):
        if geom_type != "MultiPolygon":
            continue
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        found = False

        for col in ("Water_Clas", "d_Water_Cl"):
            if col in gdf.columns:
                found = True
                for val in gdf[col].dropna().unique():
                    value_files[col][str(val)].add(gpkg_path.name)

        if not found:
            no_water_col.append(f"{gpkg_path.name}:{layer_name}")

# Report
for col in ("Water_Clas", "d_Water_Cl"):
    if col not in value_files:
        print(f"{col}: not found in any layer\n")
        continue
    print(f"{col} values (count = number of files containing that value):")
    for val, files in sorted(value_files[col].items(), key=lambda x: -len(x[1])):
        print(f"  {len(files):4d}x  '{val}'")
        if col == "Water_Clas":
            for f in sorted(files):
                print(f"           {f}")
    print()

if no_water_col:
    print(f"Layers with neither Water_Clas nor d_Water_Cl ({len(no_water_col)}):")
    for name in no_water_col:
        print(f"  {name}")
