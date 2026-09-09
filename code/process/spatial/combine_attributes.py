import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_run"))
from paths import CFG  # noqa: E402

ATTR_DIR = Path(CFG["deluge_attributes_dir"])
BASINS_PATH = Path(CFG["deluge_basins_geoparquet"])
OUTPUT_PATH = Path(CFG["deluge_attributes_csv"])

csv_files = sorted(ATTR_DIR.glob("*.csv"))
print(f"Concatenating {len(csv_files)} attribute CSVs...")
attrs = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"  {len(attrs):,} rows, {attrs.shape[1]} columns")

print("Loading basin deluge_id → HYBAS_ID mapping...")
mapping = pd.read_parquet(BASINS_PATH, columns=["deluge_id"]).reset_index()[["HYBAS_ID", "deluge_id"]]

print("Merging attributes with HYBAS_ID...")
attrs = attrs.merge(mapping, on="deluge_id", how="inner")
attrs = attrs.drop(columns=["deluge_id"]).set_index("HYBAS_ID").sort_index()

print("Rounding numeric columns...")
mm_cols = [c for c in attrs.columns if "_mm" in c]
other_num_cols = attrs.select_dtypes(include="number").columns.difference(mm_cols)
attrs[mm_cols] = attrs[mm_cols].round(0).astype("Int64")
attrs[other_num_cols] = attrs[other_num_cols].round(2)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving {len(attrs):,} rows → {OUTPUT_PATH}")
attrs.to_csv(OUTPUT_PATH)
print("Done.")
