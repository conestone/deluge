"""
DELUGE Pipeline Runner
======================
Runs every stage of the DELUGE pipeline in order.

All paths are read from _run/config.yml (via _run/paths.py). Edit that file to
point at local data — do not touch paths inside the individual scripts.

Download stages are commented out on purpose: they require external credentials
(GEE auth, CDS API key, gsutil) and are only run when refreshing raw inputs.
Uncomment the `run(...)` line for whichever source we want to (re)download.

Usage
-----
    python _run/run_pipeline.py          # full pipeline
    python _run/run_pipeline.py --check  # only verify input paths exist
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR  = REPO_ROOT / "code"
PYTHON    = sys.executable

sys.path.insert(0, str(REPO_ROOT / "_run"))
from paths import CFG  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def run(script: Path, cwd: Path | None = None) -> None:
    """Run a pipeline step as a subprocess, streaming its output."""
    rel = script.relative_to(REPO_ROOT)
    print(f"\n{'=' * 72}\n▶  {rel}\n{'=' * 72}", flush=True)
    subprocess.run(
        [PYTHON, str(script)],
        cwd=str(cwd) if cwd else None,
        check=True,
    )


# Keys in config.yml whose paths are pipeline OUTPUTS (created by the scripts).
# --check does not require these to exist yet.
_OUTPUT_KEYS = {
    "deluge_preproc_dir",
    "deluge_attributes_dir",
    "deluge_ee_subsets_dir",
    "deluge_spatial_dir",
    "deluge_basins_geoparquet",
    "deluge_inundation_geoparquet",
    "deluge_attributes_csv",
    "deluge_timeseries_dir",
    "deluge_meteorology_zarr",
    "deluge_streamflow_zarr",
    "gfd_inundated_tif_dir",
    "validation_output_dir",
    # download-target paths (only needed if we re-run the download stages)
    "hand_tiles_dir",
    "hand_output_file",
    "unosat_shp_dir",
    "unosat_gpkg_dir_rel",
    "unosat_meta_dir_rel",
    "unosat_events_json",
}

# Keys that are not paths (GEE identifiers, etc.).
_NON_PATH_KEYS = {"gee_image", "gee_project"}


def check_inputs() -> int:
    """Verify every required input path in config.yml exists on disk.

    Returns the number of missing paths (0 = all good).
    """
    print("=" * 72)
    print("  DELUGE input-path check")
    print("=" * 72)

    missing: list[tuple[str, str]] = []
    checked = 0
    for key, value in CFG.items():
        if key in _OUTPUT_KEYS or key in _NON_PATH_KEYS:
            continue
        if not isinstance(value, str) or not value.startswith("/"):
            continue  # skip relative paths and non-strings
        checked += 1
        if not Path(value).exists():
            missing.append((key, value))

    print(f"\nChecked {checked} input paths.\n")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for key, value in missing:
            print(f"  {key:32s} → {value}")
        return len(missing)
    print("All required input paths exist.")
    return 0


def run_pipeline() -> None:
    # ═════════════════════════════════════════════════════════════════════════
    # 1. DOWNLOAD  —  disabled by default (uncomment to fetch raw source data)
    # ═════════════════════════════════════════════════════════════════════════
    # Global Flood Database v1.4 — expects GFD zips at CFG["gfd_zip_dir"]
    # Prereq: gsutil -m cp -r gs://gfd_v1_4 download
    # run(CODE_DIR / "download" / "gfd.py")

    # Global HAND raster from Google Earth Engine (tiles → merged GeoTIFF)
    # Prereq: `earthengine authenticate` and a GEE project in config.yml
    # run(CODE_DIR / "download" / "hand_global.py", cwd=Path(CFG["hand_download_dir"]))

    # UNOSAT flood events — needs unosat_floodevents_webmap.txt in cwd
    # run(CODE_DIR / "download" / "unosat.py", cwd=Path(CFG["unosat_root"]))

    # ERA5-Land hourly → daily Zarr via Copernicus DataStores
    # Prereq: ~/.cdsapirc with a valid CDS API key
    # run(CODE_DIR / "download" / "era5land.py")


    # ═════════════════════════════════════════════════════════════════════════
    # 2. PREPROCESS
    # ═════════════════════════════════════════════════════════════════════════
    run(CODE_DIR / "preprocess" / "hydrosheds_merge_continents.py")
    run(CODE_DIR / "preprocess" / "gfd_InundationProcessor.py")
    run(CODE_DIR / "preprocess" / "gfd_InundationBasinLinker.py")
    run(CODE_DIR / "preprocess" / "unosat_InundationBasinLinker.py")
    run(CODE_DIR / "preprocess" / "worldfloods_InundationBasinLinker.py")


    # ═════════════════════════════════════════════════════════════════════════
    # 3. PROCESS — SPATIAL
    # ═════════════════════════════════════════════════════════════════════════
    # NOTE: Caravan_part1_Earth_Engine_static_attributes.ipynb is intentionally
    # excluded from this driver. Run it separately in Colab / Jupyter once the
    # earthengine_datasplit.py shapefiles exist and before combine_attributes.py
    # is invoked.
    run(CODE_DIR / "process" / "spatial" / "combine_basins.py")
    run(CODE_DIR / "process" / "spatial" / "combine_inundation.py")
    run(CODE_DIR / "process" / "spatial" / "combine_attributes.py")
    run(CODE_DIR / "process" / "spatial" / "final_filter.py")


    # ═════════════════════════════════════════════════════════════════════════
    # 4. PROCESS — TIMESERIES
    # ═════════════════════════════════════════════════════════════════════════
    run(CODE_DIR / "process" / "timeseries" / "earthengine_datasplit.py")
    run(CODE_DIR / "process" / "timeseries" / "aggregate_ERA5Land.py")
    run(CODE_DIR / "process" / "timeseries" / "streamflow.py")


    # ═════════════════════════════════════════════════════════════════════════
    # 5. VALIDATION
    # ═════════════════════════════════════════════════════════════════════════
    run(CODE_DIR / "validation" / "combine_basins_val.py")
    run(CODE_DIR / "validation" / "combine_inundation_val.py")
    run(CODE_DIR / "validation" / "caravan_basins_iou_val.py")
    run(CODE_DIR / "validation" / "meteorology_val.py")


    print("\n" + "=" * 72)
    print("  DELUGE pipeline complete.")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that all input paths in config.yml exist; do not run any stage.",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(check_inputs())

    run_pipeline()
