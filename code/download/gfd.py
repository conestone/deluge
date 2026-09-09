"""
Download Global Flood Database with gsutil command:
gsutil -m cp -r gs://gfd_v1_4 download

Use this script to unzip Global Flood Database v1.4 archives in parallel.

Extracts:
  - .tif  files  -> <repo_root>/tif/
  - .json files  -> <repo_root>/metadata/

Usage:
    python unzip_gfd.py [--workers 16]
"""

import argparse
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
from paths import CFG  # noqa: E402

ZIP_DIR = Path(CFG["gfd_zip_dir"])
TIF_DIR = Path(CFG["gfd_tif_dir"])
META_DIR = Path(CFG["gfd_metadata_dir"])


def extract_zip(zip_path: Path) -> tuple[str, int, list[str]]:
    """
    Extract a single zip file.

    Returns (zip name, number of files extracted, list of skipped files).
    """
    extracted = 0
    skipped = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            suffix = Path(name).suffix.lower()

            if suffix == ".tif":
                dest_dir = TIF_DIR
            elif suffix == ".json":
                dest_dir = META_DIR
            else:
                skipped.append(name)
                continue

            dest_path = dest_dir / Path(name).name

            # Skip if already extracted (allows reruns without re-work)
            if dest_path.exists():
                skipped.append(f"{name} (already exists)")
                continue

            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    return zip_path.name, extracted, skipped


def main(workers: int) -> None:
    TIF_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    total = len(zip_files)
    if total == 0:
        print(f"No .zip files found in {ZIP_DIR}")
        return

    print(f"Found {total} zip files. Extracting with {workers} workers...")

    done = 0
    total_extracted = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_zip, z): z for z in zip_files}
        for future in as_completed(futures):
            name, n_extracted, skipped = future.result()
            done += 1
            total_extracted += n_extracted
            print(f"[{done:4d}/{total}] {name}  ->  {n_extracted} file(s) extracted"
                  + (f"  ({len(skipped)} skipped)" if skipped else ""))

    print(f"\nDone. {total_extracted} files extracted in total.")
    print(f"  TIF files  : {TIF_DIR}")
    print(f"  JSON files : {META_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip GFD v1.4 archives.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel worker processes (default: 16)")
    args = parser.parse_args()
    main(args.workers)
