"""
GFDInundationProcessor
======================
Computes inundated area for GFD TIF files.

Inundated = flooded (band 1) AND NOT permanent water, where permanent water is
the union of:
  - jrc_perm_water (band 5)
  - HydroLAKES polygons (any raster cell touching a lake, all_touched=True)

Output: single-band float32 TIF (1=inundated, 0=not inundated, NaN=no data).

Usage
-----
    processor = GFDInundationProcessor(
        input_dir  = "/path/to/tif",
        output_dir = "/path/to/tif_inundated",
        lakes_path = "/path/to/HydroLAKES_polys_v10.gdb",
        n_workers  = 16,
        skip_existing = True,
    )
    processor.run()
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import label as ndimage_label
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Module-level worker function — must be importable at top level for pickling
# ---------------------------------------------------------------------------

def _process_file_worker(args: tuple) -> tuple[str, str]:
    """Worker called in a subprocess. Returns (filename, status_string)."""
    tif_path, output_dir, lakes_path, skip_existing, hand_path, hand_threshold = args
    tif_path = Path(tif_path)
    output_path = Path(output_dir) / tif_path.name

    if skip_existing and output_path.exists():
        return tif_path.name, "skipped"

    try:
        with rasterio.open(tif_path) as src:
            flooded   = src.read(1)  # float32, values: 0 / 1 / NaN
            perm_jrc  = src.read(5)  # float32, values: 0 / 1 / NaN
            profile   = src.profile.copy()
            bounds    = src.bounds
            transform = src.transform
            height, width = src.height, src.width

        # --- NaN mask (bool, cheap) ----------------------------------------
        nan_mask = np.isnan(flooded) | np.isnan(perm_jrc)   # shape: (H, W)

        # --- HydroLAKES rasterization (bool) ----------------------------------
        bbox  = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        lakes = gpd.read_file(lakes_path, bbox=bbox)

        if lakes.empty:
            lake_mask = np.zeros((height, width), dtype=bool)
        else:
            lake_mask = rasterize(
                shapes=lakes.geometry,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                default_value=1,
                dtype="uint8",
                all_touched=True,
            ).astype(bool)
        del lakes  # free geodataframe

        # --- Combined permanent water (bool) ----------------------------------
        combined_perm = (perm_jrc == 1) | lake_mask
        del perm_jrc, lake_mask  # free ~1 float32 + 1 bool array

        # --- Inundated: reuse flooded buffer to avoid extra allocation --------
        # flooded[nan]   -> NaN  (already NaN)
        # flooded[1, not perm] -> 1  (keep)
        # flooded[1, perm]     -> 0  (overwrite)
        # flooded[0]           -> 0  (keep)
        flooded[~nan_mask & (flooded == 1) & combined_perm] = 0
        del combined_perm, nan_mask

        # --- CCL: remove all isolated single-cell flood pixels (8-connectivity) -
        labeled, _ = ndimage_label(flooded == 1, structure=np.ones((3, 3), dtype=np.int8))
        component_sizes = np.bincount(labeled.ravel())
        # component_sizes[0] is the background count; skip index 0
        isolated_labels = np.where(component_sizes[1:] == 1)[0] + 1
        if isolated_labels.size > 0:
            flooded[np.isin(labeled, isolated_labels)] = 0
        del labeled, component_sizes

        # --- HAND filter: remove all flood pixels above elevation threshold ----
        if hand_path:
            with rasterio.open(hand_path) as hand_src:
                win = window_from_bounds(
                    bounds.left, bounds.bottom, bounds.right, bounds.top,
                    hand_src.transform,
                )
                hand_data = hand_src.read(
                    1,
                    window=win,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=np.nan,
                )
            flooded[(flooded == 1) & (np.isnan(hand_data) | (hand_data > hand_threshold))] = 0
            del hand_data

        # --- Write output -----------------------------------------------------
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=1, dtype="float32")

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(flooded, 1)
            dst.update_tags(1, description="Inundated Area")

        del flooded
        return tif_path.name, "ok"

    except Exception as exc:  # noqa: BLE001
        return tif_path.name, f"error: {exc}"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GFDInundationProcessor:
    """Process all GFD TIF files in a directory to inundated-only rasters."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        lakes_path: str | Path,
        hand_path: str | Path | None = None,
        hand_threshold: float = 15.0,
        n_workers: int = 16,
        skip_existing: bool = True,
    ) -> None:
        self.input_dir      = Path(input_dir)
        self.output_dir     = Path(output_dir)
        self.lakes_path     = str(lakes_path)
        self.hand_path      = str(hand_path) if hand_path else None
        self.hand_threshold = hand_threshold
        self.n_workers      = n_workers
        self.skip_existing  = skip_existing

    def _iter_tasks(self, tif_files: list[Path]):
        """Yield argument tuples for the worker function."""
        for f in tif_files:
            yield (
                str(f), str(self.output_dir), self.lakes_path, self.skip_existing,
                self.hand_path, self.hand_threshold,
            )

    def run(self) -> dict[str, list[str]]:
        """Process all TIF files. Returns dict with 'ok', 'skipped', 'error' lists."""
        tif_files = sorted(self.input_dir.glob("*.tif"))
        if not tif_files:
            print(f"No TIF files found in {self.input_dir}", file=sys.stderr)
            return {"ok": [], "skipped": [], "error": []}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, list[str]] = {"ok": [], "skipped": [], "error": []}
        n = len(tif_files)
        print(f"Found {n} TIF files. Processing with {self.n_workers} workers...")

        tasks = list(self._iter_tasks(tif_files))

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_process_file_worker, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                name, status = future.result()
                key = "error" if status.startswith("error") else status
                results[key].append(name)
                # Compact progress line
                tag = "ERR" if key == "error" else status[:2].upper()
                print(f"[{done:4d}/{n}] [{tag}] {name}" +
                      (f"  ({status})" if key == "error" else ""),
                      flush=True)

        print(
            f"\nDone — ok: {len(results['ok'])}, "
            f"skipped: {len(results['skipped'])}, "
            f"errors: {len(results['error'])}"
        )
        if results["error"]:
            print("Failed files:")
            for name in results["error"]:
                print(f"  {name}")

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_run"))
    from paths import CFG  # noqa: E402

    processor = GFDInundationProcessor(
        input_dir      = CFG["gfd_tif_dir"],
        output_dir     = CFG["gfd_inundated_tif_dir"],
        lakes_path     = CFG["hydrolakes_gdb"],
        hand_path      = CFG["hand_tif"],
        hand_threshold = 15.0,
        n_workers      = 16,
        skip_existing  = True,
    )
    processor.run()
