================================================================================
  DELUGE PIPELINE
================================================================================

Global flood-inundation dataset linked to HydroBASINS Level-6 & 9 catchments,
combining the Global Flood Database (GFD), UNOSAT and WorldFloods flood maps
with ERA5-Land meteorology, CARAVAN streamflow, and static basin attributes.


--------------------------------------------------------------------------------
  REPOSITORY LAYOUT
--------------------------------------------------------------------------------

code/
  download/     Fetch raw source data (GFD, UNOSAT, HAND, ERA5-Land)
  preprocess/   Merge HydroSHEDS; process flood rasters; link floods to basins
  process/
    spatial/    Combine per-event outputs into global GeoParquet tables
    timeseries/ Aggregate ERA5-Land + streamflow per basin (Zarr stores)
  validation/   Sanity-check outputs and validate against CARAVAN
  analyze/      Plots and summary statistics

_run/
  run_pipeline.py    Plain-Python driver that runs every stage in order.
                     Download stages are commented out — enable manually.

outputs/       Validation reports and analysis figures.


--------------------------------------------------------------------------------
  RUN ORDER
--------------------------------------------------------------------------------

Configure paths inside each script (most use hard-coded input/output paths at
the top of the file) or edit _run/run_pipeline.py to pass different arguments.


1. DOWNLOAD  (code/download/)
   Run once per source. Requires external credentials/auth (see script headers).

   1. gfd.py           — Unzip Global Flood Database v1.4 archives.
                         Prereq: `gsutil -m cp -r gs://gfd_v1_4 download`
   2. hand_global.py   — Download global 90 m HAND raster from Google Earth
                         Engine, tile by tile, and merge into one GeoTIFF.
                         Prereq: `earthengine authenticate`
   3. unosat.py        — Download UNOSAT flood events (shapefiles → GeoPackages).
                         Prereq: unosat_floodevents_webmap.txt in cwd.
   4. era5land.py      — Batched ERA5-Land download via Copernicus DataStores,
                         aggregated to daily Zarr. Prereq: ~/.cdsapirc


2. PREPROCESS  (code/preprocess/)

   1. hydrosheds_merge_continents.py
        Merges per-continent HydroSHEDS Level-9 zips into one global GPKG.

   2. gfd_InundationProcessor.py
        Removes permanent water (JRC + HydroLAKES) and applies a HAND filter
        to GFD TIFs. Output: single-band inundation-only TIFs.

   3. gfd_InundationBasinLinker.py
        Links GFD inundated-area TIFs to HydroBASINS catchments (BFS on
        NEXT_DOWN for full upstream catchments). Optional GRDC IoU match.

   4. unosat_InundationBasinLinker.py
        Same, for UNOSAT flood-extent GeoPackages.

   5. worldfloods_InundationBasinLinker.py
        Same, for WorldFloods flood GeoJSONs.


3. PROCESS — SPATIAL  (code/process/spatial/)

   1. combine_basins.py
        Merges per-event basin files → basins.geoparquet.
        Matches basins to CARAVAN gauges by polygon IoU.

   2. combine_inundation.py
        Merges per-event inundation files → inundation.geoparquet.
        Deduplicates and normalises satellite-source names.

   3. combine_attributes.py
        Concatenates per-chunk attribute CSVs → attributes.csv.
        (Requires the Caravan Earth Engine notebook — see note below —
        to have populated the attributes/ directory.)

   4. final_filter.py
        Drops basins whose inundation percentage is below the threshold,
        recomputes upstream counts, re-densifies deluge_id.

   MANUAL (run separately, not part of run_pipeline.py):
   Caravan_part1_Earth_Engine_static_attributes.ipynb
        Google Earth Engine notebook — run in Colab. Computes static basin
        attributes (topography, land cover, climate, soils) using the basin
        subsets exported by earthengine_datasplit.py.


4. PROCESS — TIMESERIES  (code/process/timeseries/)

   1. earthengine_datasplit.py
        Simplifies basin geometries and splits basins.geoparquet into
        Earth-Engine-sized shapefile chunks (input for the Caravan notebook).

   2. aggregate_ERA5Land.py
        Rasterises basins and computes per-basin daily means from the ERA5-Land
        Zarr → meteorology.zarr.

   3. streamflow.py
        Extracts CARAVAN daily streamflow for basins with a matched caravan_id
        → streamflow.zarr.


5. VALIDATION  (code/validation/)

   1. combine_basins_val.py       — Schema/consistency checks on basins.geoparquet
   2. combine_inundation_val.py   — Schema/consistency checks on inundation.geoparquet
   3. caravan_basins_iou_val.py   — Report CARAVAN <-> HYBAS matches by IoU
   4. meteorology_val.py          — Compare DELUGE meteorology to CARAVAN reference


--------------------------------------------------------------------------------
  QUICK START
--------------------------------------------------------------------------------

    # (One-off) download the raw datasets — see script headers for prereqs
    python code/download/gfd.py
    python code/download/hand_global.py
    python code/download/unosat.py
    python code/download/era5land.py

    # Verify every input path in _run/config.yml exists on disk
    python _run/run_pipeline.py --check

    # Run the full pipeline end-to-end
    python _run/run_pipeline.py
