# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Snow Road Closure Prediction App for Pittsburgh. Predicts which roads will close during snowstorms and plans routes around them. Built for TartanHacks 2026.

## Directory Structure

```
tartanhacks26/
├── pipeline/        # Data collection (18 scripts) + ML pipeline (4 scripts) + orchestrator + join
├── legacy/          # CMU-scoped demo scripts (800m radius) + their data
├── data/            # Generated pipeline output CSVs/JSONs/Parquet/model (gitignored)
├── requirements.txt
└── CLAUDE.md
```

## Architecture

### Legacy CMU-Scoped Scripts (`legacy/`)

- **roads_osm.py** — OSM road data around CMU. Outputs `roads_osm_cmu.csv`.
- **roads_demo.py** — Google Maps road data around CMU. Requires `GOOGLE_MAPS_API_KEY`. Outputs `roads_around_cmu.csv`.
- **roads_analysis.py** — Joins OSM + Google CSVs. Outputs `roads_joined_cmu.csv`.

### Pittsburgh-Wide Data Collection Pipeline (`pipeline/`)

All scripts are standalone and output CSVs to `data/` with a `data_source` column.

**Phase A — Road Network Foundation:**
- **collect_centerlines.py** — WPRDC Street Centerlines (~18,665 segments). Base road network. GeoJSON resource `8a38a51d` under dataset `9ebd073b`.
- **collect_osm_pgh.py** — OSM roads for all of Pittsburgh (bbox, ~36,439 segments). Includes `incline` tag.

**Phase B — Environmental Risk Factors (needs Phase A):**
- **collect_elevation.py** — Open-Elevation API, batch elevation queries per segment midpoint (~18,359 rows).
- **collect_slopes.py** — WPRDC 25%+ Slope Areas (dataset `25-or-greater-slope`, resource `5ce91a56`). Flags roads within 50m of steep slopes (~1,415 flagged).
- **collect_bridges.py** — WPRDC City of Pittsburgh Bridges (resource `23f675e5`, ~153 records).
- **collect_landslides.py** — WPRDC Global Landslides (resource `dde1f413`), filtered to Pittsburgh bounding box (~67 records).

**Phase C — Weather Data (independent):**
- **collect_weather_historical.py** — Open-Meteo daily weather, 11 winter seasons 2014-2025 (~1,664 days).
- **collect_weather_realtime.py** — NWS API forecast + winter alerts for Pittsburgh.
- **collect_penndot_winter.py** — PennDOT live winter conditions. Service URL: `winterconditions/winterconditions/MapServer`. Empty between storms.

**Phase D — Historical Incident Data (independent):**
- **collect_crashes.py** — WPRDC cumulative crash data (resource `2c13021f`), filtered for snow/ice (~19,897 records). Some lat/lng in DMS format — build_dataset coerces to numeric.
- **collect_311_snow.py** — WPRDC 311 data (resource `29462525`), filtered by `request_type_name` LIKE Snow/Ice (~39,080 records). Column names are lowercase in new format.
- **collect_plow_activity.py** — WPRDC plow GPS traces 2018-2020 (dataset `d0b56030`, ~722k records from 5 events).
- **collect_penndot_snow_routes.py** — PennDOT roadway segments for Allegheny County (~5,332 segments). Service URL: `opendata/roadwaysegments/MapServer`. Fields: `OVERALL_PVMNT_IDX`, `ROUGH_INDX`.
- **collect_domi_closures.py** — WPRDC DOMI street closure permits (~63k records). Resource `a9a1d93a`. Includes full/lane/parking/sidewalk closure flags.
- **collect_snow_emergency_routes.py** — Derived snow priority routes from centerlines DOMI classification (~3,683 segments). Arterial + Collector roads = snow plowing priority.
- **collect_511pa_events.py** — 511PA traffic events for Allegheny County (real-time snapshot, append mode). CSV export from 511pa.com.
- **collect_penndot_press.py** — PennDOT District 11 press releases parsed for road closure details. Regex-based text extraction.

**Phase E — Master Join:**
- **build_dataset.py** — Joins all CSVs into `data/dataset_prediction_ready.csv` (~28,805 rows, 27 columns). Uses scipy KDTree for spatial joins. Coerces lat/lng to numeric before spatial operations. Adds `is_snow_emergency_route`, `domi_closure_count`, `domi_full_closure_count`. Note: dataset has ~17,550 unique objectids but ~28,805 rows due to multi-segment roads.

**Phase F — ML Pipeline (needs Phase E + Phase C):**
- **build_labels.py** — Detects 126 storm events from historical weather (snowfall >= 1cm, consecutive days grouped, +2 day extension). Labels each of 17,549 unique segments per storm using 4 proxy sources: 311 snow complaints (200m, temporal), winter crashes (200m, monthly proportional allocation by snowfall), plow activity gaps (100m, binary, 3/126 storms have data), DOMI full closures (200m, temporal). Composite risk score via percentile normalization + weighted sum (0.45 × 311 + 0.25 × crash + 0.15 × plow + 0.15 × DOMI). Outputs `storm_events.csv` (126 storms) and `storm_labels.csv` (~2.2M rows).
- **build_training_data.py** — Encodes 28 static features (ordinal encoding for road class/pavement/highway type, log1p for count features, -1 sentinel for missing PennDOT, bool-to-int for flags). Joins with storm weather features and labels. Writes `training_matrix.parquet` (~2.2M rows × 38 columns) via chunked processing with PyArrow to avoid OOM. Deduplicates segments by objectid.
- **train_model.py** — LightGBM regression on 28 static features (weather excluded from model to force segment-level differentiation). Storm-level random 90/10 train/val split + temporal test on 2023-25 seasons. 335 trees with strong regularization (`min_child_samples=200`, `lambda_l2=2.0`, `max_depth=7`). Outputs `model_lgbm.txt`, `model_metadata.json`, `feature_importance.csv`.
- **predict.py** — Two input modes: NWS forecast JSON parsing or CLI args (`--snowfall_cm`, `--min_temp_c`, `--max_wind_kmh`, `--duration_days`). Loads model for segment-level risk, applies weather severity multiplier (snowfall/temp/wind/duration → 0.1–1.0 scale), min-max normalizes to [0,1], categorizes (very_low/low/moderate/high/very_high). Outputs `predictions_latest.csv` and `predictions_latest.json` (GeoJSON).

**Orchestrator:**
- **collect_all.py** — Runs all scripts in dependency order (Phases A→F).

### ML Model Details

**Label construction** (no ground-truth closure labels exist):
- Storm events detected from weather history (snowfall >= 1cm days, grouped consecutively)
- Per-segment risk constructed from 4 noisy proxy sources via spatial+temporal joins
- Within-storm percentile normalization ensures relative ranking, not absolute risk

**Model architecture**:
- LightGBM regression, 335 boosted trees, 28 static features
- Weather excluded from model features — all segments share the same weather per prediction, so weather would dominate and prevent segment differentiation
- Weather applied as a post-hoc severity multiplier at prediction time
- Top features: snow_complaint_count (13.5%), plow_coverage_score (12.5%), domi_closure_count (9.8%), mid_lat (9.3%), roadwidth (8.5%), elevation_m (8.0%)

**Prediction output**:
- Light snow (3cm): nearly all roads very_low, ~10 reach low
- Moderate storm (15cm): ~7% low, ~0.1% moderate, 4 high
- Blizzard (30cm, -15C, 60km/h): 35% low, 1.4% moderate, 59 high, 16 very_high
- Highest-risk roads: steep areas (Hazelwood, Riverview), bridge corridors (Carson St, Ohio River Blvd), poorly-plowed local roads

## Running the Pipeline

```bash
pip install -r requirements.txt

# Full pipeline (data collection + ML):
python pipeline/collect_all.py

# ML pipeline only (needs data/ CSVs from collection):
python pipeline/build_labels.py
python pipeline/build_training_data.py
python pipeline/train_model.py
python pipeline/predict.py --snowfall_cm 15 --min_temp_c -8 --max_wind_kmh 40 --duration_days 2
```

Or run individual scripts:
```bash
python pipeline/collect_centerlines.py   # Run first — base road network
python pipeline/collect_osm_pgh.py       # Run first — OSM data
python pipeline/collect_elevation.py     # Needs data/centerlines_pgh.csv
python pipeline/build_dataset.py         # Needs all CSVs in data/
python pipeline/predict.py --use_forecast  # Use NWS forecast JSON
```

## Dependencies

```
requirements.txt: requests, pandas, numpy, scipy, lightgbm, scikit-learn, pyarrow
```

`googlemaps` is only needed for the legacy `legacy/roads_demo.py` script. `libomp` is required on macOS for LightGBM (`brew install libomp`).

## Data Notes

- `dataset_prediction_ready.csv` has ~28,805 rows but only ~17,550 unique `objectid`s (duplicate segments from multi-segment roads). ML scripts deduplicate by objectid, keeping first occurrence.
- `storm_labels.csv` is ~91MB (~2.2M rows). `build_training_data.py` reads it in 500K-row chunks to avoid OOM.
- `training_matrix.parquet` uses float32 and chunked Parquet writing via PyArrow for memory efficiency.
- Plow activity data only covers 3 of 126 storms (2018-2020 events). `storm_plow_gap` is NaN for the other 123 storms; label weights are renormalized accordingly.

## External API Notes

WPRDC resource IDs and PennDOT ArcGIS service URLs change over time. If a script starts returning 404/409 errors:
- **WPRDC**: Use `package_show` or `package_search` to find current resource IDs: `https://data.wprdc.org/api/3/action/package_show?id=<dataset-slug>`
- **PennDOT ArcGIS**: Check the services directory at `https://gis.penndot.pa.gov/arcgis/rest/services?f=json`. The old `PennShare/` folder was reorganized into `winterconditions/` and `opendata/`.
- **WPRDC 311 data**: New format uses lowercase column names (`request_type_name`, `create_date_et`, `latitude`, `longitude`).
- **WPRDC crash data**: Some records have DMS-format lat/lng (e.g. `40 26:33.550`). Use `pd.to_numeric(errors="coerce")` before spatial operations.

## Conventions

- Each script is standalone with its own `main()` function
- All pipeline scripts resolve paths via `DATA_DIR` relative to the project root
- CSV output with `csv.DictWriter`, JSON with 2-space indent
- `data_source` column in every collection output CSV
- Coordinates as `mid_lat`, `mid_lng`
- Cache: skip re-fetch if output CSV already exists in `data/`
- Progress: `"Fetching batch 15/190..."`
- Retry on HTTP 429/5xx with exponential backoff
- Stream large downloads with `requests.get(url, stream=True)`
- Generated data files go in `data/` (gitignored)
- ML scripts use `build_training_data.encode_static_features()` as the single source of truth for feature encoding (shared between training and prediction)
