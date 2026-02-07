# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WinterWise — Pittsburgh winter road closure prediction and route planning. Full-stack web app that predicts which roads will close during snowstorms and plans routes around them. Built for TartanHacks 2026.

## Development Commands

```bash
# ── Frontend (React + Vite) ──
cd frontend && npm install && cp .env.example .env
npm run dev          # Dev server at http://localhost:5173
npm run build        # TypeScript check + Vite build → dist/
npm run preview      # Preview production build

# ── Backend (Express + TypeScript) ──
cd backend && npm install && cp .env.example .env
npm run dev          # tsx watch at http://localhost:4000 (hot reload)
npm run build        # tsc → dist/
npm run start        # node dist/index.js (needs build first)

# ── ML Pipeline (Python) ──
pip install -r requirements.txt   # libomp needed on macOS: brew install libomp
python pipeline/collect_all.py    # Full pipeline: data collection → ML training
python pipeline/predict.py --snowfall_cm 15 --min_temp_c -8 --max_wind_kmh 40 --duration_days 2

# Lingering snow (no new snow, but roads still buried):
python pipeline/predict.py --snowfall_cm 0 --recent_snowfall_cm 20 --hours_since_snow 18 --min_temp_c -5
```

## Architecture

Three independent layers connected via REST API and Python child_process:

```
frontend/ (React + MapLibre + Tailwind + Vite)
    ↕ REST API (fetch via utils/api.ts)
backend/ (Express + TypeScript, in-memory spatial store)
    ↕ execFile("python3", predict_batch.py)
pipeline/ (Python: 18 collection scripts + 7 ML scripts)
    → data/ (CSVs, Parquet, GeoJSON, LightGBM models)
```

### Frontend (`frontend/`)

React 18 SPA with MapLibre GL for interactive map of Pittsburgh road segments colored by closure risk (green/yellow/orange/red). Key components:
- `App.tsx` — Map initialization, conditions loading on viewport move, prediction triggering
- `UnifiedDayBar.tsx` — 7-day forecast bar with expandable hourly precipitation graphs and auto-snow detection (replaced separate WeatherBar/TimeSlider)
- `NavigationPanel.tsx` — Sidebar with route search (Google Places autocomplete) and segment info
- `LocationButton.tsx` — User geolocation tracking
- `utils/api.ts` — Fetch wrapper for backend API

Map renders road segments at zoom-dependent detail: major roads always visible, tertiary at zoom 13+, residential at zoom 14+. Uses CartoDB Voyager tiles as basemap. During navigation, risk visibility filters to only the active route segments (wider lines, full opacity).

### Backend (`backend/`)

Express API with in-memory spatial data store (no database in current architecture). On startup, loads `data/centerlines_pgh.geojson` into a grid-based spatial index (0.002° cells ~200m) for fast bbox queries.

Key endpoints:
- `GET /v1/conditions?bbox=...&kind=road&day_offset=0` — Road segments with risk in viewport
- `GET /v1/route-safest?from=lng,lat&to=lng,lat` — Risk-aware A* routing (builds road network graph on startup)
- `POST /v1/route-risk` — Route risk analysis (samples coordinates every ~50m, matches to road segments)
- `POST /v1/predict` — Triggers Python `predict_batch.py` via `execFile`, stores results in memory
- `GET /v1/maps/geocode`, `/autocomplete`, `/place-details`, `/reverse-geocode`, `/directions` — Google Maps API proxies (requires `GOOGLE_MAPS_KEY` env var)
- `GET /healthz` — Health check

**Routing service** (`backend/src/services/graph.ts`): On startup, builds a bidirectional road network graph from GeoJSON. Auto-stitches disconnected endpoints within 15m. A* pathfinding uses risk-aware edge costs: `lengthM * (1 + RISK_WEIGHT * risk)` where `RISK_WEIGHT = 10`. Generates turn-by-turn instructions via bearing-based turn detection.

Config (`backend/src/config.ts`): resolves `DATA_DIR` and `PIPELINE_DIR` relative to project root. Zod-validated env vars: `PORT` (4000), `CORS_ORIGIN` (http://localhost:5173), `GOOGLE_MAPS_KEY` (optional).

**Integration flow:** Frontend WeatherBar → `POST /v1/predict` → backend spawns `python3 pipeline/predict_batch.py` → parses JSON stdout → stores in `predictionCache` (Map<dayOffset, Map<objectid, Prediction>>) → frontend reloads conditions from viewport.

### ML Pipeline (`pipeline/`)

All scripts are standalone with `main()` functions, resolve paths via `DATA_DIR` relative to project root. Output CSVs to `data/` with a `data_source` column.

**Phase A — Road Network:** `collect_centerlines.py` (WPRDC ~18,665 segments), `collect_osm_pgh.py` (OSM bbox ~36,439 segments)

**Phase B — Environmental Risk Factors (needs A):** `collect_elevation.py`, `collect_slopes.py`, `collect_bridges.py`, `collect_landslides.py`

**Phase C — Weather (independent):** `collect_weather_historical.py` (Open-Meteo, 11 winters), `collect_weather_realtime.py` (NWS), `collect_penndot_winter.py`

**Phase D — Historical Incidents (independent):** `collect_crashes.py`, `collect_311_snow.py`, `collect_plow_activity.py`, `collect_penndot_snow_routes.py`, `collect_domi_closures.py`, `collect_snow_emergency_routes.py`, `collect_511pa_events.py`, `collect_penndot_press.py`

**Phase E — Master Join:** `build_dataset.py` → `data/dataset_prediction_ready.csv` (~28,805 rows, ~17,550 unique objectids). Uses scipy KDTree for spatial joins.

**Phase F — ML Pipeline (needs E + C):**
- `build_labels.py` — Detects 126 storms from weather (snowfall >= 1cm), expands to per-day rows via `expand_storm_days()`. Labels segments using 4 proxy sources (311 complaints, crashes, plow gaps, DOMI closures) with spatial `query_ball_point` joins. Outputs `storm_events.csv`, `storm_days.csv`, `storm_labels.csv` (~10M rows).
- `build_training_data.py` — Encodes 31 static + 6 weather + 6 temporal + 8 interaction features. Auto-detects `storm_days.csv` for per-day granularity. Writes `training_matrix.parquet` via chunked PyArrow.
- `train_model.py` — LightGBM regression (35 features, weather excluded). Outputs `model_lgbm.txt`.
- `train_ranker.py` — LightGBM LambdaRank (41 features including weather). Primary model. Outputs `ranker_lgbm.txt`.
- `evaluate.py` — Temporal holdout evaluation (2023-25 seasons).
- `predict.py` — Prefers ranker, falls back to regression. CLI args or NWS forecast JSON. Supports lingering snow decay. Outputs `predictions_latest.csv` + `predictions_latest.json` (GeoJSON).
- `predict_batch.py` — Runs predictions for multiple day offsets (0–6), outputs JSON to stdout for backend consumption.
- `collect_all.py` — Orchestrator, runs Phases A→F in dependency order.

## Key Patterns

- **Feature encoding:** `build_training_data.encode_static_features()` is the single source of truth, shared between training and prediction
- **Deduplication:** Dataset has ~28,805 rows but ~17,550 unique objectids. All ML scripts and predict.py deduplicate by objectid, keeping first occurrence
- **Label normalization:** `log1p(x) / log1p(max)` — zeros map to 0.0 (no inflation)
- **Spatial operations:** `query_ball_point` for labeling (all segments within radius), KDTree for dataset joins
- **Memory efficiency:** `storm_labels.csv` read in 500K-row chunks; `training_matrix.parquet` written via chunked PyArrow
- **Prediction calibration:** Sigmoid ranking + weather-severity power transform (light snow compresses scores, blizzard spreads them)
- **Lingering snow:** `compute_effective_weather()` in predict.py uses thermal decay `k(T) = max(0.005, 0.02 + 0.013 * max(0, T))` + plow linear decay
- **Backend spatial index:** Grid-based (0.002°) in-memory store, no database — loads GeoJSON at startup
- **Road network graph:** Built at startup from GeoJSON with auto-stitching of disconnected endpoints and connected component analysis
- **Risk categories:** `very_low` → `open`, `low` → `low_risk`, `moderate` → `moderate_risk`, `high`/`very_high` → `closed` (see `backend/src/utils/risk.ts`)
- **Pipeline caching:** Collection scripts skip re-fetch if output CSV already exists in `data/`

## Data Notes

- `storm_labels.csv` has per-day granularity: (objectid, storm_id, day_offset) with risk_score and label variants
- `storm_days.csv` (~500-630 rows): temporal features per (storm_id, day_offset)
- `training_matrix.parquet`: ~57 columns, float32
- Plow data only covers 3/126 storms (2018-2020); label weights renormalized for missing plow
- Weather column names: `snowfall_cm`, `temp_min_c`, `temp_max_c`, `wind_max_kmh`, `precip_mm`
- Some crash lat/lng in DMS format — use `pd.to_numeric(errors="coerce")`

## External API Notes

WPRDC resource IDs and PennDOT ArcGIS URLs change over time. If scripts return 404/409:
- **WPRDC:** `https://data.wprdc.org/api/3/action/package_show?id=<dataset-slug>` to find current resource IDs
- **PennDOT ArcGIS:** Check `https://gis.penndot.pa.gov/arcgis/rest/services?f=json` (old `PennShare/` reorganized into `winterconditions/` and `opendata/`)
- **WPRDC 311:** Lowercase column names (`request_type_name`, `create_date_et`, `latitude`, `longitude`)

## Conventions

- Pipeline scripts: standalone `main()`, `DATA_DIR` relative to project root, `csv.DictWriter`, `data_source` column, coordinates as `mid_lat`/`mid_lng`
- HTTP: retry on 429/5xx with exponential backoff, stream large downloads
- Backend: Zod schema validation for all inputs, TypeScript strict mode
- Frontend: React hooks for state, Tailwind for styling, Vite for bundling
