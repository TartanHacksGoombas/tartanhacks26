# WinterWise

Pittsburgh winter road closure prediction and risk-aware route planning. Predicts which roads are likely to close during snowstorms and routes around them using ML-powered risk scores. Built for TartanHacks 2026.

- `frontend/` — React + TypeScript + Tailwind + MapLibre GL
- `backend/` — Node + Express + TypeScript (in-memory spatial store)
- `pipeline/` — Python ML pipeline (data collection, LightGBM training, prediction)

## Quick Start

### 1) ML Pipeline (Python)

```bash
pip install -r requirements.txt   # macOS: brew install libomp (for LightGBM)
python pipeline/collect_all.py    # Full pipeline: data collection → training → model
```

This produces `data/centerlines_pgh.geojson` (road network) and `data/ranker_lgbm.txt` (ML model), which the backend needs at startup.

### 2) Backend

```bash
cd backend
cp .env.example .env              # Edit to add GOOGLE_MAPS_KEY if needed
npm install
npm run dev                       # http://localhost:4000
```

Environment variables:
| Variable | Default | Description |
|---|---|---|
| `PORT` | `4000` | Server port |
| `CORS_ORIGIN` | `http://localhost:5173` | Frontend origin |
| `GOOGLE_MAPS_KEY` | _(none)_ | Enables geocoding/directions proxy endpoints |

On startup the backend loads `data/centerlines_pgh.geojson` into a grid-based spatial index and builds a road network graph for A* routing.

### 3) Frontend

```bash
cd frontend
cp .env.example .env              # VITE_API_BASE_URL=http://localhost:4000
npm install
npm run dev                       # http://localhost:5173
```

### Running predictions

From the UI: adjust snowfall/temperature/wind in the weather bar and predictions run automatically.

From the CLI:
```bash
# Active storm
python pipeline/predict.py --snowfall_cm 15 --min_temp_c -8 --max_wind_kmh 40 --duration_days 2

# Lingering snow (no new snow, roads still buried)
python pipeline/predict.py --snowfall_cm 0 --recent_snowfall_cm 20 --hours_since_snow 18 --min_temp_c -5
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Health check |
| `GET` | `/v1/conditions?bbox=...&kind=road&day_offset=0` | Road segments with risk in viewport |
| `GET` | `/v1/route-safest?from=lng,lat&to=lng,lat` | Risk-aware A* routing with turn-by-turn |
| `POST` | `/v1/route-risk` | Risk analysis for a given route |
| `POST` | `/v1/predict` | Trigger ML predictions (spawns Python) |
| `GET` | `/v1/maps/geocode` | Google Maps geocoding proxy |
| `GET` | `/v1/maps/autocomplete` | Google Places autocomplete proxy |
| `GET` | `/v1/maps/place-details` | Place ID → coordinates |
| `GET` | `/v1/maps/reverse-geocode` | Coordinates → address |
| `GET` | `/v1/maps/directions` | Google directions proxy |

## Architecture

```
frontend/ (React + MapLibre + Tailwind + Vite)
    ↕ REST API
backend/ (Express + TypeScript, in-memory spatial index + road graph)
    ↕ execFile("python3", predict_batch.py)
pipeline/ (Python: 18 collection scripts + 7 ML scripts)
    → data/ (CSVs, Parquet, GeoJSON, LightGBM models)
```

**Prediction flow:** Frontend weather bar → `POST /v1/predict` → backend spawns `python3 pipeline/predict_batch.py` → parses JSON stdout → stores in `predictionCache` → frontend reloads conditions.

**Routing:** A* pathfinding on the road network graph with risk-aware edge costs (`length * (1 + 10 * risk)`). Disconnected road segments are auto-stitched within 15m.

**ML model:** LightGBM LambdaRank trained on 11 winters of Pittsburgh data — 311 complaints, crash reports, plow activity, and DOMI closures as proxy labels for road closure risk.
