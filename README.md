# Snow Route Pittsburgh MVP

Road-first winter passability map scaffold with:

- `frontend/`: React + TypeScript + Tailwind + MapLibre
- `backend/`: Node + Express + TypeScript
- `backend/sql/001_init.sql`: Supabase/Postgres + PostGIS schema

## 1) Database (Supabase)

1. Open your Supabase SQL editor.
2. Run `backend/sql/001_init.sql`.
3. Run `backend/sql/002_seed_pittsburgh.sql` so the map has sample segments.
4. Run `backend/sql/003_ml_predictions.sql` to add 4-level labels and `closure_probability` for ML (green/yellow/orange/red).
5. Ensure your project has PostGIS enabled (the migration includes `CREATE EXTENSION postgis`).

## 2) Backend setup

```bash
cd /Users/edwin/Documents/coding/tartanhacks26/backend
cp .env.example .env
npm install
npm run dev
```

Required env vars:

- `DATABASE_URL`: Supabase Postgres connection string
- `ADMIN_TOKEN`: token for protected ingest/recompute endpoints
- `CORS_ORIGIN`: frontend origin (default `http://localhost:5173`)

API base URL: `http://localhost:4000`

### Backend endpoints

- `GET /healthz`
- `GET /v1/conditions?bbox=minLng,minLat,maxLng,maxLat&kind=road|sidewalk|all`
- `GET /v1/segment/:id`
- `GET /v1/closures/active?bbox=minLng,minLat,maxLng,maxLat`
- `GET /v1/timeline?bbox=minLng,minLat,maxLng,maxLat&kind=road|sidewalk|all&at=ISO_TIMESTAMP`
- `GET /v1/route-safest?from=lng,lat&to=lng,lat&kind=road|sidewalk`
- `POST /v1/admin/ingest/:provider` (header `x-admin-token`)
- `POST /v1/admin/ingest/ml-predictions` (header `x-admin-token`) — ML closure probability per segment; see **docs/ML_API.md**
- `POST /v1/admin/recompute-scores` (header `x-admin-token`)

## 3) Frontend setup

```bash
cd /Users/edwin/Documents/coding/tartanhacks26/frontend
cp .env.example .env
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` and loads map segments from backend.

## 4) Ingest payload example

`POST /v1/admin/ingest/penndot`

```json
{
  "events": [
    {
      "source_event_id": "event-123",
      "event_type": "closure",
      "severity": 5,
      "starts_at": "2026-02-07T16:00:00Z",
      "ends_at": "2026-02-07T22:00:00Z",
      "props": {
        "name": "I-376 closure"
      },
      "geom": {
        "type": "LineString",
        "coordinates": [
          [-79.99, 40.44],
          [-79.98, 40.44]
        ]
      }
    }
  ]
}
```

Then recompute:

```bash
curl -X POST http://localhost:4000/v1/admin/recompute-scores \
  -H "x-admin-token: change-me"
```

## 5) Sidewalk phase

Schema and API already support sidewalks using `segments.kind = 'sidewalk'`.

Next step is to ingest sidewalk network data and sidewalk-impact events, then query with:

- `GET /v1/conditions?...&kind=sidewalk`

## Hackathon demo (Snowstorm + roads + plow → open/closed)

1. Run DB migrations and seed (steps 1–2 above), then start backend and frontend.
2. Ingest NWS weather alerts (Pittsburgh zone):  
   `cd backend && ADMIN_TOKEN=your-token npm run ingest:weather`
3. Recompute scores:  
   `curl -X POST http://localhost:4000/v1/admin/recompute-scores -H "x-admin-token: your-token"`
4. Open the app: roads colored green/amber/red by passability; legend in the sidebar.

See **NEXT_STEPS.md** for full roadmap: real road network (OSM), 511 PA closures, plow data, and frontend polish.

## Notes

- `route-safest` currently returns a corridor-based safest segment set (`mode: "corridor_mvp"`). Replace with pgRouting for turn-by-turn routing in the next iteration.
