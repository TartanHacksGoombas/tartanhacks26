# ML predictions API — Road closure probability

The app is set up to consume **road closure probabilities** from your ML model. The model can use current weather and historical snowstorm data (severity, coverage, road closures) and output a **closure probability per segment** (0 = open, 1 = closed). The map then shows each segment in one of four colors: **green**, **yellow**, **orange**, **red**.

---

## Probability bands (backend mapping)

| Closure probability | Label         | Map color  |
|---------------------|---------------|------------|
| `0 ≤ p < 0.25`      | `open`        | Green      |
| `0.25 ≤ p < 0.5`    | `low_risk`    | Yellow     |
| `0.5 ≤ p < 0.75`    | `moderate_risk` | Orange   |
| `0.75 ≤ p ≤ 1`      | `closed`      | Red        |

The backend maps your probability to these labels when you POST predictions.

---

## Endpoint

**`POST /v1/admin/ingest/ml-predictions`**

- **Auth:** Header `x-admin-token: <ADMIN_TOKEN>` (same as other admin endpoints).
- **Body (JSON):**

```json
{
  "predictions": [
    { "segment_id": 1, "closure_probability": 0.1 },
    { "segment_id": 2, "closure_probability": 0.85 },
    { "segment_id": 3, "closure_probability": 0.45 }
  ]
}
```

- **Fields:**
  - `segment_id` (integer, required): Primary key of a row in the `segments` table (road/sidewalk segment).
  - `closure_probability` (number, required): Probability of closure in `[0, 1]`. 0 = no closure, 1 = closed.

- **Response:** `{ "ok": true, "updated": 3 }`

- **Behavior:** For each prediction, the backend updates `segment_status_current` for that segment: it sets `closure_probability`, derives `label` from the bands above, and sets `score = (1 - p) * 100`. Existing rule-based status for that segment is overwritten.

---

## Getting segment IDs

- **Conditions API:** `GET /v1/conditions?bbox=minLng,minLat,maxLng,maxLat&kind=road` returns a GeoJSON FeatureCollection. Each feature has `id` (segment_id) and properties (name, current label, etc.). Your ML pipeline can use this to know which segments exist and their IDs.
- **Database:** Table `segments` has `id` (BIGSERIAL). You can export segment IDs (and optionally geometry) for the city for batch prediction.

---

## Example: curl

```bash
curl -X POST http://localhost:4000/v1/admin/ingest/ml-predictions \
  -H "Content-Type: application/json" \
  -H "x-admin-token: YOUR_ADMIN_TOKEN" \
  -d '{
    "predictions": [
      { "segment_id": 1, "closure_probability": 0.05 },
      { "segment_id": 2, "closure_probability": 0.6 }
    ]
  }'
```

After posting, refresh the map; segments 1 and 2 will appear green and orange respectively (assuming they are in view).

---

## Database (optional)

If you run the migration `backend/sql/003_ml_predictions.sql`:

- `segment_status_current` gets an optional column `closure_probability` (0–1). The ingest endpoint stores it there.
- The conditions API returns `closureProbability` in each feature’s properties when present (for tooltips / popups).

The frontend shows “Closure prob.: X%” in the segment popup when `closureProbability` is set.
