# WinterWise — What to Do Next

Hackathon goal: **A beautiful app that uses Snowstorm (weather), road map data, snowplow (and other data) to visualize road closure and open roads/sidewalks for users in Pittsburgh.**

---

## What You Already Have

- **Map**: OSM basemap of Pittsburgh; segments colored green (open) / amber (caution) / red (closed).
- **Backend**: Conditions API, ingest API (`POST /v1/admin/ingest/:provider`), scoring that combines closures, winter conditions, weather alerts, and plow passes.
- **Schema**: `segments` (roads/sidewalks), `source_events` (closure, winter_condition, weather_alert, plow_pass), `segment_status_current` (computed scores).
- **Seed data**: A handful of sample segments; run `002_seed_pittsburgh.sql` and recompute to see colors.

---

## Prioritized Next Steps

### 1. **Real road (and sidewalk) network** — so the map shows all of Pittsburgh

Right now only a few seed segments exist. To show the full network:

- **Option A – OpenStreetMap**: Export Pittsburgh-area roads (and optionally sidewalks) from OSM, convert to LineStrings, insert into `segments`. Tools: [Overpass API](https://overpass-api.de/) (query `way["highway"]` in a Pittsburgh bbox) or [osmium](https://osmcode.org/osmium-tool/), then a small script to INSERT with `ST_GeomFromGeoJSON`.
- **Option B – PennDOT**: If you get RCRS/511 API access, use their road geometry or link events to your seed segments by name/ref.

**Hackathon shortcut**: Use a pre-built GeoJSON of Pittsburgh roads (e.g. from [Allegheny County Open Data](https://data.wprdc.org/) or OSM extract), then a one-off script to `INSERT INTO segments (kind, name, length_m, geom) SELECT ... FROM ST_GeomFromGeoJSON(...)`.

---

### 2. **Weather / “Snowstorm” data → `weather_alert` and `winter_condition`**

- **NWS (api.weather.gov)**  
  - Active alerts: `GET https://api.weather.gov/alerts/active?zone=PAZ021` (Pittsburgh/Allegheny).  
  - Send a **User-Agent** header (e.g. `SnowRoutePittsburgh/1.0 (hackathon)`).  
  - Response includes GeoJSON polygons for each alert; map to `source_events` with `event_type: "weather_alert"`, optional `winter_condition` for winter-specific alerts (e.g. Winter Storm Warning).  
  - Script in repo: `backend/scripts/ingest-weather-alerts.ts` (or `.js`) that fetches, maps to your ingest payload, and `POST`s to `POST /v1/admin/ingest/weather.gov`.

- **Optional**: Grid endpoint for precipitation/snow (e.g. by point or county) to derive “snow on the ground” and feed into severity or a separate `winter_condition` event.

---

### 3. **Snowplow and closure data**

- **511 PA / PennDOT RCRS**  
  - [PennDOT developer resources](https://www.pa.gov/agencies/penndot/programs-and-doing-business/online-services/developer-resources-documentation-api.html): “Road Condition Reporting System (RCRS) Event Data Service” — `liveEvents`, `plannedEvents`, `winterConditions`.  
  - Request API access, then map:  
    - Closures → `event_type: "closure"` with geometry or segment refs.  
    - Winter conditions → `event_type: "winter_condition"`.  
  - New ingest provider, e.g. `POST /v1/admin/ingest/511pa`.

- **Plow passes**  
  - If Pittsburgh or Allegheny County exposes plow GPS/history, ingest as `event_type: "plow_pass"` with `starts_at` and point/line `geom`. Your scoring already uses “recent plow within 6h” to reduce penalty.  
  - If no real API: for demo, a script that posts synthetic `plow_pass` events along a few routes so some segments show as “recently plowed.”

---

### 4. **Frontend polish (beautiful, clear for users)**

- **Legend**: “Open / Caution / Closed” with the same green / amber / red and a one-line explanation.
- **Safest route**: You have `GET /v1/route-safest?from=lng,lat&to=lng,lat&kind=road|sidewalk`. Add a “Get safest route” control: two clicks or two address/point inputs, then draw the route on the map (and optionally list segments with status).
- **Last updated**: Show “Conditions updated at &lt;time&gt;” from the API or from `segment_status_current.updated_at`.
- **Mobile**: Ensure the panel and map are usable on small screens (stack panel below map or collapsible).
- **Empty state**: If no segments in view, show “No segment data in this area” instead of a blank map.

---

### 5. **Demo and docs for judges**

- **One-command demo**:  
  - Start backend + frontend.  
  - Run seed SQL if needed.  
  - Run ingest script(s): e.g. weather alerts, then optional 511 + synthetic plow.  
  - `POST /v1/admin/recompute-scores`.  
  - Open app and show Pittsburgh with colored network and (if implemented) safest route.
- **README**: Add a “Hackathon demo” section: what data sources you use (Snowstorm/NWS, 511, plow if any), how to run the ingest and recompute, and one sentence on how scoring works (closures + winter + weather + plow + slope → open/caution/closed).

---

## Suggested data sources summary

| Source            | Use in app              | Event type(s)        | Notes |
|------------------|-------------------------|----------------------|-------|
| **api.weather.gov** | Snowstorm / weather     | `weather_alert`, `winter_condition` | Free; User-Agent required. |
| **511 PA / PennDOT RCRS** | Closures, winter conditions | `closure`, `winter_condition` | Request API access. |
| **Pittsburgh / County plow** | Plow activity           | `plow_pass`          | If available; else synthetic for demo. |
| **OpenStreetMap** | Road/sidewalk network    | Populate `segments`  | Overpass or extract + import script. |
| **Allegheny Co. / WPRDC** | Roads, sidewalks, other  | Populate `segments`  | Optional alternative to OSM. |

---

## Quick wins for the next hour

1. **Run the weather ingest script** (once added) and recompute — see weather alerts affecting segment colors.  
2. **Add the map legend** (Open / Caution / Closed) to the frontend.  
3. **Document in README** the one-time seed + ingest + recompute flow so anyone can reproduce the demo.
