"""
Collect PennDOT winter road conditions from their GIS service.

Queries 5 condition layers: impassable, icy, snowpacked, snowslush, wetfreezing.
Data is only populated during active storms — handles empty results gracefully.

Source: PennDOT Winter Conditions GIS
Output: penndot_conditions_pgh.csv (may be empty between storms)

Requires: pip install requests
"""

import csv
import os
import time
import requests

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# PennDOT Winter Conditions ArcGIS service layers
BASE_URL = "https://gis.penndot.pa.gov/arcgis/rest/services/winterconditions/winterconditions/MapServer"

LAYERS = {
    0: "impassable",
    1: "icy",
    2: "snowpacked",
    3: "snowslush",
    4: "wetfreezing",
}

# Allegheny County bounding box (approximate)
BBOX = "-80.10,40.36,-79.86,40.50"

OUTPUT_CSV = os.path.join(DATA_DIR, "penndot_conditions_pgh.csv")

FIELDNAMES = [
    "condition_type", "route", "segment_desc", "county",
    "latitude", "longitude", "last_updated", "data_source",
]

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, params=None):
    """Fetch with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = RETRY_BACKOFF ** (attempt + 1)
                print(f"  HTTP {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF ** (attempt + 1)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after {MAX_RETRIES} retries")


def query_layer(layer_id, condition_name):
    """Query a single PennDOT winter condition layer."""
    url = f"{BASE_URL}/{layer_id}/query"
    params = {
        "where": "1=1",
        "geometry": BBOX,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "json",
    }

    print(f"  Querying layer {layer_id} ({condition_name})...")
    try:
        resp = fetch_with_retry(url, params=params)
        data = resp.json()

        if "error" in data:
            print(f"    Error: {data['error'].get('message', 'unknown')}")
            return []

        features = data.get("features", [])
        print(f"    Found {len(features)} features")
        return features
    except Exception as e:
        print(f"    Error querying layer {layer_id}: {e}")
        return []


def extract_rows(features, condition_name):
    """Extract rows from ArcGIS features."""
    rows = []
    for f in features:
        attrs = f.get("attributes", {})
        geom = f.get("geometry", {})

        # Try to get a representative point
        lat, lng = "", ""
        if "y" in geom and "x" in geom:
            lat, lng = geom["y"], geom["x"]
        elif "paths" in geom:
            # Polyline — use midpoint of first path
            paths = geom["paths"]
            if paths and paths[0]:
                mid_idx = len(paths[0]) // 2
                lng, lat = paths[0][mid_idx]

        # Normalize attribute keys
        a = {k.lower(): v for k, v in attrs.items()}

        rows.append({
            "condition_type": condition_name,
            "route": a.get("route") or a.get("st_rt_no") or a.get("road_name", ""),
            "segment_desc": a.get("seg_desc") or a.get("description") or a.get("seg_bgn_desc", ""),
            "county": a.get("county") or a.get("cty_code", ""),
            "latitude": lat,
            "longitude": lng,
            "last_updated": a.get("last_updt") or a.get("update_date", ""),
            "data_source": "penndot_winter",
        })

    return rows


def main():
    print("Fetching PennDOT winter road conditions...")
    print("(Data is only populated during active storms — empty results are normal)\n")

    all_rows = []

    for layer_id, condition_name in LAYERS.items():
        features = query_layer(layer_id, condition_name)
        rows = extract_rows(features, condition_name)
        all_rows.extend(rows)
        time.sleep(0.5)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} condition records to {OUTPUT_CSV}")
    if not all_rows:
        print("  (empty — no active winter conditions reported)")


if __name__ == "__main__":
    main()
