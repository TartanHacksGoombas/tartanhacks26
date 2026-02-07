"""
Collect PennDOT snow route classifications from ArcGIS REST service.

Queries PennDOT RMS data filtered to Allegheny County (CTY_CODE='02'),
paginated at 1,000 records per request.

Source: PennDOT RMS ArcGIS service
Output: penndot_snow_routes_pgh.csv

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

# PennDOT State Roads ArcGIS REST endpoint
BASE_URL = "https://gis.penndot.pa.gov/arcgis/rest/services/opendata/roadwaysegments/MapServer/0/query"

OUTPUT_CSV = os.path.join(DATA_DIR, "penndot_snow_routes_pgh.csv")

FIELDNAMES = [
    "route_no", "street_name", "lane_count", "aadt",
    "pavement_condition", "roughness_index", "mid_lat", "mid_lng",
    "data_source",
]

PAGE_SIZE = 1000
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


def fetch_all_routes():
    """Fetch all PennDOT routes in Allegheny County with pagination."""
    all_features = []
    offset = 0

    print("Fetching PennDOT snow routes for Allegheny County...")
    while True:
        params = {
            "where": "CTY_CODE='02'",
            "outFields": "ST_RT_NO,STREET_NAME,LANE_CNT,CUR_AADT,OVERALL_PVMNT_IDX,ROUGH_INDX,CTY_CODE",
            "outSR": "4326",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }

        resp = fetch_with_retry(BASE_URL, params=params)
        data = resp.json()

        if "error" in data:
            print(f"  Error: {data['error'].get('message', 'unknown')}")
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        page = offset // PAGE_SIZE + 1
        print(f"  Page {page}: {len(features)} features (total: {len(all_features)})")

        # Check if there are more records
        if len(features) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)

    print(f"  Total: {len(all_features)} route segments")
    return all_features


def extract_midpoint(geometry):
    """Extract midpoint from ArcGIS polyline geometry."""
    paths = geometry.get("paths", [])
    if paths and paths[0]:
        mid_idx = len(paths[0]) // 2
        point = paths[0][mid_idx]
        return round(point[1], 6), round(point[0], 6)  # lat, lng
    return "", ""


def build_rows(features):
    """Convert ArcGIS features to output rows."""
    rows = []
    for f in features:
        attrs = f.get("attributes", {})
        geom = f.get("geometry", {})
        a = {k.upper(): v for k, v in attrs.items()}

        mid_lat, mid_lng = extract_midpoint(geom)

        rows.append({
            "route_no": a.get("ST_RT_NO", ""),
            "street_name": (a.get("STREET_NAME") or "").strip(),
            "lane_count": a.get("LANE_CNT", ""),
            "aadt": a.get("CUR_AADT", ""),
            "pavement_condition": a.get("OVERALL_PVMNT_IDX", ""),
            "roughness_index": a.get("ROUGH_INDX", ""),
            "mid_lat": mid_lat,
            "mid_lng": mid_lng,
            "data_source": "penndot_rms",
        })

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    features = fetch_all_routes()
    rows = build_rows(features)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} PennDOT route segments to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
