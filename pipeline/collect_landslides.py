"""
Collect landslide data from WPRDC.

Two datasets:
  - Landslide Prone Areas (37 records): 09b067e2-80d8-4704-8b7f-cb7e73096e87
  - Landslide Inventory (7,242 records): ae652677-baa7-4e42-9cf4-b5dffe07cf84

Output: landslides_pgh.csv

Requires: pip install requests
"""

import csv
import json
import os
import time
import requests

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

RESOURCE_LANDSLIDES = "dde1f413-c849-413c-b791-0f861bf219ce"

CKAN_BASE = "https://data.wprdc.org/api/3/action"

OUTPUT_CSV = os.path.join(DATA_DIR, "landslides_pgh.csv")

FIELDNAMES = ["landslide_id", "latitude", "longitude", "source_study", "data_source"]

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


def fetch_ckan_records(resource_id, label):
    """Fetch all records from a CKAN resource."""
    all_records = []
    limit = 5000
    offset = 0

    print(f"Fetching {label}...")
    while True:
        sql = f'SELECT * FROM "{resource_id}" LIMIT {limit} OFFSET {offset}'
        resp = fetch_with_retry(
            f"{CKAN_BASE}/datastore_search_sql",
            params={"sql": sql},
        )
        records = resp.json().get("result", {}).get("records", [])
        if not records:
            break
        all_records.extend(records)
        offset += limit

    print(f"  Found {len(all_records)} records")
    return all_records


def extract_lat_lng(rec):
    """Try to extract lat/lng from various possible field names."""
    r = {k.strip().lower(): v for k, v in rec.items()}

    lat = r.get("latitude") or r.get("lat") or r.get("y") or ""
    lng = r.get("longitude") or r.get("lng") or r.get("lon") or r.get("x") or ""

    # Try to parse from a geometry/point field
    if not lat or not lng:
        for key in ("geom", "geometry", "the_geom", "shape"):
            val = r.get(key, "")
            if isinstance(val, str) and "POINT" in val.upper():
                # Parse "POINT(lng lat)" or similar
                import re
                match = re.search(r"[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*", val)
                if match:
                    parts = match.group().split()
                    lng, lat = float(parts[0]), float(parts[1])
                    break

    try:
        return float(lat), float(lng)
    except (ValueError, TypeError):
        return None, None


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    # Fetch landslide dataset (global — we filter for Pittsburgh area)
    all_records = fetch_ckan_records(RESOURCE_LANDSLIDES, "Global Landslides")

    rows = []
    idx = 0

    # Pittsburgh bounding box (approximate)
    PGH_LAT_MIN, PGH_LAT_MAX = 40.30, 40.55
    PGH_LNG_MIN, PGH_LNG_MAX = -80.15, -79.80

    for rec in all_records:
        r = {k.strip().lower(): v for k, v in rec.items()}
        lat, lng = extract_lat_lng(rec)
        # Filter to Pittsburgh area
        if lat and lng:
            if not (PGH_LAT_MIN <= lat <= PGH_LAT_MAX and PGH_LNG_MIN <= lng <= PGH_LNG_MAX):
                continue
        else:
            # Keep records from Pennsylvania even without exact coords
            div = (r.get("div_name") or r.get("state_province") or "").strip()
            if "Pennsylvania" not in div and "PA" not in div:
                continue
        idx += 1
        rows.append({
            "landslide_id": r.get("ev_id") or r.get("objectid") or r.get("id") or r.get("_id") or idx,
            "latitude": lat or "",
            "longitude": lng or "",
            "source_study": r.get("ls_cat") or r.get("source_study") or "global",
            "data_source": "wprdc_landslides",
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with_coords = sum(1 for r in rows if r["latitude"] and r["longitude"])
    print(f"\nSaved {len(rows)} landslide records to {OUTPUT_CSV} ({with_coords} with coordinates)")


if __name__ == "__main__":
    main()
