"""
Collect Allegheny County Street Centerlines from WPRDC.

Downloads the full dataset via CKAN datastore_search_sql and also fetches
the GeoJSON to extract midpoint lat/lng per segment.

Source: https://data.wprdc.org/dataset/allegheny-county-street-centerlines
Resource ID: dd13c2e7-4a83-40aa-a040-08a27385759b

Output: centerlines_pgh.csv (~19,000 rows)

Requires: pip install requests
"""

import csv
import json
import os
import time
import requests

RESOURCE_ID = "dd13c2e7-4a83-40aa-a040-08a27385759b"
CKAN_BASE = "https://data.wprdc.org/api/3/action"
GEOJSON_URL = "https://data.wprdc.org/dataset/9ebd073b-f637-4f33-a7c2-619d23dd085a/resource/8a38a51d-5000-4600-8114-3f9e92202a64/download/pgh_centerlines.geojson"

OUTPUT_CSV = "centerlines_pgh.csv"
GEOJSON_CACHE = "centerlines_pgh.geojson"

FIELDNAMES = [
    "objectid", "streetname", "domi_class", "paveclass", "speedlimit",
    "num_lanes", "roadwidth", "oneway", "owner", "shape_length",
    "mid_lat", "mid_lng", "data_source",
]

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, params=None, stream=False):
    """Fetch URL with exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, stream=stream, timeout=120)
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
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries")


def download_geojson():
    """Download the GeoJSON file (streamed to disk) if not cached."""
    if os.path.exists(GEOJSON_CACHE):
        print(f"Using cached {GEOJSON_CACHE}")
        return

    print("Downloading GeoJSON (this may take a minute, ~98 MB)...")
    resp = fetch_with_retry(GEOJSON_URL, stream=True)
    with open(GEOJSON_CACHE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Saved to {GEOJSON_CACHE}")


def extract_midpoints_from_geojson():
    """Parse GeoJSON and compute midpoint lat/lng for each feature."""
    print("Parsing GeoJSON for midpoints...")
    midpoints = {}

    with open(GEOJSON_CACHE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        oid = props.get("OBJECTID") or props.get("objectid") or props.get("FID")
        if oid is None:
            continue

        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        geom_type = geom.get("type", "")

        # Flatten coordinates depending on geometry type
        flat = []
        if geom_type == "LineString":
            flat = coords
        elif geom_type == "MultiLineString":
            for line in coords:
                flat.extend(line)
        elif geom_type == "Point":
            flat = [coords]

        if flat:
            lngs = [c[0] for c in flat]
            lats = [c[1] for c in flat]
            midpoints[int(oid)] = {
                "mid_lat": round(sum(lats) / len(lats), 6),
                "mid_lng": round(sum(lngs) / len(lngs), 6),
            }

    print(f"  Extracted midpoints for {len(midpoints)} segments")
    return midpoints


def fetch_records():
    """Fetch all records from CKAN datastore using SQL with pagination."""
    all_records = []
    limit = 5000
    offset = 0

    print("Fetching street centerline records from WPRDC...")
    while True:
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"LIMIT {limit} OFFSET {offset}"
        )
        resp = fetch_with_retry(
            f"{CKAN_BASE}/datastore_search_sql",
            params={"sql": sql},
        )
        result = resp.json().get("result", {})
        records = result.get("records", [])
        if not records:
            break
        all_records.extend(records)
        print(f"  Fetched {len(all_records)} records...")
        offset += limit

    print(f"  Total: {len(all_records)} records")
    return all_records


def normalize_key(key):
    """Normalize CKAN field names (may come as uppercase)."""
    return key.strip().lower()


def build_rows(records, midpoints):
    """Convert CKAN records to output rows with midpoints."""
    rows = []
    for rec in records:
        # Normalize keys
        r = {normalize_key(k): v for k, v in rec.items()}

        oid = r.get("objectid") or r.get("fid")
        if oid is None:
            continue
        oid = int(oid)

        mid = midpoints.get(oid, {})

        row = {
            "objectid": oid,
            "streetname": (r.get("st_name") or r.get("streetname") or "").strip(),
            "domi_class": r.get("domi_class", ""),
            "paveclass": r.get("paveclass", ""),
            "speedlimit": r.get("speedlimit", ""),
            "num_lanes": r.get("num_lanes", ""),
            "roadwidth": r.get("roadwidth", ""),
            "oneway": r.get("oneway", ""),
            "owner": r.get("owner", ""),
            "shape_length": r.get("shape_leng") or r.get("shape_length", ""),
            "mid_lat": mid.get("mid_lat", ""),
            "mid_lng": mid.get("mid_lng", ""),
            "data_source": "wprdc_centerlines",
        }
        rows.append(row)

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    # 1. Download GeoJSON for midpoints
    download_geojson()
    midpoints = extract_midpoints_from_geojson()

    # 2. Fetch tabular data from CKAN
    records = fetch_records()

    # 3. Build rows
    rows = build_rows(records, midpoints)

    # 4. Save to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} road segments to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
