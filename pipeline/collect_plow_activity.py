"""
Collect snow plow GPS trace data from WPRDC.

Downloads representative storm event GeoJSON files from the 2018-2020 plow
tracker dataset.

Source: https://data.wprdc.org/dataset/plow-activity
Output: plow_activity_pgh.csv

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

CKAN_BASE = "https://data.wprdc.org/api/3/action"
DATASET_ID = "d0b56030-3391-49db-87bd-4f1c16490fbc"

OUTPUT_CSV = os.path.join(DATA_DIR, "plow_activity_pgh.csv")

FIELDNAMES = [
    "event_date", "truck_id", "latitude", "longitude", "timestamp", "data_source",
]

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, params=None, stream=False):
    """Fetch with exponential backoff."""
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
    raise RuntimeError(f"Failed to fetch after {MAX_RETRIES} retries")


def get_resource_list():
    """Get list of resources (GeoJSON files) in the plow activity dataset."""
    print("Fetching plow activity dataset metadata...")
    resp = fetch_with_retry(
        f"{CKAN_BASE}/package_show",
        params={"id": DATASET_ID},
    )
    result = resp.json().get("result", {})
    resources = result.get("resources", [])

    # Filter for GeoJSON/JSON resources
    geojson_resources = [
        r for r in resources
        if r.get("format", "").upper() in ("GEOJSON", "JSON", "")
        and r.get("url", "").endswith((".geojson", ".json"))
    ]

    # If no extension filter works, try all resources
    if not geojson_resources:
        geojson_resources = resources[:5]  # Take first 5

    print(f"  Found {len(resources)} resources, {len(geojson_resources)} downloadable")
    return geojson_resources[:5]  # Limit to 5 representative events


def download_and_parse(resource):
    """Download a plow activity GeoJSON and extract GPS points."""
    url = resource.get("url", "")
    name = resource.get("name", "unknown")
    print(f"  Downloading {name}...")

    rows = []
    try:
        resp = fetch_with_retry(url, stream=True)
        content = resp.content.decode("utf-8", errors="replace")
        data = json.loads(content)

        features = data.get("features", [])
        # If it's a flat list instead of GeoJSON
        if not features and isinstance(data, list):
            features = data

        for f in features:
            if isinstance(f, dict) and "geometry" in f:
                geom = f.get("geometry", {})
                props = f.get("properties", {})
                coords = geom.get("coordinates", [])

                if geom.get("type") == "Point" and len(coords) >= 2:
                    rows.append({
                        "event_date": name,
                        "truck_id": props.get("truck_id") or props.get("TRUCK_ID") or props.get("id", ""),
                        "latitude": coords[1],
                        "longitude": coords[0],
                        "timestamp": props.get("timestamp") or props.get("TIMESTAMP") or props.get("time", ""),
                        "data_source": "wprdc_plow",
                    })
                elif geom.get("type") == "LineString":
                    # Take midpoint
                    mid_idx = len(coords) // 2
                    if coords:
                        rows.append({
                            "event_date": name,
                            "truck_id": props.get("truck_id") or props.get("TRUCK_ID") or props.get("id", ""),
                            "latitude": coords[mid_idx][1] if len(coords[mid_idx]) >= 2 else "",
                            "longitude": coords[mid_idx][0] if len(coords[mid_idx]) >= 2 else "",
                            "timestamp": props.get("timestamp") or props.get("TIMESTAMP") or "",
                            "data_source": "wprdc_plow",
                        })

        print(f"    Extracted {len(rows)} GPS points")

    except Exception as e:
        print(f"    Error processing {name}: {e}")

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    resources = get_resource_list()
    if not resources:
        print("No plow activity resources found. Writing empty CSV.")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        return

    all_rows = []
    for resource in resources:
        rows = download_and_parse(resource)
        all_rows.extend(rows)
        time.sleep(1)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} plow GPS records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
