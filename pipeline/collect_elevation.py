"""
Collect elevation data for Pittsburgh road segment midpoints using Open-Elevation API.

Reads midpoints from centerlines_pgh.csv, batch-queries elevation (100 points
per request), and outputs elevation per segment.

Output: elevation_pgh.csv (~19,000 rows)

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

INPUT_CSV = os.path.join(DATA_DIR, "centerlines_pgh.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "elevation_pgh.csv")

ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

FIELDNAMES = ["objectid", "mid_lat", "mid_lng", "elevation_m", "data_source"]

BATCH_SIZE = 100
SLEEP_BETWEEN = 1.0

MAX_RETRIES = 5
RETRY_BACKOFF = 2


def fetch_with_retry(url, json_data):
    """POST with exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=json_data, timeout=60)
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


def load_midpoints():
    """Load objectid and midpoint coords from centerlines CSV."""
    points = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = row.get("mid_lat", "").strip()
            lng = row.get("mid_lng", "").strip()
            oid = row.get("objectid", "").strip()
            if lat and lng and oid:
                try:
                    points.append({
                        "objectid": int(float(oid)),
                        "mid_lat": float(lat),
                        "mid_lng": float(lng),
                    })
                except ValueError:
                    continue
    return points


def query_elevation_batch(points_batch):
    """Query Open-Elevation API for a batch of points."""
    locations = [
        {"latitude": p["mid_lat"], "longitude": p["mid_lng"]}
        for p in points_batch
    ]
    resp = fetch_with_retry(ELEVATION_URL, {"locations": locations})
    results = resp.json().get("results", [])
    return results


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run collect_centerlines.py first.")
        return

    points = load_midpoints()
    print(f"Loaded {len(points)} midpoints from {INPUT_CSV}")

    total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
    rows = []

    for i in range(0, len(points), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = points[i:i + BATCH_SIZE]
        print(f"Fetching batch {batch_num}/{total_batches}...")

        try:
            results = query_elevation_batch(batch)
            for pt, elev in zip(batch, results):
                rows.append({
                    "objectid": pt["objectid"],
                    "mid_lat": pt["mid_lat"],
                    "mid_lng": pt["mid_lng"],
                    "elevation_m": elev.get("elevation", ""),
                    "data_source": "open_elevation",
                })
        except Exception as e:
            print(f"  Error on batch {batch_num}: {e}")
            # Still add rows with empty elevation
            for pt in batch:
                rows.append({
                    "objectid": pt["objectid"],
                    "mid_lat": pt["mid_lat"],
                    "mid_lng": pt["mid_lng"],
                    "elevation_m": "",
                    "data_source": "open_elevation",
                })

        if batch_num < total_batches:
            time.sleep(SLEEP_BETWEEN)

    # Save
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    non_empty = sum(1 for r in rows if r["elevation_m"] != "")
    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV} ({non_empty} with elevation)")


if __name__ == "__main__":
    main()
