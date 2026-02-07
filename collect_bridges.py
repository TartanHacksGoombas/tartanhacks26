"""
Collect Pittsburgh bridge data from WPRDC.

Source: https://data.wprdc.org/dataset/pittsburgh-bridges
Only ~153 records — single API call.

Output: bridges_pgh.csv

Requires: pip install requests
"""

import csv
import os
import time
import requests

RESOURCE_ID = "23f675e5-f745-49c1-8fa0-c5bc35c25184"
CKAN_BASE = "https://data.wprdc.org/api/3/action"

OUTPUT_CSV = "bridges_pgh.csv"

FIELDNAMES = [
    "bridge_id", "name", "neighborhood", "latitude", "longitude",
    "year_built", "data_source",
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


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    print("Fetching Pittsburgh bridges from WPRDC...")
    sql = f'SELECT * FROM "{RESOURCE_ID}"'
    resp = fetch_with_retry(
        f"{CKAN_BASE}/datastore_search_sql",
        params={"sql": sql},
    )
    records = resp.json().get("result", {}).get("records", [])
    print(f"  Found {len(records)} bridges")

    rows = []
    for rec in records:
        # Normalize keys to lowercase
        r = {k.strip().lower(): v for k, v in rec.items()}

        row = {
            "bridge_id": r.get("id") or r.get("_id") or r.get("objectid", ""),
            "name": (r.get("bridge_name") or r.get("name") or r.get("structure", "")).strip(),
            "neighborhood": (r.get("neighborhood") or r.get("hood") or "").strip(),
            "latitude": r.get("latitude") or r.get("lat") or "",
            "longitude": r.get("longitude") or r.get("lng") or r.get("lon") or "",
            "year_built": r.get("year_built") or r.get("yearbuilt") or "",
            "data_source": "wprdc_bridges",
        }
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} bridges to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
