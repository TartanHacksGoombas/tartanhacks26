"""
Collect snow/ice-related 311 complaints from WPRDC.

Filters 311 data for snow-related request types. Historical only (data stopped
updating Feb 2025).

Source: https://data.wprdc.org/dataset/311-data
Resource: 76fda9d0-69be-4dd5-8108-0de7a583571

Output: 311_snow_pgh.csv

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

RESOURCE_ID = "29462525-62a6-45bf-9b5e-ad2e1c06348d"
CKAN_BASE = "https://data.wprdc.org/api/3/action"

OUTPUT_CSV = os.path.join(DATA_DIR, "311_snow_pgh.csv")

FIELDNAMES = [
    "group_id", "request_type", "created_date", "street",
    "neighborhood", "latitude", "longitude", "data_source",
]

# Snow-related 311 request types
SNOW_TYPES = [
    "Snow/Ice removal",
    "Snow - Loss of Access",
    "Snow - Blocked Sidewalk",
    "Snow - Cleared into Street",
    "Snow - Emergency Route",
    "Snow - Plow Damage",
    "Snow - Snow Angel",
    "Snow - Unshoveled Sidewalk",
    "Snow/Ice Removal",
    "Snow removal",
    "Icy Conditions",
]

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, params=None):
    """Fetch with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=120)
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


def fetch_311_records():
    """Fetch snow-related 311 records from WPRDC."""
    all_records = []
    limit = 5000
    offset = 0

    print("Fetching snow-related 311 data from WPRDC...")
    while True:
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"WHERE \"request_type_name\" LIKE '%Snow%' OR \"request_type_name\" LIKE '%Ice%' OR \"request_type_name\" LIKE '%Icy%' "
            f"LIMIT {limit} OFFSET {offset}"
        )
        try:
            resp = fetch_with_retry(
                f"{CKAN_BASE}/datastore_search_sql",
                params={"sql": sql},
            )
            result = resp.json().get("result", {})
            records = result.get("records", [])
        except Exception as e:
            print(f"  Error on offset {offset}: {e}")
            break

        if not records:
            break
        all_records.extend(records)
        print(f"  Fetched {len(all_records)} records...")
        offset += limit

    print(f"  Total snow 311 records: {len(all_records)}")
    return all_records


def build_rows(records):
    """Convert CKAN records to output rows."""
    rows = []
    for rec in records:
        r = {k.strip().lower(): v for k, v in rec.items()}

        rows.append({
            "group_id": r.get("group_id") or r.get("_id") or "",
            "request_type": (r.get("request_type_name") or r.get("request_type") or "").strip(),
            "created_date": r.get("create_date_et") or r.get("created_on") or "",
            "street": (r.get("street") or r.get("address") or "").strip(),
            "neighborhood": (r.get("neighborhood") or "").strip(),
            "latitude": r.get("latitude") or "",
            "longitude": r.get("longitude") or "",
            "data_source": "wprdc_311",
        })

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    records = fetch_311_records()
    rows = build_rows(records)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    types = {}
    for r in rows:
        t = r["request_type"]
        types[t] = types.get(t, 0) + 1

    print(f"\nSaved {len(rows)} snow 311 records to {OUTPUT_CSV}")
    print("  By type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:35s} : {count}")


if __name__ == "__main__":
    main()
