"""
Collect winter-related crash data from WPRDC Allegheny County Crash Data.

Filters for crashes involving snow/slush or icy road conditions.

Source: https://data.wprdc.org/dataset/allegheny-county-crash-data
Resource: bf8b3c7e-8d60-40df-9134-21606a451c1a

Output: crashes_winter_pgh.csv (~19,897 rows)

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

RESOURCE_ID = "2c13021f-74a9-4289-a1e5-fe0472c89881"
CKAN_BASE = "https://data.wprdc.org/api/3/action"

OUTPUT_CSV = os.path.join(DATA_DIR, "crashes_winter_pgh.csv")

FIELDNAMES = [
    "crash_crn", "crash_year", "crash_month", "weather_code",
    "road_condition_code", "icy_road", "snow_slush_road",
    "latitude", "longitude", "street_name", "speed_limit",
    "lane_count", "fatal_count", "injury_count", "data_source",
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


def fetch_crash_records():
    """Fetch winter-related crash records from WPRDC."""
    all_records = []
    limit = 5000
    offset = 0

    print("Fetching winter crash data from WPRDC...")
    while True:
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"WHERE \"SNOW_SLUSH_ROAD\" = '1' OR \"ICY_ROAD\" = '1' "
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

    print(f"  Total winter crash records: {len(all_records)}")
    return all_records


def build_rows(records):
    """Convert CKAN records to output rows."""
    rows = []
    for rec in records:
        r = {k.strip().upper(): v for k, v in rec.items()}

        rows.append({
            "crash_crn": r.get("CRN") or r.get("CRASH_CRN", ""),
            "crash_year": r.get("CRASH_YEAR") or r.get("YEAR", ""),
            "crash_month": r.get("CRASH_MONTH") or r.get("MONTH", ""),
            "weather_code": r.get("WEATHER") or r.get("WEATHER_CODE", ""),
            "road_condition_code": r.get("ROAD_CONDITION") or r.get("RDWY_SURF_TYPE_CD", ""),
            "icy_road": r.get("ICY_ROAD", ""),
            "snow_slush_road": r.get("SNOW_SLUSH_ROAD", ""),
            "latitude": r.get("DEC_LAT") or r.get("LATITUDE", ""),
            "longitude": r.get("DEC_LONG") or r.get("LONGITUDE", ""),
            "street_name": (r.get("STREET_NAME") or r.get("RDWY_SEQ_DESCR") or "").strip(),
            "speed_limit": r.get("SPEED_LIMIT") or "",
            "lane_count": r.get("LANE_COUNT") or r.get("RDWY_ORIENT_TYPE_CD", ""),
            "fatal_count": r.get("FATAL_COUNT") or r.get("FATAL", "0"),
            "injury_count": r.get("INJURY_COUNT") or r.get("INJ_COUNT") or r.get("INJURY", "0"),
            "data_source": "wprdc_crashes",
        })

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    records = fetch_crash_records()
    rows = build_rows(records)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    years = set(r["crash_year"] for r in rows if r["crash_year"])
    print(f"\nSaved {len(rows)} winter crash records to {OUTPUT_CSV}")
    print(f"  Year range: {min(years) if years else '?'} - {max(years) if years else '?'}")

    fatal = sum(1 for r in rows if str(r["fatal_count"]) not in ("", "0"))
    print(f"  Fatal crashes: {fatal}")


if __name__ == "__main__":
    main()
