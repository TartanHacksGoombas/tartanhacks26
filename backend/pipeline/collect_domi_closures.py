"""
Collect DOMI (Dept of Mobility & Infrastructure) street closure permits from WPRDC.

All permitted street closures in Pittsburgh — includes full closures, lane closures,
sidewalk closures, etc. Useful as training labels and features for road closure prediction.

Source: https://data.wprdc.org/dataset/domi-street-closures
Resource: a9a1d93a-9d3b-4c18-bd80-82ed6f86404a

Output: domi_closures_pgh.csv (~63k records)

Requires: pip install requests
"""

import csv
import os
import re
import time
import requests

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

RESOURCE_ID = "a9a1d93a-9d3b-4c18-bd80-82ed6f86404a"
CKAN_BASE = "https://data.wprdc.org/api/3/action"

OUTPUT_CSV = os.path.join(DATA_DIR, "domi_closures_pgh.csv")

FIELDNAMES = [
    "closure_id", "permit_type", "work_type",
    "from_date", "to_date", "active",
    "primary_street", "from_street", "to_street",
    "full_closure", "travel_lane", "parking_lane", "sidewalk",
    "latitude", "longitude",
    "data_source",
]

MAX_RETRIES = 3
RETRY_BACKOFF = 2
PAGE_SIZE = 5000


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


def parse_geometry(geom_str):
    """Parse midpoint lat/lng from WPRDC geometry text field.

    Handles multiple formats:
    - JSON array: "[[40.423, -79.978], [40.424, -79.979]]"
    - WKT LINESTRING: "LINESTRING (40.423 -79.978, 40.424 -79.979)"
    - WKT POINT: "POINT(-79.978 40.423)"
    - Simple pair: "40.423, -79.978"

    Returns midpoint (lat, lng) or ("", "").
    """
    if not geom_str or not isinstance(geom_str, str):
        return "", ""

    geom_str = geom_str.strip()

    # Try JSON array format: [[lat, lng], [lat, lng], ...]
    if geom_str.startswith("["):
        try:
            import json
            coords = json.loads(geom_str)
            if coords and isinstance(coords, list):
                # Handle nested array [[lat,lng], [lat,lng]]
                if isinstance(coords[0], list):
                    mid_idx = len(coords) // 2
                    lat, lng = coords[mid_idx][0], coords[mid_idx][1]
                    if 39 < lat < 42 and -81 < lng < -79:
                        return round(lat, 6), round(lng, 6)
                # Handle flat array [lat, lng]
                elif len(coords) == 2:
                    lat, lng = coords[0], coords[1]
                    if 39 < lat < 42 and -81 < lng < -79:
                        return round(lat, 6), round(lng, 6)
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            pass

    # Try WKT LINESTRING format: LINESTRING(lat lng, lat lng, ...)
    match = re.match(r"LINESTRING\s*\((.+)\)", geom_str, re.IGNORECASE)
    if match:
        pairs = match.group(1).split(",")
        mid_pair = pairs[len(pairs) // 2].strip()
        parts = mid_pair.split()
        if len(parts) == 2:
            try:
                a, b = float(parts[0]), float(parts[1])
                if 39 < a < 42 and -81 < b < -79:
                    return round(a, 6), round(b, 6)
                if 39 < b < 42 and -81 < a < -79:
                    return round(b, 6), round(a, 6)
            except ValueError:
                pass

    # Try "lat, lng" format
    match = re.match(r"^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$", geom_str)
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        if 39 < a < 42 and -81 < b < -79:
            return round(a, 6), round(b, 6)
        if 39 < b < 42 and -81 < a < -79:
            return round(b, 6), round(a, 6)

    # Try WKT POINT format: POINT(lng lat)
    match = re.match(r"POINT\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\)", geom_str, re.IGNORECASE)
    if match:
        lng, lat = float(match.group(1)), float(match.group(2))
        if 39 < lat < 42 and -81 < lng < -79:
            return round(lat, 6), round(lng, 6)

    return "", ""


def fetch_records():
    """Fetch all DOMI closure records from WPRDC."""
    all_records = []
    offset = 0

    print("Fetching DOMI street closures from WPRDC...")
    while True:
        sql = (
            f'SELECT * FROM "{RESOURCE_ID}" '
            f"LIMIT {PAGE_SIZE} OFFSET {offset}"
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
        offset += PAGE_SIZE

    print(f"  Total DOMI closure records: {len(all_records)}")
    return all_records


def build_rows(records):
    """Convert CKAN records to output rows."""
    rows = []
    for rec in records:
        r = {k.strip().lower(): v for k, v in rec.items()}

        # Try to get lat/lng from geometry field or direct lat/lng columns
        lat = r.get("latitude") or ""
        lng = r.get("longitude") or ""

        if not lat or not lng:
            geom = r.get("geometry") or r.get("geom") or r.get("the_geom") or ""
            parsed_lat, parsed_lng = parse_geometry(str(geom))
            if parsed_lat:
                lat, lng = parsed_lat, parsed_lng

        rows.append({
            "closure_id": r.get("id") or r.get("closure_id") or r.get("_id") or "",
            "permit_type": (r.get("permit_type") or r.get("permittype") or "").strip(),
            "work_type": (r.get("work_type") or r.get("worktype") or "").strip(),
            "from_date": r.get("from_date") or r.get("fromdate") or r.get("start_date") or "",
            "to_date": r.get("to_date") or r.get("todate") or r.get("end_date") or "",
            "active": r.get("active") or r.get("is_active") or "",
            "primary_street": (r.get("primary_street") or r.get("primarystreet") or r.get("street") or "").strip(),
            "from_street": (r.get("from_street") or r.get("fromstreet") or "").strip(),
            "to_street": (r.get("to_street") or r.get("tostreet") or "").strip(),
            "full_closure": r.get("full_closure") or r.get("fullclosure") or "",
            "travel_lane": r.get("travel_lane") or r.get("travellane") or "",
            "parking_lane": r.get("parking_lane") or r.get("parkinglane") or "",
            "sidewalk": r.get("sidewalk") or "",
            "latitude": lat,
            "longitude": lng,
            "data_source": "wprdc_domi_closures",
        })

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    records = fetch_records()
    rows = build_rows(records)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    types = {}
    for r in rows:
        t = r["permit_type"] or "(unknown)"
        types[t] = types.get(t, 0) + 1

    full_count = sum(1 for r in rows if str(r["full_closure"]).lower() in ("true", "1", "yes"))

    print(f"\nSaved {len(rows)} DOMI closure records to {OUTPUT_CSV}")
    print(f"  Full closures: {full_count}")
    print("  By permit type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:35s} : {count}")


if __name__ == "__main__":
    main()
