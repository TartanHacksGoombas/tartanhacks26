"""
Collect 511PA traffic events for Allegheny County.

Real-time snapshot of traffic events from PennDOT's 511PA system. Each run
appends a timestamped snapshot so historical data accumulates over time.

Uses the 511PA DataTables AJAX API (POST /List/GetData/traffic) which returns
JSON with event records. Requires establishing a session first.

Source: https://www.511pa.com
Output: 511pa_events_pgh.csv (real-time snapshot, appended each run)

Requires: pip install requests
"""

import csv
import os
import time
from datetime import datetime, timezone
import requests

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 511PA uses a DataTables server-side API. A session cookie is needed.
SESSION_URL = "https://www.511pa.com/list/events/traffic"
DATA_URL = "https://www.511pa.com/List/GetData/traffic"

OUTPUT_CSV = os.path.join(DATA_DIR, "511pa_events_pgh.csv")

FIELDNAMES = [
    "event_id", "event_type", "roadway", "direction", "county",
    "description", "location_description", "is_full_closure",
    "start_time", "end_time", "last_updated",
    "fetched_at", "data_source",
]

PAGE_SIZE = 500
MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_events():
    """Fetch traffic events from 511PA DataTables API."""
    print("Fetching 511PA traffic events...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; PittsburghSnowProject/1.0)",
    })

    # Establish session cookie
    print("  Establishing session...")
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(SESSION_URL, timeout=30)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed to establish session: {e}")
            wait = RETRY_BACKOFF ** (attempt + 1)
            print(f"  Retry in {wait}s: {e}")
            time.sleep(wait)

    # Fetch data pages
    all_records = []
    offset = 0
    total = None

    while True:
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.post(
                    DATA_URL,
                    data={"draw": 1, "start": offset, "length": PAGE_SIZE},
                    headers={
                        "Accept": "application/json",
                        "X-Requested-With": "XMLHttpRequest",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  Error fetching page at offset {offset}: {e}")
                    return all_records
                wait = RETRY_BACKOFF ** (attempt + 1)
                print(f"  Retry in {wait}s: {e}")
                time.sleep(wait)

        records = data.get("data", [])
        if total is None:
            total = data.get("recordsTotal", 0)
            print(f"  Total events available: {total}")

        if not records:
            break

        all_records.extend(records)
        print(f"  Fetched {len(all_records)}/{total} events...")

        if len(all_records) >= total:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)

    print(f"  Total events fetched: {len(all_records)}")
    return all_records


def filter_allegheny(records):
    """Filter events to Allegheny County."""
    filtered = [r for r in records if (r.get("county") or "").lower() == "allegheny"]
    print(f"  Allegheny County events: {len(filtered)}")
    return filtered


def build_rows(records):
    """Convert API records to output rows."""
    rows = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for rec in records:
        rows.append({
            "event_id": rec.get("id") or "",
            "event_type": (rec.get("type") or "").strip(),
            "roadway": (rec.get("roadwayName") or "").strip(),
            "direction": (rec.get("direction") or "").strip(),
            "county": (rec.get("county") or "").strip(),
            "description": (rec.get("description") or "").strip(),
            "location_description": (rec.get("locationDescription") or "").strip(),
            "is_full_closure": rec.get("isFullClosure") or False,
            "start_time": rec.get("startDate") or "",
            "end_time": rec.get("endDate") or "",
            "last_updated": rec.get("lastUpdated") or "",
            "fetched_at": fetched_at,
            "data_source": "511pa",
        })

    return rows


def main():
    # Append mode — each run adds a timestamped snapshot
    try:
        records = fetch_events()
    except Exception as e:
        print(f"  Failed to fetch 511PA events: {e}")
        print("  511PA may not be available. Skipping.")
        return

    if not records:
        print("  No events returned. This may be normal if no active incidents.")
        return

    allegheny = filter_allegheny(records)
    rows = build_rows(allegheny)

    if not rows:
        print("  No Allegheny County events to save.")
        return

    # Append if file exists, create with header if not
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Appended' if file_exists else 'Saved'} {len(rows)} 511PA events to {OUTPUT_CSV}")

    # Summary by type
    types = {}
    for r in rows:
        t = r["event_type"] or "(unknown)"
        types[t] = types.get(t, 0) + 1

    closures = sum(1 for r in rows if r["is_full_closure"])
    print(f"  Full closures: {closures}")
    print("  By type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:35s} : {count}")


if __name__ == "__main__":
    main()
