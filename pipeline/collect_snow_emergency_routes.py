"""
Derive Pittsburgh snow priority routes from centerlines road classification.

Pittsburgh does not publish an official "snow emergency routes" GIS dataset.
Instead, this script derives snow priority routes from the WPRDC Street
Centerlines data using the DOMI road classification (domi_class):
  - "A" = Arterial roads (highest priority for snow plowing)
  - "M" = Major collector roads

These are the roads that Pittsburgh prioritizes for plowing during snow events.
The output flags road segments that are snow priority routes.

Source: data/centerlines_pgh.csv (must run collect_centerlines.py first)
Output: snow_emergency_routes_pgh.csv

Requires: pip install requests (for collect_centerlines.py dependency only)
"""

import csv
import os

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CENTERLINES_CSV = os.path.join(DATA_DIR, "centerlines_pgh.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "snow_emergency_routes_pgh.csv")

FIELDNAMES = [
    "objectid", "streetname", "domi_class", "speedlimit",
    "mid_lat", "mid_lng",
    "data_source",
]

# DOMI road classifications that are snow priority routes
# Pittsburgh plows arterials and collectors first during snow events
SNOW_PRIORITY_CLASSES = {"Principal Arterial", "Minor Arterial", "Collector"}


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    if not os.path.exists(CENTERLINES_CSV):
        print(f"Error: {CENTERLINES_CSV} not found. Run collect_centerlines.py first.")
        return

    print("Deriving snow priority routes from centerlines road classification...")

    rows = []
    total = 0
    with open(CENTERLINES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            total += 1
            domi_class = (rec.get("domi_class") or "").strip()
            if domi_class in SNOW_PRIORITY_CLASSES:
                rows.append({
                    "objectid": rec.get("objectid", ""),
                    "streetname": (rec.get("streetname") or "").strip(),
                    "domi_class": domi_class,
                    "speedlimit": rec.get("speedlimit", ""),
                    "mid_lat": rec.get("mid_lat", ""),
                    "mid_lng": rec.get("mid_lng", ""),
                    "data_source": "pgh_snow_emergency_routes",
                })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} snow priority route segments to {OUTPUT_CSV}")
    print(f"  (out of {total} total centerline segments)")

    # Summary by class
    by_class = {}
    for r in rows:
        c = r["domi_class"]
        by_class[c] = by_class.get(c, 0) + 1

    print("  By DOMI class:")
    for c, count in sorted(by_class.items()):
        label = c
        print(f"    {c} ({label:20s}) : {count}")

    streets = set(r["streetname"] for r in rows if r["streetname"])
    print(f"  Unique street names: {len(streets)}")


if __name__ == "__main__":
    main()
