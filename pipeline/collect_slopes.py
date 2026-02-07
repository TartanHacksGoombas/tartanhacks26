"""
Collect 25%+ slope areas from WPRDC and flag road segments near steep slopes.

Downloads slope GeoJSON, extracts polygon centroids, and flags road segments
within 50m of steep slope areas.

Source: https://data.wprdc.org/dataset/25-slope
Output: slopes_pgh.csv

Requires: pip install requests
"""

import csv
import json
import math
import os
import time
import requests

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SLOPE_RESOURCE_ID = "5ce91a56-0799-46ea-9585-13fa8db5979e"
GEOJSON_URL = "https://data.wprdc.org/dataset/25-or-greater-slope/resource/5ce91a56-0799-46ea-9585-13fa8db5979e/download"

INPUT_CSV = os.path.join(DATA_DIR, "centerlines_pgh.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "slopes_pgh.csv")
GEOJSON_CACHE = os.path.join(DATA_DIR, "slopes_pgh.geojson")

FIELDNAMES = ["objectid", "mid_lat", "mid_lng", "near_steep_slope", "slope_area_sqm", "data_source"]

PROXIMITY_M = 50  # flag roads within 50m of steep slope area

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, stream=False):
    """Fetch with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, stream=stream, timeout=120)
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


def haversine_m(lat1, lng1, lat2, lng2):
    """Haversine distance in meters between two lat/lng points."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def download_geojson():
    """Download slope GeoJSON if not cached."""
    if os.path.exists(GEOJSON_CACHE):
        print(f"Using cached {GEOJSON_CACHE}")
        return

    print("Downloading slope GeoJSON (~66 MB)...")
    resp = fetch_with_retry(GEOJSON_URL, stream=True)
    with open(GEOJSON_CACHE, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Saved to {GEOJSON_CACHE}")


def extract_slope_centroids():
    """Parse GeoJSON and compute centroid + area for each slope polygon."""
    print("Parsing slope GeoJSON...")
    slopes = []

    with open(GEOJSON_CACHE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        geom_type = geom.get("type", "")
        props = feature.get("properties", {})

        # Flatten to list of rings
        rings = []
        if geom_type == "Polygon":
            rings = coords
        elif geom_type == "MultiPolygon":
            for poly in coords:
                rings.extend(poly)

        if not rings:
            continue

        # Use first ring (exterior) for centroid
        exterior = rings[0]
        lngs = [c[0] for c in exterior]
        lats = [c[1] for c in exterior]
        centroid_lat = sum(lats) / len(lats)
        centroid_lng = sum(lngs) / len(lngs)

        area = props.get("shape_area") or props.get("SHAPE_AREA") or props.get("Shape__Area") or ""

        slopes.append({
            "lat": centroid_lat,
            "lng": centroid_lng,
            "area_sqm": area,
        })

    print(f"  Found {len(slopes)} slope areas")
    return slopes


def load_midpoints():
    """Load road segment midpoints from centerlines CSV."""
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


def flag_near_slopes(road_points, slope_centroids):
    """Flag each road segment if it's within PROXIMITY_M of any steep slope."""
    print(f"Flagging roads within {PROXIMITY_M}m of steep slopes...")
    results = []
    total = len(road_points)

    for i, rp in enumerate(road_points):
        if (i + 1) % 2000 == 0:
            print(f"  Processing {i+1}/{total}...")

        near = False
        nearest_area = ""
        for sp in slope_centroids:
            dist = haversine_m(rp["mid_lat"], rp["mid_lng"], sp["lat"], sp["lng"])
            if dist <= PROXIMITY_M:
                near = True
                nearest_area = sp["area_sqm"]
                break

        results.append({
            "objectid": rp["objectid"],
            "mid_lat": rp["mid_lat"],
            "mid_lng": rp["mid_lng"],
            "near_steep_slope": near,
            "slope_area_sqm": nearest_area if near else "",
            "data_source": "wprdc_slopes",
        })

    flagged = sum(1 for r in results if r["near_steep_slope"])
    print(f"  {flagged} road segments near steep slopes")
    return results


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run collect_centerlines.py first.")
        return

    download_geojson()
    slope_centroids = extract_slope_centroids()
    road_points = load_midpoints()

    results = flag_near_slopes(road_points, slope_centroids)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
