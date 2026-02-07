"""
Collect OpenStreetMap road data for all of Pittsburgh via the Overpass API.

Expanded version of roads_osm.py — uses a bounding box covering Pittsburgh
instead of an 800m radius around CMU. Also extracts the 'incline' tag for
slope data.

Output: roads_osm_pgh.csv (~15,000-30,000 rows), roads_osm_pgh.json

Requires: pip install requests
"""

import csv
import json
import os
import time
import requests

# Pittsburgh bounding box (south, west, north, east)
PGH_BBOX = (40.36, -80.10, 40.50, -79.86)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OUTPUT_CSV = "roads_osm_pgh.csv"
OUTPUT_JSON = "roads_osm_pgh.json"

FIELDNAMES = [
    "osm_id", "name", "highway_type", "surface", "lanes", "maxspeed",
    "oneway", "bridge", "tunnel", "lit", "sidewalk", "cycleway",
    "width", "access", "ref", "incline", "mid_lat", "mid_lng", "node_count",
    "data_source",
]

MAX_RETRIES = 3
RETRY_BACKOFF = 5


def fetch_with_retry(url, params=None):
    """Fetch URL with exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=180)
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


def query_roads(bbox):
    """Query the Overpass API for all roads/highways in the bounding box."""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:120][maxsize:256000000];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"Querying Overpass API for roads in Pittsburgh bbox {bbox}...")
    print("  (this may take 30-60 seconds)")
    resp = fetch_with_retry(OVERPASS_URL, params={"data": query})
    data = resp.json()

    nodes = {}
    ways = []
    for elem in data["elements"]:
        if elem["type"] == "node":
            nodes[elem["id"]] = (elem["lat"], elem["lon"])
        elif elem["type"] == "way":
            ways.append(elem)

    print(f"  Found {len(ways)} road segments and {len(nodes)} nodes")
    return ways, nodes


def parse_road_data(ways, nodes):
    """Extract useful attributes from each road way."""
    roads = []
    for way in ways:
        tags = way.get("tags", {})
        highway_type = tags.get("highway", "")

        if highway_type in ("proposed", "construction", "platform"):
            continue

        way_nodes = way.get("nodes", [])
        lats, lngs = [], []
        for nid in way_nodes:
            if nid in nodes:
                lats.append(nodes[nid][0])
                lngs.append(nodes[nid][1])

        mid_lat = round(sum(lats) / len(lats), 6) if lats else ""
        mid_lng = round(sum(lngs) / len(lngs), 6) if lngs else ""

        road = {
            "osm_id": way["id"],
            "name": tags.get("name", ""),
            "highway_type": highway_type,
            "surface": tags.get("surface", ""),
            "lanes": tags.get("lanes", ""),
            "maxspeed": tags.get("maxspeed", ""),
            "oneway": tags.get("oneway", ""),
            "bridge": tags.get("bridge", ""),
            "tunnel": tags.get("tunnel", ""),
            "lit": tags.get("lit", ""),
            "sidewalk": tags.get("sidewalk", ""),
            "cycleway": tags.get("cycleway", ""),
            "width": tags.get("width", ""),
            "access": tags.get("access", ""),
            "ref": tags.get("ref", ""),
            "incline": tags.get("incline", ""),
            "mid_lat": mid_lat,
            "mid_lng": mid_lng,
            "node_count": len(way_nodes),
            "data_source": "osm",
        }
        roads.append(road)

    return roads


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    # 1. Query roads from OSM
    ways, nodes = query_roads(PGH_BBOX)

    # 2. Parse into structured data
    roads = parse_road_data(ways, nodes)

    # 3. Print summary
    type_counts = {}
    for r in roads:
        t = r["highway_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'='*60}")
    print(f"ROAD SUMMARY: {len(roads)} road segments")
    print(f"{'='*60}")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:25s} : {count}")

    # 4. Save to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(roads)
    print(f"\nSaved {len(roads)} road segments to {OUTPUT_CSV}")

    # 5. Save raw JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(roads, f, indent=2)
    print(f"Saved raw JSON to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
