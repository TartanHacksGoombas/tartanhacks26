"""
Demo: Pull detailed road data around Carnegie Mellon University using OpenStreetMap.

Uses the Overpass API (free, no API key needed) to query OSM for roads within
~800m of CMU. Returns rich attributes: name, lanes, surface, speed limit,
road type, one-way status, and more.

Requires: pip install requests
"""

import csv
import json
import os
import requests

# Resolve paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CMU coordinates
CMU_LAT = 40.4433
CMU_LNG = -79.9436
RADIUS_M = 800  # search radius in meters

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def query_roads(lat, lng, radius):
    """Query the Overpass API for all roads/highways near a point."""
    # Overpass QL query: find all 'highway' ways within radius of CMU
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"](around:{radius},{lat},{lng});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"Querying Overpass API for roads within {radius}m of ({lat}, {lng})...")
    resp = requests.get(OVERPASS_URL, params={"data": query})
    resp.raise_for_status()
    data = resp.json()

    # Separate ways (roads) from nodes (coordinates)
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

        # Skip non-road types (buildings, paths we don't care about)
        if highway_type in ("proposed", "construction", "platform"):
            continue

        # Get the midpoint of the way for a representative coordinate
        way_nodes = way.get("nodes", [])
        lats, lngs = [], []
        for nid in way_nodes:
            if nid in nodes:
                lats.append(nodes[nid][0])
                lngs.append(nodes[nid][1])

        mid_lat = sum(lats) / len(lats) if lats else ""
        mid_lng = sum(lngs) / len(lngs) if lngs else ""

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
            "ref": tags.get("ref", ""),  # route number (e.g. US 30)
            "mid_lat": mid_lat,
            "mid_lng": mid_lng,
            "node_count": len(way_nodes),
        }
        roads.append(road)

    return roads


def print_summary(roads):
    """Print a summary of the road data found."""
    # Count by type
    type_counts = {}
    for r in roads:
        t = r["highway_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"ROAD SUMMARY: {len(roads)} road segments found")
    print(f"{'=' * 60}")

    print(f"\nBy road type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} : {count}")

    # Named roads
    named = set(r["name"] for r in roads if r["name"])
    print(f"\nNamed roads ({len(named)}):")
    for name in sorted(named):
        # Find attributes for this road
        segments = [r for r in roads if r["name"] == name]
        types = set(r["highway_type"] for r in segments)
        surfaces = set(r["surface"] for r in segments if r["surface"])
        lanes = set(r["lanes"] for r in segments if r["lanes"])
        speeds = set(r["maxspeed"] for r in segments if r["maxspeed"])

        details = []
        details.append(f"type={'/'.join(types)}")
        if surfaces:
            details.append(f"surface={'/'.join(surfaces)}")
        if lanes:
            details.append(f"lanes={'/'.join(lanes)}")
        if speeds:
            details.append(f"speed={'/'.join(speeds)}")

        print(f"  {name:35s} [{', '.join(details)}]")

    # Surface breakdown
    surfaces = {}
    for r in roads:
        s = r["surface"] or "unknown"
        surfaces[s] = surfaces.get(s, 0) + 1
    print(f"\nSurface types:")
    for s, count in sorted(surfaces.items(), key=lambda x: -x[1]):
        print(f"  {s:20s} : {count}")

    # Roads with speed limits
    with_speed = [r for r in roads if r["maxspeed"]]
    if with_speed:
        print(f"\nRoads with speed limits:")
        seen = set()
        for r in with_speed:
            key = (r["name"] or f"OSM {r['osm_id']}", r["maxspeed"])
            if key not in seen:
                seen.add(key)
                print(f"  {key[0]:35s} : {r['maxspeed']}")


def save_to_csv(roads, output_path="roads_osm_cmu.csv"):
    """Save road data to CSV."""
    if not roads:
        print("No roads to save.")
        return

    fieldnames = [
        "osm_id", "name", "highway_type", "surface", "lanes", "maxspeed",
        "oneway", "bridge", "tunnel", "lit", "sidewalk", "cycleway",
        "width", "access", "ref", "mid_lat", "mid_lng", "node_count",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(roads)

    print(f"\nSaved {len(roads)} road segments to {output_path}")


def main():
    # 1. Query roads from OSM
    ways, nodes = query_roads(CMU_LAT, CMU_LNG, RADIUS_M)

    # 2. Parse into structured data
    roads = parse_road_data(ways, nodes)

    # 3. Print summary
    print_summary(roads)

    # 4. Save to CSV
    print(f"\n{'=' * 60}")
    print("Saving to CSV...")
    print(f"{'=' * 60}")
    save_to_csv(roads, os.path.join(_SCRIPT_DIR, "roads_osm_cmu.csv"))

    # 5. Also save raw JSON for reference
    with open(os.path.join(_SCRIPT_DIR, "roads_osm_cmu.json"), "w", encoding="utf-8") as f:
        json.dump(roads, f, indent=2)
    print(f"Saved raw JSON to roads_osm_cmu.json")

    print(f"\n{'=' * 60}")
    print("DONE.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
