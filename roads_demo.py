"""
Demo: Pull road data around Carnegie Mellon University using Google Maps APIs.

Uses:
  - Geocoding API to resolve CMU's location
  - Roads API (Nearest Roads) to identify roads from sampled points
  - Directions API to get route/road info between nearby intersections

Requires: pip install googlemaps
Set your API key via environment variable GOOGLE_MAPS_API_KEY or edit the script.
"""

import os
import csv
import json
import googlemaps
from itertools import product

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "YOUR_API_KEY_HERE")

def get_client():
    if API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            "Set your Google Maps API key:\n"
            "  export GOOGLE_MAPS_API_KEY='your-key'\n"
            "Or edit API_KEY in this script."
        )
    return googlemaps.Client(key=API_KEY)


def geocode_cmu(client):
    """Geocode Carnegie Mellon University to get its lat/lng."""
    results = client.geocode("Carnegie Mellon University, Pittsburgh, PA")
    if not results:
        raise RuntimeError("Could not geocode CMU")
    location = results[0]["geometry"]["location"]
    print(f"CMU Location: {location['lat']}, {location['lng']}")
    print(f"  Address: {results[0]['formatted_address']}")
    return location["lat"], location["lng"]


def generate_grid_points(center_lat, center_lng, grid_size=5, spacing_m=200):
    """Generate a grid of lat/lng points around a center for road sampling."""
    # ~111,111 meters per degree latitude, longitude varies by cos(lat)
    import math
    lat_per_m = 1.0 / 111111.0
    lng_per_m = 1.0 / (111111.0 * math.cos(math.radians(center_lat)))

    half = grid_size // 2
    points = []
    for di, dj in product(range(-half, half + 1), range(-half, half + 1)):
        lat = center_lat + di * spacing_m * lat_per_m
        lng = center_lng + dj * spacing_m * lng_per_m
        points.append((lat, lng))
    return points


def snap_to_roads(client, points):
    """Use the Roads API to snap sample points to the nearest roads."""
    print(f"\nSnapping {len(points)} grid points to nearest roads...")

    # The nearest_roads endpoint accepts up to 100 points per request
    snapped = client.nearest_roads(points)

    # Deduplicate by placeId to get unique road segments
    seen = set()
    unique_roads = []
    for pt in snapped:
        pid = pt.get("placeId", "")
        if pid and pid not in seen:
            seen.add(pid)
            unique_roads.append(pt)

    print(f"  Found {len(unique_roads)} unique road segments (by placeId)")
    return unique_roads


def reverse_geocode_roads(client, road_points):
    """Reverse-geocode snapped road points to get street names."""
    print("\nReverse-geocoding road points to get street names...")

    road_names = {}
    for rp in road_points:
        loc = rp["location"]
        lat, lng = loc["latitude"], loc["longitude"]
        place_id = rp.get("placeId", "unknown")

        results = client.reverse_geocode((lat, lng), result_type=["route"])
        if results:
            name = results[0].get("formatted_address", "Unknown")
            # Extract just the street name from address components
            for comp in results[0].get("address_components", []):
                if "route" in comp["types"]:
                    name = comp["long_name"]
                    break
            road_names[place_id] = {
                "name": name,
                "lat": lat,
                "lng": lng,
            }

    return road_names


def get_nearby_roads_via_places(client, lat, lng, radius=800):
    """Use the Places API nearby search to find road-related features."""
    print(f"\nSearching for road-related places within {radius}m of CMU...")

    # Search for transit stations and intersections as road proxies
    results = client.places_nearby(
        location=(lat, lng),
        radius=radius,
        type="transit_station",
    )
    places = results.get("results", [])

    # Also search for general routes/roads via text search
    text_results = client.places(
        query="road near Carnegie Mellon University Pittsburgh",
        location=(lat, lng),
        radius=radius,
    )
    places.extend(text_results.get("results", []))

    print(f"  Found {len(places)} road-related places")
    for p in places[:10]:
        loc = p["geometry"]["location"]
        print(f"    - {p['name']} ({loc['lat']:.5f}, {loc['lng']:.5f})")

    return places


def get_road_directions(client, origin, destination):
    """Pull road/route data between two points using the Directions API."""
    print(f"\nGetting directions (road data) between two points...")

    result = client.directions(
        origin=origin,
        destination=destination,
        mode="driving",
        alternatives=True,
    )

    roads_seen = set()
    for route_idx, route in enumerate(result):
        print(f"\n  Route {route_idx + 1}: {route['summary']}")
        for leg in route["legs"]:
            print(f"    Distance: {leg['distance']['text']}, Duration: {leg['duration']['text']}")
            for step in leg["steps"]:
                instruction = step["html_instructions"]
                # Extract road names from the instruction HTML
                import re
                road_match = re.findall(r"<b>(.*?)</b>", instruction)
                for road in road_match:
                    if road not in ("north", "south", "east", "west", "left", "right"):
                        roads_seen.add(road)

    print(f"\n  Roads along route(s): {', '.join(sorted(roads_seen))}")
    return result, roads_seen


def save_to_csv(road_names, direction_roads, places, output_path="roads_around_cmu.csv"):
    """Save all collected road data to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "latitude", "longitude", "source", "place_id"])

        # Roads from snap + reverse geocode
        for pid, info in road_names.items():
            writer.writerow([info["name"], info["lat"], info["lng"], "roads_api", pid])

        # Roads from directions
        for road in sorted(direction_roads):
            writer.writerow([road, "", "", "directions_api", ""])

        # Places
        seen = set()
        for p in places:
            if p["name"] in seen:
                continue
            seen.add(p["name"])
            loc = p["geometry"]["location"]
            writer.writerow([
                p["name"], loc["lat"], loc["lng"],
                "places_api", p.get("place_id", ""),
            ])

    print(f"\nSaved to {output_path}")


def main():
    client = get_client()
    road_names = {}
    direction_roads = set()
    nearby_places = []

    # 1. Geocode CMU
    print("=" * 60)
    print("STEP 1: Geocoding Carnegie Mellon University")
    print("=" * 60)
    lat, lng = geocode_cmu(client)

    # 2. Snap grid points to nearest roads
    print("\n" + "=" * 60)
    print("STEP 2: Snapping grid points to roads (Roads API)")
    print("=" * 60)
    grid_points = generate_grid_points(lat, lng, grid_size=5, spacing_m=150)
    try:
        road_points = snap_to_roads(client, grid_points)
    except Exception as e:
        print(f"  Roads API error (may need Roads API enabled): {e}")
        road_points = []

    # 3. Reverse-geocode to get street names
    if road_points:
        print("\n" + "=" * 60)
        print("STEP 3: Reverse-geocoding road points for street names")
        print("=" * 60)
        road_names = reverse_geocode_roads(client, road_points[:15])  # limit to save quota
        print(f"\nUnique streets found near CMU:")
        seen_names = set()
        for pid, info in road_names.items():
            if info["name"] not in seen_names:
                seen_names.add(info["name"])
                print(f"  - {info['name']} ({info['lat']:.5f}, {info['lng']:.5f})")
    else:
        print("\n  Skipping step 3 (no road points).")

    # 4. Pull route data via Directions API
    print("\n" + "=" * 60)
    print("STEP 4: Road data via Directions API")
    print("=" * 60)
    origin = "Carnegie Mellon University, Pittsburgh, PA"
    destination = "Schenley Park, Pittsburgh, PA"
    try:
        directions, direction_roads = get_road_directions(client, origin, destination)
    except Exception as e:
        print(f"  Directions API error: {e}")

    # 5. Nearby places (transit stops, which sit on roads)
    print("\n" + "=" * 60)
    print("STEP 5: Nearby road-related places (Places API)")
    print("=" * 60)
    try:
        nearby_places = get_nearby_roads_via_places(client, lat, lng)
    except Exception as e:
        print(f"  Places API error: {e}")

    # 6. Save everything to CSV
    print("\n" + "=" * 60)
    print("STEP 6: Saving to CSV")
    print("=" * 60)
    save_to_csv(road_names, direction_roads, nearby_places)

    print("\n" + "=" * 60)
    print("DONE. See above for road data around CMU.")
    print("=" * 60)


if __name__ == "__main__":
    main()
