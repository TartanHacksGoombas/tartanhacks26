import { ConditionFeatureCollection, SegmentKind } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:4000";

export async function fetchConditions(
  bbox: [number, number, number, number],
  kind: SegmentKind
): Promise<ConditionFeatureCollection> {
  const query = new URLSearchParams({
    bbox: bbox.join(","),
    kind
  });

  const response = await fetch(`${API_BASE}/v1/conditions?${query.toString()}`);

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Failed to load conditions: ${response.status} ${body}`);
  }

  return (await response.json()) as ConditionFeatureCollection;
}

/* ── Autocomplete (Google Places via backend proxy) ── */

export type GeocodeSuggestion = {
  displayName: string;
  placeId: string;
  /** lat/lng are only filled after calling resolvePlace */
  lat: number;
  lng: number;
};

/** Returns up to 5 type-ahead suggestions using Places Autocomplete. */
export async function geocode(query: string): Promise<GeocodeSuggestion[]> {
  const res = await fetch(
    `${API_BASE}/v1/maps/autocomplete?` + new URLSearchParams({ input: query })
  );
  if (!res.ok) return [];

  const data = await res.json();
  if (data.status !== "OK" || !data.predictions?.length) return [];

  return data.predictions.slice(0, 5).map((p: any) => ({
    displayName: p.description,
    placeId: p.place_id,
    lat: 0,
    lng: 0
  }));
}

/** Resolve a place_id to lat/lng coordinates. */
export async function resolvePlace(placeId: string): Promise<{ lat: number; lng: number } | null> {
  const res = await fetch(
    `${API_BASE}/v1/maps/place-details?` + new URLSearchParams({ place_id: placeId })
  );
  if (!res.ok) return null;

  const data = await res.json();
  if (data.status !== "OK" || !data.results?.length) return null;

  const loc = data.results[0].geometry.location;
  return { lat: loc.lat, lng: loc.lng };
}

/* ── Routing (Google Directions via backend proxy) ── */

export type RouteResult = {
  geometry: { type: "LineString"; coordinates: [number, number][] };
  durationSec: number;
  distanceM: number;
  steps: Array<{ instruction: string; distanceM: number; durationSec: number }>;
};

/** Decode Google's encoded polyline into [lng, lat] pairs for GeoJSON. */
function decodePolyline(encoded: string): [number, number][] {
  const points: [number, number][] = [];
  let index = 0;
  let lat = 0;
  let lng = 0;

  while (index < encoded.length) {
    let shift = 0;
    let result = 0;
    let byte: number;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);
    lat += result & 1 ? ~(result >> 1) : result >> 1;

    shift = 0;
    result = 0;
    do {
      byte = encoded.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);
    lng += result & 1 ? ~(result >> 1) : result >> 1;

    points.push([lng / 1e5, lat / 1e5]); // GeoJSON is [lng, lat]
  }
  return points;
}

export async function fetchRoute(
  from: [number, number],
  to: [number, number]
): Promise<RouteResult> {
  const res = await fetch(
    `${API_BASE}/v1/maps/directions?` +
      new URLSearchParams({
        origin: `${from[1]},${from[0]}`,
        destination: `${to[1]},${to[0]}`
      })
  );
  if (!res.ok) throw new Error(`Directions failed: ${res.status}`);

  const data = await res.json();
  if (data.status !== "OK" || !data.routes?.length) {
    throw new Error(data.error_message ?? `No route found (${data.status})`);
  }

  const route = data.routes[0];
  const leg = route.legs[0];
  const coordinates = decodePolyline(route.overview_polyline.points);

  return {
    geometry: { type: "LineString", coordinates },
    durationSec: leg.duration.value,
    distanceM: leg.distance.value,
    steps: (leg.steps ?? []).map((s: any) => ({
      instruction: s.html_instructions?.replace(/<[^>]*>/g, "") ?? "",
      distanceM: s.distance?.value ?? 0,
      durationSec: s.duration?.value ?? 0
    }))
  };
}
