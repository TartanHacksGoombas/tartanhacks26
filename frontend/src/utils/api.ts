import { ConditionFeatureCollection, SegmentKind } from "../types";
import { decodePolyline } from "./polyline";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:4000";

/* ── Conditions ── */

export async function fetchConditions(
  bbox: [number, number, number, number],
  kind: SegmentKind,
  dayOffset: number = 0
): Promise<ConditionFeatureCollection> {
  const params: Record<string, string> = {
    bbox: bbox.join(","),
    kind
  };

  // Pass the target date so the backend can serve day-specific predictions
  if (dayOffset > 0) {
    const target = new Date();
    target.setDate(target.getDate() + dayOffset);
    params.target_date = target.toISOString().split("T")[0]; // YYYY-MM-DD
  }

  const query = new URLSearchParams(params);
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

/** Reverse-geocode coordinates to a human-readable address. */
export async function reverseGeocode(lat: number, lng: number): Promise<string | null> {
  const res = await fetch(
    `${API_BASE}/v1/maps/reverse-geocode?` + new URLSearchParams({ latlng: `${lat},${lng}` })
  );
  if (!res.ok) return null;

  const data = await res.json();
  if (data.status !== "OK" || !data.results?.length) return null;

  // Return the first result's formatted address (usually the most specific)
  return data.results[0].formatted_address ?? null;
}

/* ── Routing (Google Directions via backend proxy) ── */

export type RouteResult = {
  geometry: { type: "LineString"; coordinates: [number, number][] };
  durationSec: number;
  distanceM: number;
  steps: Array<{ instruction: string; distanceM: number; durationSec: number }>;
};

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

/* ── Weather (NWS API — free, no key) ── */

export type WeatherPeriod = {
  name: string;
  startTime: string;
  temperature: number;
  unit: string;
  shortForecast: string;
  isDaytime: boolean;
};

/** Fetch the 7-day forecast for Pittsburgh from the National Weather Service. */
export async function fetchWeather(): Promise<WeatherPeriod[]> {
  const res = await fetch("https://api.weather.gov/gridpoints/PBZ/77,65/forecast", {
    headers: { "User-Agent": "SnowRoutePittsburgh/1.0" }
  });
  if (!res.ok) return [];

  const data = await res.json();
  const periods: any[] = data.properties?.periods ?? [];

  return periods.map((p) => ({
    name: p.name,
    startTime: p.startTime,
    temperature: p.temperature,
    unit: p.temperatureUnit,
    shortForecast: p.shortForecast,
    isDaytime: p.isDaytime
  }));
}

export type HourlyPeriod = {
  startTime: string;
  temperature: number;
  unit: string;
  shortForecast: string;
  windSpeed: string;
  windDirection: string;
  precipChance: number;
};

/** Fetch hourly forecast for Pittsburgh (up to 156 hours). */
export async function fetchWeatherHourly(): Promise<HourlyPeriod[]> {
  const res = await fetch("https://api.weather.gov/gridpoints/PBZ/77,65/forecast/hourly", {
    headers: { "User-Agent": "SnowRoutePittsburgh/1.0" }
  });
  if (!res.ok) return [];

  const data = await res.json();
  const periods: any[] = data.properties?.periods ?? [];

  return periods.map((p) => ({
    startTime: p.startTime,
    temperature: p.temperature,
    unit: p.temperatureUnit,
    shortForecast: p.shortForecast,
    windSpeed: p.windSpeed ?? "",
    windDirection: p.windDirection ?? "",
    precipChance: p.probabilityOfPrecipitation?.value ?? 0
  }));
}
