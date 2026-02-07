import { ConditionFeatureCollection, SegmentKind, WeatherParams, RouteRiskResult } from "../types";
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
    kind,
    day_offset: String(dayOffset),
  };

  const query = new URLSearchParams(params);
  const response = await fetch(`${API_BASE}/v1/conditions?${query.toString()}`, {
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Failed to load conditions: ${response.status} ${body}`);
  }

  return (await response.json()) as ConditionFeatureCollection;
}

/* ── ML Predictions ── */

export async function triggerPrediction(weather: WeatherParams): Promise<{
  ok: boolean;
  days: number;
  segments: number;
  duration_ms: number;
}> {
  const res = await fetch(`${API_BASE}/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weather }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: "Prediction failed" }));
    throw new Error(data.error ?? `Prediction failed: ${res.status}`);
  }

  return res.json();
}

export async function fetchPredictionStatus(): Promise<{
  running: boolean;
  lastWeather: WeatherParams | null;
  lastRunAt: string | null;
  segmentCount: number;
  cachedDays: { day: number; predictions: number }[];
}> {
  const res = await fetch(`${API_BASE}/v1/predict/status`);
  if (!res.ok) throw new Error("Failed to fetch prediction status");
  return res.json();
}

/* ── Route Risk ── */

export async function fetchRouteRisk(
  coordinates: [number, number][],
  dayOffset: number = 0
): Promise<RouteRiskResult> {
  const res = await fetch(`${API_BASE}/v1/route-risk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ coordinates, day_offset: dayOffset }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: "Route risk failed" }));
    throw new Error(data.error ?? `Route risk failed: ${res.status}`);
  }

  return res.json();
}

/* ── Weather Analysis ── */

/** Extract ML-compatible weather params from NWS forecast data. */
export function extractWeatherParams(
  periods: WeatherPeriod[],
  hourly: HourlyPeriod[]
): WeatherParams | null {
  const snowKeywords = ["snow", "blizzard", "winter storm", "ice", "sleet", "freezing"];

  // Check if any forecast period mentions snow
  let hasSnow = false;
  let snowDays = 0;
  let minTempF = Infinity;
  let maxWindKmh = 0;
  let estimatedSnowfallCm = 0;

  for (const p of periods.slice(0, 14)) {
    const forecast = (p.shortForecast ?? "").toLowerCase();
    if (snowKeywords.some((kw) => forecast.includes(kw))) {
      hasSnow = true;
      if (p.isDaytime) snowDays++;
    }
    if (p.temperature < minTempF) minTempF = p.temperature;
  }

  if (!hasSnow) return null;

  // Estimate snowfall from hourly data
  let snowHours = 0;
  for (const h of hourly) {
    const forecast = (h.shortForecast ?? "").toLowerCase();
    const isSnowy = snowKeywords.some((kw) => forecast.includes(kw));
    if (isSnowy && h.precipChance > 30) {
      snowHours++;
      // Rough estimate: heavy snow ~2.5cm/hr, light ~0.5cm/hr
      const rate = h.precipChance > 70 ? 2.0 : h.precipChance > 50 ? 1.0 : 0.5;
      estimatedSnowfallCm += rate;
    }

    // Track max wind
    const windMatch = (h.windSpeed ?? "").match(/(\d+)/g);
    if (windMatch) {
      const windMph = Math.max(...windMatch.map(Number));
      const windKmh = windMph * 1.60934;
      if (windKmh > maxWindKmh) maxWindKmh = windKmh;
    }
  }

  // Convert min temp F to C
  const minTempC = (minTempF - 32) * 5 / 9;

  return {
    snowfall_cm: Math.max(estimatedSnowfallCm, 2), // at least 2cm if snow is mentioned
    min_temp_c: Math.round(minTempC * 10) / 10,
    max_wind_kmh: Math.round(maxWindKmh * 10) / 10,
    duration_days: Math.max(snowDays, 1),
  };
}

/* ── Autocomplete (Google Places via backend proxy) ── */

export type GeocodeSuggestion = {
  displayName: string;
  placeId: string;
  lat: number;
  lng: number;
};

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

export async function reverseGeocode(lat: number, lng: number): Promise<string | null> {
  const res = await fetch(
    `${API_BASE}/v1/maps/reverse-geocode?` + new URLSearchParams({ latlng: `${lat},${lng}` })
  );
  if (!res.ok) return null;

  const data = await res.json();
  if (data.status !== "OK" || !data.results?.length) return null;

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

/* ── Safe Route (risk-weighted Dijkstra via backend) ── */

export type SafeRouteResult = {
  found: boolean;
  geometry: { type: "LineString"; coordinates: [number, number][] };
  distanceM: number;
  segments: number;
  riskStats: {
    avgRisk: number;
    maxRisk: number;
    closedSegments: number;
  };
  elapsedMs: number;
};

export async function fetchSafeRoute(
  from: [number, number],
  to: [number, number],
  dayOffset: number
): Promise<SafeRouteResult> {
  const res = await fetch(`${API_BASE}/v1/safe-route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      origin: from, // [lng, lat]
      destination: to, // [lng, lat]
      day_offset: dayOffset,
    }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: "Safe route failed" }));
    throw new Error(data.error ?? `Safe route failed: ${res.status}`);
  }

  return res.json();
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
