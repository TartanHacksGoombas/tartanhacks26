import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { LngLatBounds } from "maplibre-gl";
import { fetchConditions, triggerPrediction, WeatherPeriod, HourlyPeriod } from "./utils/api";
import { ConditionFeatureCollection, SegmentKind, WeatherParams } from "./types";
import { useMapPadding } from "./utils/useMapPadding";
import NavigationPanel from "./components/NavigationPanel";
import UnifiedDayBar from "./components/UnifiedDayBar";
import LocationButton from "./components/LocationButton";

const SOURCE_ID = "conditions-source";
const LAYER_ID = "conditions-layer";
const USER_LOC_SOURCE = "user-location-source";
const USER_LOC_DOT = "user-location-dot";
const USER_LOC_RING = "user-location-ring";

/* ── TEMP: Generate simulated snowstorm weather ── */
function buildSimulatedWeather(params: WeatherParams, baseDate?: Date): { periods: WeatherPeriod[]; hourly: HourlyPeriod[] } {
  const { min_temp_c, max_wind_kmh, duration_days } = params;
  const minF = Math.round(min_temp_c * 9 / 5 + 32);
  const periods: WeatherPeriod[] = [];
  const hourly: HourlyPeriod[] = [];
  const now = baseDate ? new Date(baseDate) : new Date();
  now.setHours(0, 0, 0, 0);

  const dayProfiles: { forecast: string; highF: number; lowF: number; precipPct: number; nightForecast: string }[] = [];
  dayProfiles.push({ forecast: "Cloudy", highF: minF + 18, lowF: minF + 8, precipPct: 10, nightForecast: "Mostly Cloudy" });
  for (let d = 0; d < duration_days; d++) {
    const intensity = d === 0 ? "Heavy Snow" : d === duration_days - 1 ? "Snow Showers Likely" : "Snow";
    const hi = minF + 6 + d * 3;
    dayProfiles.push({ forecast: intensity, highF: hi, lowF: minF + d * 2, precipPct: 85 - d * 10, nightForecast: d < duration_days - 1 ? "Snow" : "Mostly Cloudy" });
  }
  while (dayProfiles.length < 7) {
    const d = dayProfiles.length;
    const recovering = d - duration_days - 1;
    dayProfiles.push({
      forecast: recovering < 2 ? "Partly Cloudy" : "Mostly Sunny",
      highF: minF + 20 + recovering * 4,
      lowF: minF + 10 + recovering * 3,
      precipPct: Math.max(0, 15 - recovering * 5),
      nightForecast: "Clear"
    });
  }

  const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

  for (let d = 0; d < 7; d++) {
    const dayDate = new Date(now);
    dayDate.setDate(dayDate.getDate() + d);
    const prof = dayProfiles[d];
    const dayName = d === 0 ? "Today" : d === 1 ? "Tomorrow" : dayNames[dayDate.getDay()];

    const dayStart = new Date(dayDate);
    dayStart.setHours(6);
    periods.push({
      name: dayName,
      startTime: dayStart.toISOString(),
      temperature: prof.highF,
      unit: "F",
      shortForecast: prof.forecast,
      isDaytime: true,
    });

    const nightStart = new Date(dayDate);
    nightStart.setHours(18);
    periods.push({
      name: dayName + " Night",
      startTime: nightStart.toISOString(),
      temperature: prof.lowF,
      unit: "F",
      shortForecast: prof.nightForecast,
      isDaytime: false,
    });

    const isStormDay = d >= 1 && d <= duration_days;
    for (let h = 0; h < 24; h++) {
      const hourDate = new Date(dayDate);
      hourDate.setHours(h);
      const isDayHour = h >= 6 && h < 20;
      let precip = 0;
      let hForecast = prof.forecast;
      if (isStormDay) {
        const intensity = Math.sin((h / 24) * Math.PI);
        precip = Math.min(100, Math.round(prof.precipPct * (0.5 + 0.5 * intensity)));
        hForecast = precip > 60 ? "Heavy Snow" : precip > 30 ? "Snow" : precip > 10 ? "Snow Showers" : "Cloudy";
      } else if (d === 0) {
        precip = h > 18 ? 20 : 5;
        hForecast = h > 18 ? "Cloudy" : "Mostly Cloudy";
      }
      const baseTemp = isDayHour ? prof.highF : prof.lowF;
      const tempVar = isDayHour ? Math.round(Math.sin(((h - 6) / 14) * Math.PI) * 4) : 0;

      hourly.push({
        startTime: hourDate.toISOString(),
        temperature: baseTemp + tempVar,
        unit: "F",
        shortForecast: hForecast,
        windSpeed: `${Math.round(max_wind_kmh * 0.621 * (isStormDay ? 0.7 + 0.3 * Math.random() : 0.3))} mph`,
        windDirection: "NW",
        precipChance: precip,
      });
    }
  }

  return { periods, hourly };
}

type StatusCounts = {
  open: number;
  low_risk: number;
  moderate_risk: number;
  high_risk: number;
  closed: number;
};

type PredictionStatus = "idle" | "loading" | "ready" | "error";

const emptyGeoJson: ConditionFeatureCollection = {
  type: "FeatureCollection",
  features: []
};

const MAP_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    basemap: {
      type: "raster",
      tiles: ["https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png"],
      tileSize: 256
    },
    labels: {
      type: "raster",
      tiles: ["https://basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png"],
      tileSize: 256
    }
  },
  layers: [
    { id: "basemap", type: "raster", source: "basemap", minzoom: 0, maxzoom: 19 },
    { id: "labels", type: "raster", source: "labels", minzoom: 0, maxzoom: 19 }
  ]
};

function boundsToBbox(bounds: LngLatBounds): [number, number, number, number] {
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  return [sw.lng, sw.lat, ne.lng, ne.lat];
}

export default function App() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const { sidebarRef, topBarRef, padding } = useMapPadding();
  const [kind] = useState<SegmentKind>("road");
  const [dayOffset, setDayOffset] = useState(0);
  const [mapReady, setMapReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [counts, setCounts] = useState<StatusCounts>({ open: 0, low_risk: 0, moderate_risk: 0, high_risk: 0, closed: 0 });
  const [error, setError] = useState<string | null>(null);
  const [predictionStatus, setPredictionStatus] = useState<PredictionStatus>("idle");
  const [weatherParams, setWeatherParams] = useState<WeatherParams | null>(null);
  const [userLocation, setUserLocation] = useState<[number, number] | null>(null);
  const watchIdRef = useRef<number | null>(null);
  const pendingLocationRef = useRef<{ lng: number; lat: number; accuracy: number } | null>(null);

  // ── TEMP: Simulate snowstorm toggle ──
  const SIM_PARAMS: WeatherParams = {
    snowfall_cm: 18,
    min_temp_c: -12,
    max_wind_kmh: 40,
    duration_days: 3,
  };
  const STORM_BASE_DATE = new Date(2026, 0, 21);

  const [simMode, setSimMode] = useState<"storm" | "today">("storm");
  const [simulatedWeather, setSimulatedWeather] = useState<{ periods: WeatherPeriod[]; hourly: HourlyPeriod[] } | null>(null);
  const simInitRef = useRef(false);

  const overallRisk = useMemo(() => {
    const total = counts.open + counts.low_risk + counts.moderate_risk + counts.high_risk + counts.closed;
    if (total === 0) return { label: "No Data", bg: "bg-slate-100", text: "text-slate-600" };
    const highPct = (counts.high_risk + counts.closed) / total;
    const modPct = counts.moderate_risk / total;
    if (highPct > 0.15) return { label: "High Risk", bg: "bg-red-50", text: "text-red-700" };
    if (highPct > 0.05 || modPct > 0.20) return { label: "Moderate Risk", bg: "bg-orange-50", text: "text-orange-700" };
    if (modPct > 0.05) return { label: "Low Risk", bg: "bg-yellow-50", text: "text-yellow-700" };
    return { label: "Roads Clear", bg: "bg-green-50", text: "text-green-700" };
  }, [counts]);

  // ── Trigger prediction when snow is detected ──
  const handleSnowDetected = useCallback(async (params: WeatherParams) => {
    setWeatherParams(params);
    setPredictionStatus("loading");
    try {
      await triggerPrediction(params);
      setPredictionStatus("ready");
      const map = mapRef.current;
      if (map) map.fire("moveend");
    } catch (e) {
      console.error("Prediction failed:", e);
      setPredictionStatus("error");
    }
  }, []);

  // Trigger simulation on mount and whenever simMode changes
  useEffect(() => {
    if (simMode === "storm") {
      setDayOffset(0);
      setSimulatedWeather(buildSimulatedWeather(SIM_PARAMS, STORM_BASE_DATE));
      if (!simInitRef.current) {
        simInitRef.current = true;
        handleSnowDetected(SIM_PARAMS);
      } else {
        handleSnowDetected(SIM_PARAMS);
      }
    } else {
      setDayOffset(0);
      setSimulatedWeather(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simMode]);

  // ── Apply location to map (add layers if needed, update source, pan) ──
  const applyLocationToMap = useCallback((map: maplibregl.Map, lng: number, lat: number, accuracy: number, pan: boolean) => {
    const geoJson: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "Point", coordinates: [lng, lat] }, properties: { accuracy } }]
    };

    if (map.getSource(USER_LOC_SOURCE)) {
      (map.getSource(USER_LOC_SOURCE) as maplibregl.GeoJSONSource).setData(geoJson);
    } else {
      map.addSource(USER_LOC_SOURCE, { type: "geojson", data: geoJson });

      map.addLayer({
        id: USER_LOC_RING,
        type: "circle",
        source: USER_LOC_SOURCE,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 10, 4, 14, 20, 18, 60],
          "circle-color": "rgba(66,133,244,0.15)",
          "circle-stroke-color": "rgba(66,133,244,0.3)",
          "circle-stroke-width": 1
        }
      }, "labels");

      map.addLayer({
        id: USER_LOC_DOT,
        type: "circle",
        source: USER_LOC_SOURCE,
        paint: {
          "circle-radius": 7,
          "circle-color": "#4285F4",
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 2.5
        }
      }, "labels");
    }

    if (pan) {
      map.easeTo({ center: [lng, lat], zoom: Math.max(map.getZoom(), 14), duration: 800 });
    }
  }, []);

  // ── User location handler ──
  const handleUserLocation = useCallback((lng: number, lat: number, accuracy: number) => {
    setUserLocation([lng, lat]);

    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) {
      pendingLocationRef.current = { lng, lat, accuracy };
      return;
    }

    pendingLocationRef.current = null;
    applyLocationToMap(map, lng, lat, accuracy, true);

    if (watchIdRef.current == null && navigator.geolocation) {
      watchIdRef.current = navigator.geolocation.watchPosition(
        (pos) => {
          const coord: [number, number] = [pos.coords.longitude, pos.coords.latitude];
          setUserLocation(coord);
          const m = mapRef.current;
          if (m && m.isStyleLoaded()) {
            applyLocationToMap(m, coord[0], coord[1], pos.coords.accuracy, false);
          }
        },
        () => {},
        { enableHighAccuracy: true }
      );
    }
  }, [applyLocationToMap]);

  // Request location automatically on mount
  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => handleUserLocation(pos.coords.longitude, pos.coords.latitude, pos.coords.accuracy),
      () => {},
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }, [handleUserLocation]);

  // Clean up geolocation watch on unmount
  useEffect(() => {
    return () => {
      if (watchIdRef.current != null) {
        navigator.geolocation.clearWatch(watchIdRef.current);
      }
    };
  }, []);

  // ── Map init ──
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: [-79.9959, 40.4406],
      zoom: 12,
      maplibreLogo: false,
      attributionControl: false
    });

    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl(), "bottom-right");

    map.on("error", (e) => {
      console.error("MapLibre error:", e);
      setError(e.error?.message ?? "Map failed to load");
    });

    map.on("load", () => {
      map.resize();
      setTimeout(() => map.resize(), 100);
      setTimeout(() => map.resize(), 500);

      map.addSource(SOURCE_ID, { type: "geojson", data: emptyGeoJson });

      const conditionColor: maplibregl.ExpressionSpecification = [
        "case",
        ["has", "label"],
        [
          "match", ["get", "label"],
          "open", "#16a34a",
          "low_risk", "#eab308",
          "moderate_risk", "#f97316",
          "high_risk", "#dc2626",
          "closed", "#7f1d1d",
          "#94a3b8",
        ],
        ["has", "riskCategory"],
        [
          "match", ["get", "riskCategory"],
          "very_low", "#16a34a",
          "low", "#eab308",
          "moderate", "#f97316",
          "high", "#dc2626",
          "very_high", "#7f1d1d",
          "#94a3b8",
        ],
        [
          "interpolate", ["linear"],
          ["coalesce", ["get", "closureProbability"], 0],
          0.00, "#16a34a",
          0.05, "#65a30d",
          0.15, "#eab308",
          0.35, "#f97316",
          0.55, "#dc2626",
          0.80, "#7f1d1d",
        ],
      ];

      // Major roads — always visible
      map.addLayer({
        id: LAYER_ID,
        type: "line",
        source: SOURCE_ID,
        filter: ["in", ["get", "highway"], ["literal", ["motorway", "trunk", "primary", "secondary"]]],
        paint: {
          "line-color": conditionColor,
          "line-width": ["interpolate", ["linear"], ["zoom"], 10, 2, 13, 4.5, 16, 7],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 10, 0.7, 14, 0.9]
        }
      }, "labels");

      // Tertiary roads — visible from zoom 13
      map.addLayer({
        id: LAYER_ID + "-tertiary",
        type: "line",
        source: SOURCE_ID,
        minzoom: 13,
        filter: ["==", ["get", "highway"], "tertiary"],
        paint: {
          "line-color": conditionColor,
          "line-width": ["interpolate", ["linear"], ["zoom"], 13, 2, 16, 4],
          "line-opacity": 0.85
        }
      }, "labels");

      // Residential / minor — visible from zoom 14, segments >= 80m
      map.addLayer({
        id: LAYER_ID + "-minor",
        type: "line",
        source: SOURCE_ID,
        minzoom: 14,
        filter: [
          "all",
          ["!", ["in", ["get", "highway"], ["literal", ["motorway", "trunk", "primary", "secondary", "tertiary"]]]],
          [">=", ["get", "lengthM"], 80]
        ] as any,
        paint: {
          "line-color": conditionColor,
          "line-width": ["interpolate", ["linear"], ["zoom"], 14, 1.5, 16, 3],
          "line-opacity": 0.8
        }
      }, "labels");

      // Segment click popup
      const conditionLayers = [LAYER_ID, LAYER_ID + "-tertiary", LAYER_ID + "-minor"];
      const onConditionClick = (event: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
        const feature = event.features?.[0];
        if (!feature || !feature.geometry || feature.geometry.type !== "LineString") return;

        const coordinates = feature.geometry.coordinates[0] as [number, number];
        const props = feature.properties ?? {};
        const closurePct = typeof props.closureProbability === "number"
          ? `${(props.closureProbability * 100).toFixed(0)}%` : null;
        const riskCat = props.riskCategory ?? props.label ?? "unknown";

        new maplibregl.Popup().setLngLat(coordinates).setHTML(`
          <div style="font-family: Avenir Next, Segoe UI, sans-serif; max-width: 260px;">
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#475569">Road Segment</div>
            <div style="font-size:16px;font-weight:700;margin-top:4px">${props.name ?? "Unnamed segment"}</div>
            <div style="margin-top:6px">
              <strong>Safety:</strong> ${props.score ?? "N/A"}/100 (${props.label ?? "unknown"})
              ${closurePct != null ? `<br><strong>Risk:</strong> ${closurePct} (${riskCat})` : ""}
            </div>
          </div>
        `).addTo(map);
      };
      for (const layerId of conditionLayers) {
        map.on("click", layerId, onConditionClick);
        map.on("mouseenter", layerId, () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", layerId, () => { map.getCanvas().style.cursor = ""; });
      }

      setMapReady(true);

      // Apply any location that arrived before map was ready
      const pending = pendingLocationRef.current;
      if (pending) {
        pendingLocationRef.current = null;
        applyLocationToMap(map, pending.lng, pending.lat, pending.accuracy, true);
      }
    });

    return () => { map.remove(); mapRef.current = null; };
  }, []);

  // ── Keep map padding in sync with UI overlays ──
  const initialPaddingRef = useRef(true);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (initialPaddingRef.current) {
      initialPaddingRef.current = false;
      return;
    }
    map.easeTo({ padding, duration: 300 });
  }, [padding, mapReady]);

  // ── Load conditions on move ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;

    const load = async () => {
      if (!map.isStyleLoaded() || !map.getSource(SOURCE_ID)) return;
      setLoading(true);
      setError(null);

      try {
        const bbox = boundsToBbox(map.getBounds());
        const data = await fetchConditions(bbox, kind, dayOffset);

        const mapData = {
          type: "FeatureCollection",
          features: data.features.map((f) => ({
            ...f,
            properties: {
              kind: f.properties.kind,
              name: f.properties.name ?? "Unnamed segment",
              highway: (f.properties as any).highway ?? "unclassified",
              lengthM: (f.properties as any).lengthM ?? 0,
              score: f.properties.score,
              label: f.properties.label,
              closureProbability: f.properties.closureProbability,
              riskCategory: f.properties.riskCategory ?? f.properties.label,
            }
          }))
        };

        (map.getSource(SOURCE_ID) as maplibregl.GeoJSONSource).setData(mapData as any);

        const nextCounts = data.features.reduce(
          (acc, f) => {
            const l = f.properties.label;
            if (l === "open") acc.open += 1;
            else if (l === "low_risk") acc.low_risk += 1;
            else if (l === "moderate_risk") acc.moderate_risk += 1;
            else if (l === "high_risk") acc.high_risk += 1;
            else if (l === "closed") acc.closed += 1;
            return acc;
          },
          { open: 0, low_risk: 0, moderate_risk: 0, high_risk: 0, closed: 0 } as StatusCounts
        );
        setCounts(nextCounts);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Failed to load map data");
      } finally {
        setLoading(false);
      }
    };

    load();
    map.on("moveend", load);
    return () => { map.off("moveend", load); };
  }, [kind, dayOffset, mapReady]);

  // ── Render ──
  return (
    <div className="relative w-full h-full" style={{ minHeight: "100vh", height: "100vh" }}>
      <div
        ref={containerRef}
        role="application"
        aria-label="Map"
        className="absolute inset-0 w-full h-full"
        style={{ width: "100%", height: "100%", minHeight: "100vh" }}
      />

      {/* Unified top bar */}
      <div className="absolute top-4 z-20" style={{ left: "calc(320px + 2rem + 1rem)", right: "1rem" }}>
        <UnifiedDayBar
          ref={topBarRef}
          value={dayOffset}
          onChange={setDayOffset}
          onSnowDetected={handleSnowDetected}
          simulatedWeather={simulatedWeather}
        />
      </div>

      {/* Location button — above MapLibre nav controls */}
      <div className="absolute z-10" style={{ bottom: 120, right: 10 }}>
        <LocationButton onLocation={handleUserLocation} />
      </div>

      <aside ref={sidebarRef} className="absolute left-4 top-4 z-10 w-[320px] max-h-[90vh] overflow-y-auto rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-xl backdrop-blur">
        {/* Header */}
        <div className="mb-3">
          <h1 className="text-xl font-bold tracking-tight text-slate-900">Snow Route Pittsburgh</h1>
          <p className="text-sm text-slate-600">
            ML-powered road closure risk predictions.
          </p>
        </div>

        {/* Overall risk badge */}
        <div className={`mb-3 rounded-xl px-3 py-2.5 text-center ${overallRisk.bg}`}>
          <div className={`text-lg font-bold ${overallRisk.text}`}>{overallRisk.label}</div>
          <div className="mt-0.5 text-[11px] font-medium text-slate-500">
            {dayOffset === 0 ? "Today" : `Day +${dayOffset}`} · model forecast
          </div>
        </div>

        {/* Prediction status (loading + error only) */}
        {predictionStatus === "loading" && (
          <div className="mb-3 rounded-lg px-3 py-2 text-xs bg-blue-50 text-blue-700">
            Running ML prediction model...
          </div>
        )}
        {predictionStatus === "error" && (
          <div className="mb-3 rounded-lg px-3 py-2 text-xs bg-red-50 text-red-700">
            Prediction failed — showing fallback data
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-3 rounded-lg px-3 py-2 text-xs bg-red-50 text-red-700 truncate" title={error}>
            {error}
          </div>
        )}

        {/* Navigation */}
        <NavigationPanel mapRef={mapRef} mapPadding={padding} dayOffset={dayOffset} />
      </aside>

      {/* TEMP: Date toggle — bottom-left */}
      <div className="absolute bottom-6 left-4 z-20">
        <button
          onClick={() => setSimMode(simMode === "storm" ? "today" : "storm")}
          className={`rounded-xl border px-4 py-2.5 text-xs font-semibold shadow-lg backdrop-blur transition-colors ${
            simMode === "storm"
              ? "border-blue-300 bg-blue-50/90 text-blue-800 hover:bg-blue-100/90"
              : "border-slate-300 bg-white/90 text-slate-700 hover:bg-slate-50/90"
          }`}
        >
          <div className="flex items-center gap-2">
            <span className="text-base">{simMode === "storm" ? "❄️" : "☀️"}</span>
            <div className="text-left">
              <div>{simMode === "storm" ? "Storm Mode: Jan 21" : "Live Mode: Today"}</div>
              <div className="text-[10px] font-normal opacity-70">
                {simMode === "storm" ? "Simulated snowstorm week" : "Real weather data"}
              </div>
            </div>
          </div>
        </button>
      </div>
    </div>
  );
}
