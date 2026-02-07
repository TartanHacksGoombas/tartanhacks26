import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { LngLatBounds } from "maplibre-gl";
import { fetchConditions, triggerPrediction } from "./utils/api";
import { ConditionFeatureCollection, SegmentKind, WeatherParams } from "./types";
import { useMapPadding } from "./utils/useMapPadding";
import StatusCard from "./components/StatusCard";
import NavigationPanel from "./components/NavigationPanel";
import WeatherBar from "./components/WeatherBar";
import TimeSlider from "./components/TimeSlider";

const SOURCE_ID = "conditions-source";
const LAYER_ID = "conditions-layer";

type StatusCounts = {
  open: number;
  low_risk: number;
  moderate_risk: number;
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
  const { sidebarRef, weatherRef, padding } = useMapPadding();
  const [kind] = useState<SegmentKind>("road");
  const [dayOffset, setDayOffset] = useState(0);
  const [mapReady, setMapReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [counts, setCounts] = useState<StatusCounts>({ open: 0, low_risk: 0, moderate_risk: 0, closed: 0 });
  const [error, setError] = useState<string | null>(null);
  const [predictionStatus, setPredictionStatus] = useState<PredictionStatus>("idle");
  const [weatherParams, setWeatherParams] = useState<WeatherParams | null>(null);

  const totalVisible = useMemo(
    () => counts.open + counts.low_risk + counts.moderate_risk + counts.closed,
    [counts]
  );

  // ── Trigger prediction when snow is detected ──
  const handleSnowDetected = useCallback(async (params: WeatherParams) => {
    setWeatherParams(params);
    setPredictionStatus("loading");
    try {
      await triggerPrediction(params);
      setPredictionStatus("ready");
      // Force map reload to show new predictions
      const map = mapRef.current;
      if (map) map.fire("moveend");
    } catch (e) {
      console.error("Prediction failed:", e);
      setPredictionStatus("error");
    }
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
        "match", ["get", "label"],
        "open", "#16a34a",
        "low_risk", "#eab308",
        "moderate_risk", "#f97316",
        "closed", "#dc2626",
        "#64748b"
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
    });

    return () => { map.remove(); mapRef.current = null; };
  }, []);

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
            else if (l === "closed") acc.closed += 1;
            return acc;
          },
          { open: 0, low_risk: 0, moderate_risk: 0, closed: 0 } as StatusCounts
        );
        setCounts(nextCounts);
        setLastUpdated(new Date().toISOString());
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

      <WeatherBar ref={weatherRef} activeDayOffset={dayOffset} onSnowDetected={handleSnowDetected} />

      {/* Time slider */}
      <div className="absolute top-4 z-20" style={{ left: "calc(320px + 2rem + 1rem)", right: "calc(500px + 2rem + 1rem)" }}>
        <TimeSlider value={dayOffset} onChange={setDayOffset} />
      </div>

      <aside ref={sidebarRef} className="absolute left-4 top-4 z-10 w-[320px] max-h-[90vh] overflow-y-auto rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-xl backdrop-blur">
        {/* Header */}
        <div className="mb-3">
          <h1 className="text-xl font-bold tracking-tight text-slate-900">Snow Route Pittsburgh</h1>
          <p className="text-sm text-slate-600">
            ML-powered road closure risk predictions. Green = safe, red = high risk.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
            <span className="flex items-center gap-1"><span className="h-1.5 w-4 rounded bg-green-600" /> Open</span>
            <span className="flex items-center gap-1"><span className="h-1.5 w-4 rounded bg-yellow-500" /> Low risk</span>
            <span className="flex items-center gap-1"><span className="h-1.5 w-4 rounded bg-orange-500" /> Moderate</span>
            <span className="flex items-center gap-1"><span className="h-1.5 w-4 rounded bg-red-600" /> High risk</span>
          </div>
        </div>

        {/* Prediction status */}
        {predictionStatus !== "idle" && (
          <div className={`mb-3 rounded-lg px-3 py-2 text-xs ${
            predictionStatus === "loading" ? "bg-blue-50 text-blue-700" :
            predictionStatus === "ready" ? "bg-green-50 text-green-700" :
            "bg-red-50 text-red-700"
          }`}>
            {predictionStatus === "loading" && "Running ML prediction model..."}
            {predictionStatus === "ready" && weatherParams && (
              <>Predictions ready: {weatherParams.snowfall_cm}cm snow, {weatherParams.min_temp_c}°C, {weatherParams.duration_days}d storm</>
            )}
            {predictionStatus === "error" && "Prediction failed — showing fallback data"}
          </div>
        )}

        {/* Status cards */}
        <div className="grid grid-cols-4 gap-2 text-sm">
          <StatusCard label="Open" count={counts.open} bgClass="bg-green-50" textClass="text-green-800" />
          <StatusCard label="Low risk" count={counts.low_risk} bgClass="bg-yellow-50" textClass="text-yellow-800" />
          <StatusCard label="Moderate" count={counts.moderate_risk} bgClass="bg-orange-50" textClass="text-orange-800" />
          <StatusCard label="High risk" count={counts.closed} bgClass="bg-red-50" textClass="text-red-800" />
        </div>

        {/* Status bar */}
        <div className="mt-4 space-y-1 text-xs text-slate-600">
          <div>Visible segments: {totalVisible}</div>
          <div>
            {loading ? "Refreshing..." : "Map synced to viewport"}
            {dayOffset > 0 && (
              <span className="ml-1 rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700">
                +{dayOffset}d forecast
              </span>
            )}
          </div>
          {lastUpdated ? <div>Updated: {new Date(lastUpdated).toLocaleTimeString()}</div> : null}
          <div className="h-4 truncate text-red-700" title={error ?? ""}>{error ?? "\u00A0"}</div>
        </div>

        {/* Navigation */}
        <NavigationPanel mapRef={mapRef} mapPadding={padding} dayOffset={dayOffset} />
      </aside>
    </div>
  );
}
