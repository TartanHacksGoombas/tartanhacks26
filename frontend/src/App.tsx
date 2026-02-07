import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { LngLatBounds } from "maplibre-gl";
import { fetchConditions, triggerPrediction } from "./utils/api";
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
  const { sidebarRef, topBarRef, padding } = useMapPadding();
  const [kind] = useState<SegmentKind>("road");
  const [dayOffset, setDayOffset] = useState(0);
  const [mapReady, setMapReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [counts, setCounts] = useState<StatusCounts>({ open: 0, low_risk: 0, moderate_risk: 0, closed: 0 });
  const [error, setError] = useState<string | null>(null);
  const [predictionStatus, setPredictionStatus] = useState<PredictionStatus>("idle");
  const [weatherParams, setWeatherParams] = useState<WeatherParams | null>(null);
  const [userLocation, setUserLocation] = useState<[number, number] | null>(null);
  const watchIdRef = useRef<number | null>(null);
  const pendingLocationRef = useRef<{ lng: number; lat: number; accuracy: number } | null>(null);

  const overallRisk = useMemo(() => {
    const total = counts.open + counts.low_risk + counts.moderate_risk + counts.closed;
    if (total === 0) return { label: "No Data", bg: "bg-slate-100", text: "text-slate-600" };
    const highPct = counts.closed / total;
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
      // Force map reload to show new predictions
      const map = mapRef.current;
      if (map) map.fire("moveend");
    } catch (e) {
      console.error("Prediction failed:", e);
      setPredictionStatus("error");
    }
  }, []);

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
      // Map not ready yet — queue the location to apply once it loads
      pendingLocationRef.current = { lng, lat, accuracy };
      return;
    }

    pendingLocationRef.current = null;
    applyLocationToMap(map, lng, lat, accuracy, true);

    // Start continuous watch if not already watching
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

  // Request location automatically on mount (pending ref handles map not ready yet)
  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => handleUserLocation(pos.coords.longitude, pos.coords.latitude, pos.coords.accuracy),
      () => {},  // silently ignore denial on auto-request
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
    // Skip the first render — map starts un-padded and we don't want to animate on load
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
            else if (l === "closed") acc.closed += 1;
            return acc;
          },
          { open: 0, low_risk: 0, moderate_risk: 0, closed: 0 } as StatusCounts
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
    </div>
  );
}
