import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { fetchRoute, fetchRouteRisk, fetchSafeRoute, reverseGeocode, GeocodeSuggestion, RouteResult, SafeRouteResult } from "../utils/api";
import { RouteRiskResult } from "../types";
import SearchInput from "./SearchInput";
import Button from "./Button";
import ToggleButton from "./ToggleButton";

const ROUTE_SOURCE = "nav-route-src";
const ROUTE_OUTLINE = "nav-route-outline";
const ROUTE_LINE = "nav-route-line";
const ROUTE_LABEL = "nav-route-label";

const RISK_SOURCE = "nav-risk-src";
const RISK_LAYER = "nav-risk-layer";

const SAFE_SOURCE = "nav-safe-src";
const SAFE_OUTLINE = "nav-safe-outline";
const SAFE_LINE = "nav-safe-line";
const SAFE_LABEL = "nav-safe-label";

const SAFE_RISK_SOURCE = "nav-safe-risk-src";
const SAFE_RISK_LAYER = "nav-safe-risk-layer";

type MapPadding = { top: number; right: number; bottom: number; left: number };

type NavigationPanelProps = {
  mapRef: React.RefObject<maplibregl.Map | null>;
  mapPadding: MapPadding;
  dayOffset?: number;
};

/** Ensure the route source + layers exist on the map. Safe to call repeatedly. */
function ensureRouteLayers(map: maplibregl.Map) {
  if (!map.getSource(ROUTE_SOURCE)) {
    map.addSource(ROUTE_SOURCE, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });
  }
  if (!map.getLayer(ROUTE_OUTLINE)) {
    map.addLayer({
      id: ROUTE_OUTLINE,
      type: "line",
      source: ROUTE_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#1e40af", "line-width": 12, "line-opacity": 0.65 }
    });
  }
  if (!map.getLayer(ROUTE_LINE)) {
    map.addLayer({
      id: ROUTE_LINE,
      type: "line",
      source: ROUTE_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#2563eb", "line-width": 7, "line-opacity": 1 }
    });
  }
  if (!map.getLayer(ROUTE_LABEL)) {
    map.addLayer({
      id: ROUTE_LABEL,
      type: "symbol",
      source: ROUTE_SOURCE,
      layout: {
        "symbol-placement": "line",
        "text-field": "Day-of route",
        "text-size": 11,
        "text-font": ["Open Sans Semibold", "Arial Unicode MS Bold"],
        "text-allow-overlap": true,
        "text-ignore-placement": true
      },
      paint: {
        "text-color": "#1e3a8a",
        "text-halo-color": "#ffffff",
        "text-halo-width": 1.5,
        "text-halo-blur": 0.2
      }
    });
  }
}

const RISK_COLOR_EXPR: maplibregl.ExpressionSpecification = [
  "match", ["get", "riskCategory"],
  "very_low", "#16a34a",
  "low", "#eab308",
  "moderate", "#f97316",
  "high", "#dc2626",
  "very_high", "#7f1d1d",
  "#64748b"
];

/** Ensure the risk overlay source + layer exist. Safe to call repeatedly. */
function ensureRiskLayers(map: maplibregl.Map) {
  if (!map.getSource(RISK_SOURCE)) {
    map.addSource(RISK_SOURCE, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });
  }
  if (!map.getLayer(RISK_LAYER)) {
    map.addLayer({
      id: RISK_LAYER,
      type: "line",
      source: RISK_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": RISK_COLOR_EXPR,
        "line-width": 7,
        "line-opacity": 0.95
      }
    });
  }
}

/** Ensure the safe-route source + layers exist. Safe to call repeatedly. */
function ensureSafeLayers(map: maplibregl.Map) {
  if (!map.getSource(SAFE_SOURCE)) {
    map.addSource(SAFE_SOURCE, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });
  }
  if (!map.getLayer(SAFE_OUTLINE)) {
    map.addLayer({
      id: SAFE_OUTLINE,
      type: "line",
      source: SAFE_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#065f46", "line-width": 12, "line-opacity": 0.7 }
    });
  }
  if (!map.getLayer(SAFE_LINE)) {
    map.addLayer({
      id: SAFE_LINE,
      type: "line",
      source: SAFE_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": "#10b981",
        "line-width": 6,
        "line-opacity": 0.9,
        "line-dasharray": [1.5, 1.5]
      }
    });
  }
  if (!map.getLayer(SAFE_LABEL)) {
    map.addLayer({
      id: SAFE_LABEL,
      type: "symbol",
      source: SAFE_SOURCE,
      layout: {
        "symbol-placement": "line",
        "text-field": "Safer route",
        "text-size": 11,
        "text-font": ["Open Sans Semibold", "Arial Unicode MS Bold"],
        "text-allow-overlap": true,
        "text-ignore-placement": true
      },
      paint: {
        "text-color": "#065f46",
        "text-halo-color": "#ffffff",
        "text-halo-width": 1.5,
        "text-halo-blur": 0.2
      }
    });
  }
}

/** Ensure the safe-route risk coloring source + layer exist. */
function ensureSafeRiskLayers(map: maplibregl.Map) {
  if (!map.getSource(SAFE_RISK_SOURCE)) {
    map.addSource(SAFE_RISK_SOURCE, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });
  }
  if (!map.getLayer(SAFE_RISK_LAYER)) {
    map.addLayer({
      id: SAFE_RISK_LAYER,
      type: "line",
      source: SAFE_RISK_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": RISK_COLOR_EXPR,
        "line-width": 6,
        "line-opacity": 0.85,
        "line-dasharray": [1.5, 1.5]
      }
    });
  }
}

function riskColor(category: string): string {
  switch (category) {
    case "very_low": return "#16a34a";
    case "low": return "#eab308";
    case "moderate": return "#f97316";
    case "high": return "#dc2626";
    case "very_high": return "#7f1d1d";
    default: return "#64748b";
  }
}

export default function NavigationPanel({ mapRef, mapPadding, dayOffset = 0 }: NavigationPanelProps) {
  const [open, setOpen] = useState(false);
  const [fromText, setFromText] = useState("");
  const [toText, setToText] = useState("");
  const [fromCoord, setFromCoord] = useState<[number, number] | null>(null);
  const [toCoord, setToCoord] = useState<[number, number] | null>(null);
  const [routeResult, setRouteResult] = useState<RouteResult | null>(null);
  const [routeRisk, setRouteRisk] = useState<RouteRiskResult | null>(null);
  const [safeRouteRisk, setSafeRouteRisk] = useState<RouteRiskResult | null>(null);
  const [safeRoute, setSafeRoute] = useState<SafeRouteResult | null>(null);
  const [safeRouteLoading, setSafeRouteLoading] = useState(false);
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState<string | null>(null);
  const startMarkerRef = useRef<maplibregl.Marker | null>(null);
  const endMarkerRef = useRef<maplibregl.Marker | null>(null);

  // Keep latest padding in a ref so imperative pan calls always use fresh values
  const paddingRef = useRef(mapPadding);
  paddingRef.current = mapPadding;

  const computePad = useCallback((uiPad: MapPadding, map?: maplibregl.Map) => {
    const buffer = 24;
    const minPad = 28;
    const raw = {
      top: Math.max(uiPad.top + buffer, minPad),
      right: Math.max(uiPad.right + buffer, minPad),
      bottom: Math.max(uiPad.bottom + buffer, minPad),
      left: Math.max(uiPad.left + buffer, minPad),
    };

    if (!map) return raw;
    const w = Math.max(map.getContainer().clientWidth, 1);
    const h = Math.max(map.getContainer().clientHeight, 1);
    const maxX = Math.max(40, Math.floor(w * 0.45));
    const maxY = Math.max(40, Math.floor(h * 0.45));
    return {
      top: Math.min(raw.top, maxY),
      right: Math.min(raw.right, maxX),
      bottom: Math.min(raw.bottom, maxY),
      left: Math.min(raw.left, maxX),
    };
  }, []);

  // Keep map padding in sync with widget sizes for all interactions
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.setPadding(computePad(mapPadding, map));
  }, [mapPadding, mapRef, computePad]);

  const geolocatedRef = useRef(false);

  /** Lazily create the route layers when the panel first opens. */
  useEffect(() => {
    if (!open) return;
    const map = mapRef.current;
    if (!map) return;

    const addLayers = () => {
      if (map.isStyleLoaded()) {
        ensureRouteLayers(map);
        ensureRiskLayers(map);
        ensureSafeLayers(map);
        ensureSafeRiskLayers(map);
      }
    };

    // Style might already be loaded, or we wait for it
    addLayers();
    map.on("styledata", addLayers);
    return () => { map.off("styledata", addLayers); };
  }, [open, mapRef]);

  // Dim background conditions layers when a route is active
  const CONDITION_LAYERS = ["conditions-layer", "conditions-layer-tertiary", "conditions-layer-minor"];
  const dimmedRef = useRef(false);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const hasRoute = routeResult !== null;
    if (hasRoute && !dimmedRef.current) {
      dimmedRef.current = true;
      for (const id of CONDITION_LAYERS) {
        if (map.getLayer(id)) map.setPaintProperty(id, "line-opacity", 0.2);
      }
    } else if (!hasRoute && dimmedRef.current) {
      dimmedRef.current = false;
      // Restore original opacities
      if (map.getLayer(CONDITION_LAYERS[0]))
        map.setPaintProperty(CONDITION_LAYERS[0], "line-opacity", ["interpolate", ["linear"], ["zoom"], 10, 0.7, 14, 0.9]);
      if (map.getLayer(CONDITION_LAYERS[1]))
        map.setPaintProperty(CONDITION_LAYERS[1], "line-opacity", 0.85);
      if (map.getLayer(CONDITION_LAYERS[2]))
        map.setPaintProperty(CONDITION_LAYERS[2], "line-opacity", 0.8);
    }
  }, [routeResult, mapRef]);

  /** Create a pin-shaped SVG marker element. */
  const createPinElement = useCallback((color: string, label: string) => {
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width:30px;height:42px;cursor:pointer;";
    wrapper.innerHTML = `
      <svg viewBox="0 0 30 42" width="30" height="42" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="pin-shadow-${label}" x="-20%" y="-10%" width="140%" height="130%">
            <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-opacity="0.3"/>
          </filter>
        </defs>
        <path d="M15 0C8.1 0 2.5 5.6 2.5 12.5C2.5 22.5 15 42 15 42S27.5 22.5 27.5 12.5C27.5 5.6 21.9 0 15 0Z"
              fill="${color}" stroke="#fff" stroke-width="1.5" filter="url(#pin-shadow-${label})"/>
        <circle cx="15" cy="12.5" r="5" fill="#fff"/>
        <text x="15" y="15.5" text-anchor="middle" fill="${color}" font-size="9" font-weight="bold" font-family="system-ui">${label}</text>
      </svg>
    `;
    return wrapper;
  }, []);

  /** Auto-fill "From" with user's current location when the panel first opens. */
  useEffect(() => {
    if (!open || geolocatedRef.current || fromCoord) return;
    if (!("geolocation" in navigator)) return;

    geolocatedRef.current = true;
    setFromText("Locating…");

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        const coord: [number, number] = [longitude, latitude];
        setFromCoord(coord);

        // Place the marker
        const map = mapRef.current;
        if (map) {
          const el = createPinElement("#16a34a", "A");
          if (startMarkerRef.current) startMarkerRef.current.remove();
          startMarkerRef.current = new maplibregl.Marker({ element: el, anchor: "bottom" })
            .setLngLat(coord)
            .addTo(map);
        }

        // Reverse-geocode for a readable label
        const address = await reverseGeocode(latitude, longitude);
        setFromText(address ? address.split(",")[0] : "My Location");
      },
      () => {
        // Geolocation denied or failed — just clear the placeholder
        setFromText("");
        geolocatedRef.current = false;
      },
      { enableHighAccuracy: true, timeout: 8000 }
    );
  }, [open, fromCoord, mapRef, createPinElement]);

  /** Place a pin marker. */
  const placeMarker = useCallback(
    (lngLat: [number, number], color: string, label: string, ref: React.MutableRefObject<maplibregl.Marker | null>) => {
      const map = mapRef.current;
      if (!map) return;
      if (ref.current) ref.current.remove();
      const el = createPinElement(color, label);
      ref.current = new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat(lngLat).addTo(map);
    },
    [mapRef, createPinElement]
  );

  /**
   * Imperatively fit the map to the given points with generous spacing.
   * Uses paddingRef so it always reads the latest dynamic padding.
   */
  const panToFit = useCallback(
    (points: [number, number][]) => {
      const map = mapRef.current;
      if (!map || points.length === 0) return;

      // Use overlay-aware padding with a modest buffer so the route is
      // as large as possible while staying clear of widgets and borders.
      const uiPad = paddingRef.current;
      const pad = computePad(uiPad, map);
      map.setPadding(pad);

      if (points.length === 1) {
        map.flyTo({ center: points[0], zoom: 15, duration: 600 });
        return;
      }

      const bounds = points.reduce(
        (b, c) => b.extend(c),
        new maplibregl.LngLatBounds(points[0], points[0])
      );
      map.fitBounds(bounds, { maxZoom: 16, duration: 600 });
    },
    [mapRef]
  );

  const pickFrom = useCallback(
    (s: GeocodeSuggestion) => {
      console.log("[Nav] pickFrom called:", s.displayName, s.lng, s.lat);
      const coord: [number, number] = [s.lng, s.lat];
      setFromText(s.displayName.split(",")[0]);
      setFromCoord(coord);
      placeMarker(coord, "#16a34a", "A", startMarkerRef);
      const pts: [number, number][] = [coord];
      if (endMarkerRef.current) {
        const ll = endMarkerRef.current.getLngLat();
        pts.push([ll.lng, ll.lat]);
      }
      console.log("[Nav] pickFrom calling panToFit with", pts.length, "points");
      panToFit(pts);
    },
    [placeMarker, panToFit]
  );

  const pickTo = useCallback(
    (s: GeocodeSuggestion) => {
      console.log("[Nav] pickTo called:", s.displayName, s.lng, s.lat);
      const coord: [number, number] = [s.lng, s.lat];
      setToText(s.displayName.split(",")[0]);
      setToCoord(coord);
      placeMarker(coord, "#dc2626", "B", endMarkerRef);
      const pts: [number, number][] = [coord];
      if (startMarkerRef.current) {
        const ll = startMarkerRef.current.getLngLat();
        pts.push([ll.lng, ll.lat]);
      }
      console.log("[Nav] pickTo calling panToFit with", pts.length, "points");
      panToFit(pts);
    },
    [placeMarker, panToFit]
  );

  const setRouteData = useCallback(
    (coords: [number, number][]) => {
      const map = mapRef.current;
      if (!map) return;

      ensureRouteLayers(map);

      const src = map.getSource(ROUTE_SOURCE) as maplibregl.GeoJSONSource | undefined;
      if (!src) {
        console.error("[Route] source still missing after ensureRouteLayers");
        return;
      }

      src.setData({
        type: "FeatureCollection",
        features: coords.length > 0
          ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } }]
          : []
      });
    },
    [mapRef]
  );

  /** Update the risk overlay GeoJSON on the map. */
  const setRiskData = useCallback(
    (features: any[]) => {
      const map = mapRef.current;
      if (!map) return;
      ensureRiskLayers(map);
      const src = map.getSource(RISK_SOURCE) as maplibregl.GeoJSONSource | undefined;
      if (!src) return;
      src.setData({ type: "FeatureCollection", features });
    },
    [mapRef]
  );

  /** Update the safe-route risk coloring GeoJSON on the map. */
  const setSafeRiskData = useCallback(
    (features: any[]) => {
      const map = mapRef.current;
      if (!map) return;
      ensureSafeRiskLayers(map);
      const src = map.getSource(SAFE_RISK_SOURCE) as maplibregl.GeoJSONSource | undefined;
      if (!src) return;
      src.setData({ type: "FeatureCollection", features });
    },
    [mapRef]
  );

  /** Update the safe-route GeoJSON on the map. */
  const setSafeRouteLineData = useCallback(
    (coords: [number, number][]) => {
      const map = mapRef.current;
      if (!map) return;
      ensureSafeLayers(map);
      const src = map.getSource(SAFE_SOURCE) as maplibregl.GeoJSONSource | undefined;
      if (!src) return;
      src.setData({
        type: "FeatureCollection",
        features: coords.length > 0
          ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } }]
          : []
      });
    },
    [mapRef]
  );

  /**
   * When there's a route, fetch risk overlay + safe alternative for the selected day.
   * Risk coloring is always shown on routes. Safe alternative only on day 1+.
   */
  useEffect(() => {
    if (!routeResult || !fromCoord || !toCoord) {
      setRiskData([]);
      setSafeRouteLineData([]);
      setSafeRiskData([]);
      setSafeRoute(null);
      setSafeRouteRisk(null);
      return;
    }

    const coords = routeResult.geometry.coordinates as [number, number][];
    if (coords.length < 2) { setRiskData([]); setSafeRouteLineData([]); setSafeRiskData([]); return; }

    let cancelled = false;

    // Always fetch route risk for risk-colored route display
    fetchRouteRisk(coords, dayOffset)
      .then((risk) => {
        if (cancelled) return;
        setRouteRisk(risk);
        // Use routeSegments (colored along the route polyline) if available, fall back to segment geometries
        setRiskData(risk.routeSegments?.features ?? risk.segments.features);
      })
      .catch(() => {
        if (!cancelled) { setRouteRisk(null); setRiskData([]); }
      });

    // Fetch safe alternative route when on a future day
    if (dayOffset > 0) {
      setSafeRouteLoading(true);
      fetchSafeRoute(fromCoord, toCoord, dayOffset)
        .then((result) => {
          if (cancelled) return;
          setSafeRoute(result);
          setSafeRouteLineData(result.geometry.coordinates);
          // Refit map to include both routes
          const safeCoords = result.geometry.coordinates as [number, number][];
          panToFit([fromCoord, toCoord, ...coords, ...safeCoords]);
          // Fetch segment-level risk for the safe route coloring
          if (safeCoords.length >= 2) {
            fetchRouteRisk(safeCoords, dayOffset)
              .then((risk) => {
                if (!cancelled) {
                  setSafeRouteRisk(risk);
                  setSafeRiskData(risk.routeSegments?.features ?? risk.segments.features);
                }
              })
              .catch(() => { if (!cancelled) { setSafeRouteRisk(null); setSafeRiskData([]); } });
          }
        })
        .catch(() => {
          if (!cancelled) { setSafeRoute(null); setSafeRouteLineData([]); setSafeRouteRisk(null); setSafeRiskData([]); }
        })
        .finally(() => { if (!cancelled) setSafeRouteLoading(false); });
    } else {
      setSafeRoute(null);
      setSafeRouteLineData([]);
      setSafeRouteLoading(false);
      setSafeRouteRisk(null);
      setSafeRiskData([]);
      // Refit to just the original route
      panToFit([fromCoord, toCoord, ...coords]);
    }

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dayOffset, routeResult, fromCoord, toCoord, setRiskData, setSafeRouteLineData, setSafeRiskData]);

  const handleRoute = useCallback(async () => {
    if (!fromCoord || !toCoord) return;
    setRouteLoading(true);
    setRouteError(null);
    setRouteResult(null);
    setRouteRisk(null);
    setSafeRoute(null);
    setSafeRouteRisk(null);
    setRiskData([]);
    setSafeRouteLineData([]);
    setSafeRiskData([]);
    try {
      const result = await fetchRoute(fromCoord, toCoord);
      setRouteResult(result);
      const coords = result.geometry.coordinates as [number, number][];
      setRouteData(coords);
      panToFit([fromCoord, toCoord, ...coords]);
      // Route risk + safe route are fetched reactively by the dayOffset useEffect
    } catch (e) {
      setRouteError(e instanceof Error ? e.message : "Route failed");
    } finally {
      setRouteLoading(false);
    }
  }, [fromCoord, toCoord, setRouteData, setRiskData, setSafeRouteLineData, setSafeRiskData, panToFit]);

  const clearRoute = useCallback(() => {
    setRouteResult(null);
    setRouteRisk(null);
    setSafeRoute(null);
    setSafeRouteRisk(null);
    setRouteError(null);
    setFromText("");
    setToText("");
    setFromCoord(null);
    setToCoord(null);
    startMarkerRef.current?.remove();
    endMarkerRef.current?.remove();
    startMarkerRef.current = null;
    endMarkerRef.current = null;
    setRouteData([]);
    setRiskData([]);
    setSafeRouteLineData([]);
    setSafeRiskData([]);
  }, [setRouteData, setRiskData, setSafeRouteLineData, setSafeRiskData]);

  return (
    <div className="mt-4 border-t border-slate-200 pt-4">
      <ToggleButton
        open={open}
        openLabel="Close navigation"
        closedLabel="Get directions"
        onClick={() => { setOpen(!open); if (open) clearRoute(); }}
      />

      {open && (
        <div className="mt-3 space-y-2">
          <SearchInput
            value={fromText}
            onChange={(t) => { setFromText(t); setFromCoord(null); if (routeResult) { setRouteResult(null); setRouteRisk(null); setSafeRoute(null); setSafeRouteRisk(null); setRouteData([]); setRiskData([]); setSafeRouteLineData([]); setSafeRiskData([]); } }}
            onSelect={pickFrom}
            placeholder="From (address or place)"
            hasCoord={fromCoord !== null}
          />
          <SearchInput
            value={toText}
            onChange={(t) => { setToText(t); setToCoord(null); if (routeResult) { setRouteResult(null); setRouteRisk(null); setSafeRoute(null); setSafeRouteRisk(null); setRouteData([]); setRiskData([]); setSafeRouteLineData([]); setSafeRiskData([]); } }}
            onSelect={pickTo}
            placeholder="To (address or place)"
            hasCoord={toCoord !== null}
          />

          <div className="flex gap-2">
            <Button
              label={routeLoading ? "Routing..." : "Route"}
              onClick={handleRoute}
              disabled={!fromCoord || !toCoord || routeLoading}
              fullWidth
            />
            {routeResult && (
              <Button label="Clear" onClick={clearRoute} variant="secondary" />
            )}
          </div>

          {routeError && <div className="text-xs text-red-600">{routeError}</div>}

          {routeResult && (
            <div className="rounded-lg bg-slate-50 p-2 text-xs text-slate-700">
              <div className="font-semibold">
                {(routeResult.distanceM / 1609.34).toFixed(1)} mi · {Math.ceil(routeResult.durationSec / 60)} min
              </div>

              {/* Route legend */}
              <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[10px] text-slate-500">
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2.5 w-4 rounded-sm border-[3px] border-blue-800"
                    style={{ background: "linear-gradient(90deg, #16a34a, #eab308, #f97316, #dc2626)" }} /> Day-of route
                </span>
                {dayOffset > 0 && safeRoute && (
                  <span className="flex items-center gap-1">
                    <span className="inline-block h-2.5 w-4 rounded-sm border-[3px] border-emerald-700 border-dashed"
                      style={{ background: "linear-gradient(90deg, #16a34a, #eab308, #f97316, #dc2626)" }} /> Safer route (Day +{dayOffset})
                  </span>
                )}
                {dayOffset > 0 && !safeRoute && safeRouteLoading && (
                  <span className="text-slate-400 italic">Computing safe route...</span>
                )}
              </div>

              {/* Route comparison + inline street-level alerts */}
              {dayOffset > 0 && routeRisk && safeRoute && (() => {
                const fastRiskPct = Math.round(routeRisk.routeRisk.average * 100);
                const safeRiskPct = Math.round(safeRoute.riskStats.avgRisk * 100);
                const fastDistMi = routeResult.distanceM / 1609.34;
                const safeDistMi = safeRoute.distanceM / 1609.34;
                const distDiffMi = safeDistMi - fastDistMi;
                const riskReduction = fastRiskPct - safeRiskPct;
                const isSameRoute = Math.abs(distDiffMi) < 0.15 && Math.abs(riskReduction) < 3;

                // ── Helpers: extract, group, and render at-risk streets per route ──
                type StreetGroup = { name: string; maxRisk: number; category: string; count: number; allCoords: [number, number][] };

                const extractSegs = (risk: RouteRiskResult) =>
                  risk.segments.features
                    .map((f) => ({
                      name: (f.properties as any).name as string | null,
                      riskScore: (f.properties as any).riskScore as number,
                      riskCategory: (f.properties as any).riskCategory as string,
                      coords: f.geometry.coordinates as [number, number][],
                    }))
                    .filter((s) => s.riskScore >= 0.35)
                    .sort((a, b) => b.riskScore - a.riskScore);

                const groupSegs = (list: ReturnType<typeof extractSegs>): StreetGroup[] => {
                  const grouped: StreetGroup[] = [];
                  for (const s of list) {
                    const label = s.name || "Unnamed road";
                    const existing = grouped.find((g) => g.name === label);
                    if (existing) {
                      existing.count++;
                      existing.allCoords.push(...s.coords);
                      if (s.riskScore > existing.maxRisk) { existing.maxRisk = s.riskScore; existing.category = s.riskCategory; }
                    } else {
                      grouped.push({ name: label, maxRisk: s.riskScore, category: s.riskCategory, count: 1, allCoords: [...s.coords] });
                    }
                  }
                  return grouped.sort((a, b) => b.maxRisk - a.maxRisk);
                };

                const flyToStreet = (allCoords: [number, number][]) => {
                  const map = mapRef.current;
                  if (!map || allCoords.length === 0) return;
                  if (allCoords.length === 1) { map.flyTo({ center: allCoords[0], zoom: 17, duration: 600 }); return; }
                  const bounds = allCoords.reduce((b, c) => b.extend(c), new maplibregl.LngLatBounds(allCoords[0], allCoords[0]));
                  map.fitBounds(bounds, { padding: 80, maxZoom: 17, duration: 600 });
                };

                const renderStreetRow = (g: StreetGroup, i: number, badge: string) => (
                  <button key={i} onClick={() => flyToStreet(g.allCoords)}
                    className="flex w-full items-center justify-between rounded px-1.5 py-0.5 text-left transition-opacity hover:opacity-80 cursor-pointer"
                    style={{ backgroundColor: riskColor(g.category) + "10" }}
                    title={`Click to view ${g.name} on map`}>
                    <div className="flex items-center gap-1 min-w-0">
                      <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: riskColor(g.category) }} />
                      <span className="truncate text-[11px] font-medium text-slate-700">{g.name}</span>
                    </div>
                    <div className="flex shrink-0 items-center gap-1 ml-1">
                      <span className="rounded px-1 py-px text-[9px] font-semibold text-white" style={{ backgroundColor: riskColor(g.category) }}>
                        {Math.round(g.maxRisk * 100)}% · {badge}
                      </span>
                      <span className="text-[9px] text-slate-400">↗</span>
                    </div>
                  </button>
                );

                const renderStreetList = (risk: RouteRiskResult, maxRows: number) => {
                  const segs = extractSegs(risk);
                  if (segs.length === 0) return null;
                  const closures = groupSegs(segs.filter(s => s.riskScore >= 0.75));
                  const hazards = groupSegs(segs.filter(s => s.riskScore >= 0.35 && s.riskScore < 0.75));
                  if (closures.length === 0 && hazards.length === 0) return null;
                  return (
                    <div className="mt-1.5 space-y-1 border-t border-inherit pt-1.5">
                      {closures.length > 0 && (
                        <div>
                          <div className="text-[10px] font-semibold text-red-600 mb-0.5">🚫 Predicted Closures — click to view</div>
                          <div className="max-h-20 space-y-0.5 overflow-auto">
                            {closures.slice(0, maxRows).map((g, i) => renderStreetRow(g, i, "Closure"))}
                            {closures.length > maxRows && <div className="text-[9px] text-slate-400 pl-1">+{closures.length - maxRows} more</div>}
                          </div>
                        </div>
                      )}
                      {hazards.length > 0 && (
                        <div>
                          <div className="text-[10px] font-semibold text-amber-600 mb-0.5">⚠️ Hazardous Conditions — click to view</div>
                          <div className="max-h-20 space-y-0.5 overflow-auto">
                            {hazards.slice(0, maxRows).map((g, i) => renderStreetRow(g, i, "Caution"))}
                            {hazards.length > maxRows && <div className="text-[9px] text-slate-400 pl-1">+{hazards.length - maxRows} more</div>}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                };

                // Counts for summary bullets
                const fastSegs = extractSegs(routeRisk);
                const closureCount = fastSegs.filter(s => s.riskScore >= 0.75).length;
                const hazardCount = fastSegs.filter(s => s.riskScore >= 0.35 && s.riskScore < 0.75).length;

                if (isSameRoute) {
                  return (
                    <div className="mt-1.5 rounded-md border border-amber-300 bg-amber-50 px-2.5 py-2">
                      <div className="flex items-center gap-1.5 font-semibold text-amber-800">
                        <span className="text-sm">⚠️</span> Routes are the same
                      </div>
                      <p className="mt-1 text-slate-600 leading-relaxed">
                        No better alternative found. Be aware:
                      </p>
                      <ul className="mt-1 space-y-0.5 text-slate-600">
                        {closureCount > 0 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-red-500">•</span>
                            <span><strong className="text-red-600">{closureCount} predicted closure{closureCount > 1 ? "s" : ""}</strong> — model forecasts these roads will close</span>
                          </li>
                        )}
                        {hazardCount > 0 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-amber-600">•</span>
                            <span><strong className="text-amber-700">{hazardCount} hazardous segment{hazardCount > 1 ? "s" : ""}</strong> — historically dangerous in storms</span>
                          </li>
                        )}
                        {closureCount === 0 && hazardCount === 0 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-green-600">•</span>
                            <span>Overall conditions look <strong className="text-green-700">manageable</strong> — no closures predicted</span>
                          </li>
                        )}
                        <li className="flex items-start gap-1">
                          <span className="mt-0.5 text-slate-400">•</span>
                          <span>Allow extra travel time for winter conditions</span>
                        </li>
                      </ul>
                      {renderStreetList(routeRisk, 4)}
                    </div>
                  );
                }

                // Routes differ — show comparison with inline street alerts
                return (
                  <div className="mt-1.5 space-y-1.5">
                    {/* Fastest route issues + its street-level alerts */}
                    <div className="rounded-md border-2 border-red-200 bg-red-50 px-2.5 py-2">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 font-semibold text-red-700">
                          <span className="inline-block h-2 w-3 rounded-sm border-[3px] border-blue-800 bg-blue-400" /> Fastest Route Issues
                        </span>
                        <span className="text-slate-500">{fastDistMi.toFixed(1)} mi</span>
                      </div>
                      <ul className="mt-1 space-y-0.5 text-slate-600">
                        {closureCount > 0 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-red-500">✕</span>
                            <span><strong className="text-red-600">{closureCount} predicted closure{closureCount > 1 ? "s" : ""}</strong> — roads likely impassable</span>
                          </li>
                        )}
                        {hazardCount > 0 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-amber-500">✕</span>
                            <span><strong className="text-amber-700">{hazardCount} hazardous segment{hazardCount > 1 ? "s" : ""}</strong> — historically risky in storms</span>
                          </li>
                        )}
                        {routeRisk.routeRisk.max > 0.5 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-red-500">✕</span>
                            <span>Peak risk: <strong className="text-red-600">{Math.round(routeRisk.routeRisk.max * 100)}%</strong></span>
                          </li>
                        )}
                      </ul>
                      {renderStreetList(routeRisk, 3)}
                    </div>

                    {/* Recommended route pros + its street-level alerts */}
                    <div className="rounded-md border-2 border-emerald-200 bg-emerald-50 px-2.5 py-2">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 font-semibold text-emerald-700">
                          <span className="inline-block h-2 w-3 rounded-sm border-[3px] border-emerald-700 bg-emerald-400" /> Recommended Route
                        </span>
                        <span className="text-slate-500">{safeDistMi.toFixed(1)} mi</span>
                      </div>
                      <ul className="mt-1 space-y-0.5 text-slate-600">
                        {riskReduction > 0 && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-emerald-600">✓</span>
                            <span><strong className="text-emerald-700">{riskReduction}% lower overall risk</strong> ({safeRiskPct}% avg)</span>
                          </li>
                        )}
                        {safeRoute.riskStats.closedSegments === 0 ? (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-emerald-600">✓</span>
                            <span><strong className="text-emerald-700">No predicted closures</strong> on this path</span>
                          </li>
                        ) : (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-amber-500">△</span>
                            <span>{safeRoute.riskStats.closedSegments} predicted closure{safeRoute.riskStats.closedSegments > 1 ? "s" : ""} unavoidable</span>
                          </li>
                        )}
                        {safeRoute.riskStats.maxRisk < routeRisk.routeRisk.max && (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-emerald-600">✓</span>
                            <span>Avoids worst segments (peak {Math.round(safeRoute.riskStats.maxRisk * 100)}% vs {Math.round(routeRisk.routeRisk.max * 100)}%)</span>
                          </li>
                        )}
                        {distDiffMi > 0.1 ? (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-amber-500">△</span>
                            <span>+{distDiffMi.toFixed(1)} mi longer ({Math.round((distDiffMi / fastDistMi) * 100)}% more)</span>
                          </li>
                        ) : distDiffMi < -0.1 ? (
                          <li className="flex items-start gap-1">
                            <span className="mt-0.5 text-emerald-600">✓</span>
                            <span>{Math.abs(distDiffMi).toFixed(1)} mi shorter</span>
                          </li>
                        ) : null}
                      </ul>
                      {safeRouteRisk && renderStreetList(safeRouteRisk, 3)}
                    </div>
                  </div>
                );
              })()}

              {/* Day 0 (today) — simple risk summary with inline street alerts */}
              {dayOffset === 0 && routeRisk && (() => {
                type StreetGroup0 = { name: string; maxRisk: number; category: string; count: number; allCoords: [number, number][] };
                const segs0 = routeRisk.segments.features
                  .map((f) => ({
                    name: (f.properties as any).name as string | null,
                    riskScore: (f.properties as any).riskScore as number,
                    riskCategory: (f.properties as any).riskCategory as string,
                    coords: f.geometry.coordinates as [number, number][],
                  }))
                  .filter((s) => s.riskScore >= 0.35)
                  .sort((a, b) => b.riskScore - a.riskScore);

                const grouped0: StreetGroup0[] = [];
                for (const s of segs0) {
                  const label = s.name || "Unnamed road";
                  const existing = grouped0.find((g) => g.name === label);
                  if (existing) { existing.count++; existing.allCoords.push(...s.coords); if (s.riskScore > existing.maxRisk) { existing.maxRisk = s.riskScore; existing.category = s.riskCategory; } }
                  else { grouped0.push({ name: label, maxRisk: s.riskScore, category: s.riskCategory, count: 1, allCoords: [...s.coords] }); }
                }
                grouped0.sort((a, b) => b.maxRisk - a.maxRisk);

                const flyTo0 = (allCoords: [number, number][]) => {
                  const map = mapRef.current;
                  if (!map || allCoords.length === 0) return;
                  if (allCoords.length === 1) { map.flyTo({ center: allCoords[0], zoom: 17, duration: 600 }); return; }
                  const bounds = allCoords.reduce((b, c) => b.extend(c), new maplibregl.LngLatBounds(allCoords[0], allCoords[0]));
                  map.fitBounds(bounds, { padding: 80, maxZoom: 17, duration: 600 });
                };

                return (
                  <div className="mt-1.5 rounded-md border px-2 py-1.5" style={{
                    borderColor: riskColor(routeRisk.routeRisk.category),
                    backgroundColor: riskColor(routeRisk.routeRisk.category) + "10",
                  }}>
                    <div className="flex items-center justify-between">
                      <span className="font-semibold" style={{ color: riskColor(routeRisk.routeRisk.category) }}>
                        Conditions: {routeRisk.routeRisk.category.replace("_", " ").toUpperCase()}
                      </span>
                      <span className="text-slate-500">
                        {routeRisk.matchedSegments} segments analyzed
                      </span>
                    </div>
                    {grouped0.length > 0 && (
                      <div className="mt-1.5 space-y-0.5 border-t pt-1.5" style={{ borderColor: riskColor(routeRisk.routeRisk.category) + "30" }}>
                        <div className="text-[10px] font-semibold text-slate-500 mb-0.5">Segments on your route — click to view</div>
                        <div className="max-h-20 space-y-0.5 overflow-auto">
                          {grouped0.slice(0, 4).map((g, i) => (
                            <button key={i} onClick={() => flyTo0(g.allCoords)}
                              className="flex w-full items-center justify-between rounded px-1.5 py-0.5 text-left transition-opacity hover:opacity-80 cursor-pointer"
                              style={{ backgroundColor: riskColor(g.category) + "10" }}
                              title={`Click to view ${g.name} on map`}>
                              <div className="flex items-center gap-1 min-w-0">
                                <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: riskColor(g.category) }} />
                                <span className="truncate text-[11px] font-medium text-slate-700">{g.name}</span>
                              </div>
                              <div className="flex shrink-0 items-center gap-1 ml-1">
                                <span className="rounded px-1 py-px text-[9px] font-semibold text-white" style={{ backgroundColor: riskColor(g.category) }}>
                                  {Math.round(g.maxRisk * 100)}%
                                </span>
                                <span className="text-[9px] text-slate-400">↗</span>
                              </div>
                            </button>
                          ))}
                          {grouped0.length > 4 && <div className="text-[9px] text-slate-400 pl-1">+{grouped0.length - 4} more</div>}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Per-day risk badges */}
              {routeRisk && routeRisk.riskByDay.length > 1 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {routeRisk.riskByDay.map((d) => (
                    <span
                      key={d.day}
                      className={`rounded px-1 py-0.5 text-[10px] font-medium text-white ${d.day === dayOffset ? "ring-1 ring-white ring-offset-1" : ""}`}
                      style={{ backgroundColor: riskColor(d.category) }}
                      title={`Day +${d.day}: ${d.category} (${Math.round(d.avgRisk * 100)}%)`}
                    >
                      +{d.day}d
                    </span>
                  ))}
                </div>
              )}

              <ol className="mt-1 max-h-32 list-decimal space-y-0.5 overflow-auto pl-4">
                {routeResult.steps
                  .filter((s) => s.instruction)
                  .slice(0, 12)
                  .map((s, i) => (
                    <li key={i}>{s.instruction} ({(s.distanceM / 1609.34).toFixed(2)} mi)</li>
                  ))}
              </ol>

              <button
                onClick={() => {
                  const pts: [number, number][] = [];
                  if (fromCoord) pts.push(fromCoord);
                  if (toCoord) pts.push(toCoord);
                  const routeCoords = routeResult.geometry.coordinates as [number, number][];
                  pts.push(...routeCoords);
                  if (safeRoute) pts.push(...(safeRoute.geometry.coordinates as [number, number][]));
                  panToFit(pts);
                }}
                className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-800 active:bg-slate-100"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                  <path fillRule="evenodd" d="M2 8a6 6 0 1 1 12 0A6 6 0 0 1 2 8Zm6-5a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 8 3Zm0 8a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 8 11Zm-5-3a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5h-1.5A.75.75 0 0 1 3 8Zm8 0a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5h-1.5A.75.75 0 0 1 11 8Zm-3-1a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z" clipRule="evenodd" />
                </svg>
                Center Route
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
