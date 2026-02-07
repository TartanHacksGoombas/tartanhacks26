import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { fetchRoute, GeocodeSuggestion, RouteResult } from "../api";
import SearchInput from "./SearchInput";

const ROUTE_SOURCE = "nav-route-src";
const ROUTE_OUTLINE = "nav-route-outline";
const ROUTE_LINE = "nav-route-line";

/** Extra left padding so the map doesn't place content behind the sidebar. */
const SIDEBAR_PADDING = { top: 80, right: 80, bottom: 80, left: 380 };

type NavigationPanelProps = {
  mapRef: React.RefObject<maplibregl.Map | null>;
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
      paint: { "line-color": "#1e3a8a", "line-width": 10, "line-opacity": 0.6 }
    });
  }
  if (!map.getLayer(ROUTE_LINE)) {
    map.addLayer({
      id: ROUTE_LINE,
      type: "line",
      source: ROUTE_SOURCE,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#3b82f6", "line-width": 6, "line-opacity": 1 }
    });
  }
}

export default function NavigationPanel({ mapRef }: NavigationPanelProps) {
  const [open, setOpen] = useState(false);
  const [fromText, setFromText] = useState("");
  const [toText, setToText] = useState("");
  const [fromCoord, setFromCoord] = useState<[number, number] | null>(null);
  const [toCoord, setToCoord] = useState<[number, number] | null>(null);
  const [routeResult, setRouteResult] = useState<RouteResult | null>(null);
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState<string | null>(null);
  const startMarkerRef = useRef<maplibregl.Marker | null>(null);
  const endMarkerRef = useRef<maplibregl.Marker | null>(null);

  /** Lazily create the route layers when the panel first opens. */
  useEffect(() => {
    if (!open) return;
    const map = mapRef.current;
    if (!map) return;

    const addLayers = () => {
      if (map.isStyleLoaded()) {
        ensureRouteLayers(map);
      }
    };

    // Style might already be loaded, or we wait for it
    addLayers();
    map.on("styledata", addLayers);
    return () => { map.off("styledata", addLayers); };
  }, [open, mapRef]);

  const placeMarker = useCallback(
    (lngLat: [number, number], color: string, ref: React.MutableRefObject<maplibregl.Marker | null>) => {
      const map = mapRef.current;
      if (!map) return;
      if (ref.current) ref.current.remove();
      const el = document.createElement("div");
      el.style.cssText = `width:14px;height:14px;border-radius:50%;background:${color};border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.3)`;
      ref.current = new maplibregl.Marker({ element: el }).setLngLat(lngLat).addTo(map);

      const px = map.project(lngLat);
      if (px.x < SIDEBAR_PADDING.left) {
        map.easeTo({ center: map.unproject([px.x + SIDEBAR_PADDING.left / 2, px.y]), duration: 400 });
      }
    },
    [mapRef]
  );

  const pickFrom = useCallback(
    (s: GeocodeSuggestion) => {
      setFromText(s.displayName.split(",")[0]);
      setFromCoord([s.lng, s.lat]);
      placeMarker([s.lng, s.lat], "#16a34a", startMarkerRef);
    },
    [placeMarker]
  );

  const pickTo = useCallback(
    (s: GeocodeSuggestion) => {
      setToText(s.displayName.split(",")[0]);
      setToCoord([s.lng, s.lat]);
      placeMarker([s.lng, s.lat], "#dc2626", endMarkerRef);
    },
    [placeMarker]
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

  const handleRoute = useCallback(async () => {
    if (!fromCoord || !toCoord) return;
    setRouteLoading(true);
    setRouteError(null);
    setRouteResult(null);
    try {
      const result = await fetchRoute(fromCoord, toCoord);
      setRouteResult(result);
      const coords = result.geometry.coordinates as [number, number][];

      setRouteData(coords);

      const map = mapRef.current;
      if (map && coords.length > 0) {
        const bounds = coords.reduce(
          (b, c) => b.extend(c),
          new maplibregl.LngLatBounds(coords[0], coords[0])
        );
        map.fitBounds(bounds, { padding: SIDEBAR_PADDING });
      }
    } catch (e) {
      setRouteError(e instanceof Error ? e.message : "Route failed");
    } finally {
      setRouteLoading(false);
    }
  }, [fromCoord, toCoord, mapRef, setRouteData]);

  const clearRoute = useCallback(() => {
    setRouteResult(null);
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
  }, [setRouteData]);

  return (
    <div className="mt-4 border-t border-slate-200 pt-4">
      <button
        onClick={() => { setOpen(!open); if (open) clearRoute(); }}
        className="flex w-full items-center justify-between rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700"
      >
        <span>{open ? "Close navigation" : "Get directions"}</span>
        <span className="text-lg leading-none">{open ? "×" : "→"}</span>
      </button>

      {open && (
        <div className="mt-3 space-y-2">
          <SearchInput
            value={fromText}
            onChange={(t) => { setFromText(t); setFromCoord(null); }}
            onSelect={pickFrom}
            placeholder="From (address or place)"
            hasCoord={fromCoord !== null}
          />
          <SearchInput
            value={toText}
            onChange={(t) => { setToText(t); setToCoord(null); }}
            onSelect={pickTo}
            placeholder="To (address or place)"
            hasCoord={toCoord !== null}
          />

          <div className="flex gap-2">
            <button
              onClick={handleRoute}
              disabled={!fromCoord || !toCoord || routeLoading}
              className="flex-1 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-40 hover:bg-blue-700"
            >
              {routeLoading ? "Routing..." : "Route"}
            </button>
            {routeResult && (
              <button onClick={clearRoute} className="rounded-lg bg-slate-200 px-3 py-2 text-sm text-slate-700 hover:bg-slate-300">
                Clear
              </button>
            )}
          </div>

          {routeError && <div className="text-xs text-red-600">{routeError}</div>}

          {routeResult && (
            <div className="rounded-lg bg-slate-50 p-2 text-xs text-slate-700">
              <div className="font-semibold">
                {(routeResult.distanceM / 1609.34).toFixed(1)} mi · {Math.ceil(routeResult.durationSec / 60)} min
              </div>
              <ol className="mt-1 max-h-32 list-decimal space-y-0.5 overflow-auto pl-4">
                {routeResult.steps
                  .filter((s) => s.instruction)
                  .slice(0, 12)
                  .map((s, i) => (
                    <li key={i}>{s.instruction} ({(s.distanceM / 1609.34).toFixed(2)} mi)</li>
                  ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
