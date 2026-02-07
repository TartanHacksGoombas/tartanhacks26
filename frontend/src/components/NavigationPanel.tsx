import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { fetchRoute, fetchRouteRisk, reverseGeocode, GeocodeSuggestion, RouteResult } from "../utils/api";
import { RouteRiskResult } from "../types";
import SearchInput from "./SearchInput";
import Button from "./Button";
import ToggleButton from "./ToggleButton";

const ROUTE_SOURCE = "nav-route-src";
const ROUTE_OUTLINE = "nav-route-outline";
const ROUTE_LINE = "nav-route-line";

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
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState<string | null>(null);
  const startMarkerRef = useRef<maplibregl.Marker | null>(null);
  const endMarkerRef = useRef<maplibregl.Marker | null>(null);

  // Keep latest padding in a ref so imperative pan calls always use fresh values
  const paddingRef = useRef(mapPadding);
  paddingRef.current = mapPadding;

  const geolocatedRef = useRef(false);

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

      const pad = paddingRef.current;
      const EXTRA = 60;
      const fitPad = {
        top: pad.top + EXTRA,
        right: pad.right + EXTRA,
        bottom: pad.bottom + EXTRA,
        left: pad.left + EXTRA,
      };

      const mapW = map.getContainer().clientWidth;
      const mapH = map.getContainer().clientHeight;
      const usableW = mapW - fitPad.left - fitPad.right;
      const usableH = mapH - fitPad.top - fitPad.bottom;

      // If padding eats up the viewport, fall back to simple padding
      const safePad = (usableW < 100 || usableH < 100)
        ? { top: 50, right: 50, bottom: 50, left: 50 }
        : fitPad;

      if (points.length === 1) {
        map.easeTo({ center: points[0], padding: safePad, zoom: 15, duration: 400 });
        return;
      }

      const bounds = points.reduce(
        (b, c) => b.extend(c),
        new maplibregl.LngLatBounds(points[0], points[0])
      );
      map.fitBounds(bounds, { padding: safePad, maxZoom: 16, duration: 500 });
    },
    [mapRef]
  );

  const pickFrom = useCallback(
    (s: GeocodeSuggestion) => {
      const coord: [number, number] = [s.lng, s.lat];
      setFromText(s.displayName.split(",")[0]);
      setFromCoord(coord);
      placeMarker(coord, "#16a34a", "A", startMarkerRef);
      // Pan: use the coord we just got + whatever toCoord is currently set
      const pts: [number, number][] = [coord];
      if (endMarkerRef.current) {
        const ll = endMarkerRef.current.getLngLat();
        pts.push([ll.lng, ll.lat]);
      }
      panToFit(pts);
    },
    [placeMarker, panToFit]
  );

  const pickTo = useCallback(
    (s: GeocodeSuggestion) => {
      const coord: [number, number] = [s.lng, s.lat];
      setToText(s.displayName.split(",")[0]);
      setToCoord(coord);
      placeMarker(coord, "#dc2626", "B", endMarkerRef);
      // Pan: use the coord we just got + whatever fromCoord is currently set
      const pts: [number, number][] = [coord];
      if (startMarkerRef.current) {
        const ll = startMarkerRef.current.getLngLat();
        pts.push([ll.lng, ll.lat]);
      }
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

  const handleRoute = useCallback(async () => {
    if (!fromCoord || !toCoord) return;
    setRouteLoading(true);
    setRouteError(null);
    setRouteResult(null);
    setRouteRisk(null);
    try {
      const result = await fetchRoute(fromCoord, toCoord);
      setRouteResult(result);
      const coords = result.geometry.coordinates as [number, number][];
      setRouteData(coords);
      panToFit([fromCoord, toCoord, ...coords]);

      // Fetch route risk in background
      fetchRouteRisk(coords, dayOffset).then(setRouteRisk).catch(() => {});
    } catch (e) {
      setRouteError(e instanceof Error ? e.message : "Route failed");
    } finally {
      setRouteLoading(false);
    }
  }, [fromCoord, toCoord, dayOffset, setRouteData, panToFit]);

  const clearRoute = useCallback(() => {
    setRouteResult(null);
    setRouteRisk(null);
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
            onChange={(t) => { setFromText(t); setFromCoord(null); if (routeResult) { setRouteResult(null); setRouteData([]); } }}
            onSelect={pickFrom}
            placeholder="From (address or place)"
            hasCoord={fromCoord !== null}
          />
          <SearchInput
            value={toText}
            onChange={(t) => { setToText(t); setToCoord(null); if (routeResult) { setRouteResult(null); setRouteData([]); } }}
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

              {routeRisk && (
                <div className="mt-1.5 rounded-md border px-2 py-1.5" style={{
                  borderColor: riskColor(routeRisk.routeRisk.category),
                  backgroundColor: riskColor(routeRisk.routeRisk.category) + "10",
                }}>
                  <div className="flex items-center justify-between">
                    <span className="font-semibold" style={{ color: riskColor(routeRisk.routeRisk.category) }}>
                      Road Closure Risk: {routeRisk.routeRisk.category.replace("_", " ").toUpperCase()}
                    </span>
                    <span className="text-slate-500">
                      {Math.round(routeRisk.routeRisk.average * 100)}% avg · {routeRisk.matchedSegments} segments
                    </span>
                  </div>
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
            </div>
          )}
        </div>
      )}
    </div>
  );
}
