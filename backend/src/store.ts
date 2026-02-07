import * as fs from "fs";
import * as path from "path";

/* ── Types ── */

export type StoredSegment = {
  objectid: number;
  geometry: { type: "LineString"; coordinates: [number, number][] };
  streetname: string;
  domiClass: string;
  roadwidth: number;
  lengthM: number;
  bbox: [number, number, number, number]; // [minLng, minLat, maxLng, maxLat]
  midLat: number;
  midLng: number;
};

export type Prediction = {
  riskScore: number;
  riskCategory: string;
};

/* ── Spatial grid ── */

const GRID_SIZE = 0.002; // ~200m at Pittsburgh's latitude

function gridKey(lng: number, lat: number): string {
  const col = Math.floor(lng / GRID_SIZE);
  const row = Math.floor(lat / GRID_SIZE);
  return `${col},${row}`;
}

/* ── Store ── */

const segments = new Map<number, StoredSegment>();
const spatialGrid = new Map<string, number[]>(); // gridKey → objectid[]
const predictionCache = new Map<number, Map<number, Prediction>>(); // dayOffset → objectid → Prediction

let loaded = false;

function computeBbox(coords: [number, number][]): [number, number, number, number] {
  let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
  for (const [lng, lat] of coords) {
    if (lng < minLng) minLng = lng;
    if (lat < minLat) minLat = lat;
    if (lng > maxLng) maxLng = lng;
    if (lat > maxLat) maxLat = lat;
  }
  return [minLng, minLat, maxLng, maxLat];
}

function computeMidpoint(coords: [number, number][]): [number, number] {
  const midIdx = Math.floor(coords.length / 2);
  return [coords[midIdx][1], coords[midIdx][0]]; // [lat, lng]
}

function indexSegment(seg: StoredSegment) {
  // Add to all grid cells this segment touches
  const [minLng, minLat, maxLng, maxLat] = seg.bbox;
  const startCol = Math.floor(minLng / GRID_SIZE);
  const endCol = Math.floor(maxLng / GRID_SIZE);
  const startRow = Math.floor(minLat / GRID_SIZE);
  const endRow = Math.floor(maxLat / GRID_SIZE);

  for (let col = startCol; col <= endCol; col++) {
    for (let row = startRow; row <= endRow; row++) {
      const key = `${col},${row}`;
      let list = spatialGrid.get(key);
      if (!list) {
        list = [];
        spatialGrid.set(key, list);
      }
      list.push(seg.objectid);
    }
  }
}

export async function loadStore(dataDir: string) {
  if (loaded) return;

  const geojsonPath = path.join(dataDir, "centerlines_pgh.geojson");
  console.log(`Loading centerlines from ${geojsonPath}...`);

  const raw = fs.readFileSync(geojsonPath, "utf-8");
  const geojson = JSON.parse(raw);

  let count = 0;
  for (const feature of geojson.features) {
    if (!feature.geometry || feature.geometry.type !== "LineString") continue;
    const props = feature.properties ?? {};
    const objectid = props.objectid;
    if (objectid == null || segments.has(objectid)) continue;

    const coords: [number, number][] = feature.geometry.coordinates;
    if (coords.length < 2) continue;

    const bbox = computeBbox(coords);
    const [midLat, midLng] = computeMidpoint(coords);

    const seg: StoredSegment = {
      objectid,
      geometry: { type: "LineString", coordinates: coords },
      streetname: props.streetname ?? "",
      domiClass: props.domi_class ?? "Local",
      roadwidth: props.roadwidth ?? 0,
      lengthM: props.measurlgth ?? props.Shape__Length ?? 0,
      bbox,
      midLat,
      midLng,
    };

    segments.set(objectid, seg);
    indexSegment(seg);
    count++;
  }

  console.log(`  Loaded ${count} segments into memory`);
  console.log(`  Spatial grid: ${spatialGrid.size} cells`);

  // Load fallback predictions (day inferred from metadata)
  const predictionsPath = path.join(dataDir, "predictions_latest.json");
  if (fs.existsSync(predictionsPath)) {
    console.log(`Loading fallback predictions from ${predictionsPath}...`);
    const predRaw = fs.readFileSync(predictionsPath, "utf-8");
    const predJson = JSON.parse(predRaw);
    const dayOffset =
      Number(predJson?.metadata?.temporal_features?.day_offset ?? predJson?.metadata?.day_offset ?? 0) || 0;
    const dayMap = new Map<number, Prediction>();

    for (const feature of predJson.features ?? []) {
      const p = feature.properties;
      if (!p?.objectid) continue;
      dayMap.set(p.objectid, {
        riskScore: p.risk_score ?? 0,
        riskCategory: p.risk_category ?? "very_low",
      });
    }

    predictionCache.set(dayOffset, dayMap);
    console.log(`  Loaded ${dayMap.size} fallback predictions for day ${dayOffset}`);
  }

  loaded = true;
}

export function getSegmentsInBbox(bbox: [number, number, number, number]): StoredSegment[] {
  const [minLng, minLat, maxLng, maxLat] = bbox;
  const seen = new Set<number>();
  const result: StoredSegment[] = [];

  const startCol = Math.floor(minLng / GRID_SIZE);
  const endCol = Math.floor(maxLng / GRID_SIZE);
  const startRow = Math.floor(minLat / GRID_SIZE);
  const endRow = Math.floor(maxLat / GRID_SIZE);

  for (let col = startCol; col <= endCol; col++) {
    for (let row = startRow; row <= endRow; row++) {
      const key = `${col},${row}`;
      const ids = spatialGrid.get(key);
      if (!ids) continue;
      for (const id of ids) {
        if (seen.has(id)) continue;
        seen.add(id);
        const seg = segments.get(id);
        if (!seg) continue;
        // Check actual bbox overlap
        if (seg.bbox[2] < minLng || seg.bbox[0] > maxLng ||
            seg.bbox[3] < minLat || seg.bbox[1] > maxLat) continue;
        result.push(seg);
      }
    }
  }

  return result;
}

export function getPrediction(dayOffset: number, objectid: number): Prediction | undefined {
  return predictionCache.get(dayOffset)?.get(objectid);
}

export function setPredictions(dayOffset: number, predictions: Map<number, Prediction>) {
  predictionCache.set(dayOffset, predictions);
}

export function clearPredictions() {
  predictionCache.clear();
}

export function getPredictionDays(): number[] {
  return Array.from(predictionCache.keys()).sort((a, b) => a - b);
}

export function getSegmentCount(): number {
  return segments.size;
}

export function getPredictionCount(dayOffset: number): number {
  return predictionCache.get(dayOffset)?.size ?? 0;
}

export function getAllSegments(): StoredSegment[] {
  return Array.from(segments.values());
}

/** Haversine distance in meters. */
function haversineM(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371000;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Find nearest segments to a coordinate within a radius. */
export function getNearestSegments(lng: number, lat: number, radiusM: number): { seg: StoredSegment; distanceM: number }[] {
  // Convert radius to approx degrees for grid search
  const degBuffer = (radiusM / 111000) * 1.5;
  const startCol = Math.floor((lng - degBuffer) / GRID_SIZE);
  const endCol = Math.floor((lng + degBuffer) / GRID_SIZE);
  const startRow = Math.floor((lat - degBuffer) / GRID_SIZE);
  const endRow = Math.floor((lat + degBuffer) / GRID_SIZE);

  const seen = new Set<number>();
  const results: { seg: StoredSegment; distanceM: number }[] = [];

  for (let col = startCol; col <= endCol; col++) {
    for (let row = startRow; row <= endRow; row++) {
      const key = `${col},${row}`;
      const ids = spatialGrid.get(key);
      if (!ids) continue;
      for (const id of ids) {
        if (seen.has(id)) continue;
        seen.add(id);
        const seg = segments.get(id);
        if (!seg) continue;

        // Find closest point on the segment's polyline to the query point
        let minDist = Infinity;
        for (const [sLng, sLat] of seg.geometry.coordinates) {
          const d = haversineM(lat, lng, sLat, sLng);
          if (d < minDist) minDist = d;
        }

        if (minDist <= radiusM) {
          results.push({ seg, distanceM: minDist });
        }
      }
    }
  }

  return results.sort((a, b) => a.distanceM - b.distanceM);
}
