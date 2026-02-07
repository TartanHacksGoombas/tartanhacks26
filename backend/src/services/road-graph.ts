/**
 * In-memory road network graph built from stored road segments.
 * Supports risk-weighted Dijkstra routing to find safe alternative routes.
 */

import { StoredSegment, getAllSegments, getPrediction } from "../store";

/* ── Types ── */

type NodeId = string; // "lng,lat" snapped to grid

interface Edge {
  to: NodeId;
  segmentId: number;
  lengthM: number;
  coords: [number, number][]; // full polyline for this segment
}

interface GraphNode {
  id: NodeId;
  lng: number;
  lat: number;
  edges: Edge[];
}

/* ── Constants ── */

// Snap precision: 5 decimal places ≈ 1.1m at Pittsburgh's latitude
const SNAP_DECIMALS = 5;
const SNAP_FACTOR = 10 ** SNAP_DECIMALS;

// Risk penalty: how much extra cost high-risk segments incur
const RISK_PENALTY = 8; // riskScore 1.0 → edge cost multiplied by 9×
const CLOSED_PENALTY = 50; // effectively blocked

/* ── Graph storage ── */

const nodes = new Map<NodeId, GraphNode>();
let built = false;

/* ── Helpers ── */

function snap(v: number): number {
  return Math.round(v * SNAP_FACTOR) / SNAP_FACTOR;
}

function nodeId(lng: number, lat: number): NodeId {
  return `${snap(lng)},${snap(lat)}`;
}

function haversineM(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getOrCreateNode(lng: number, lat: number): GraphNode {
  const id = nodeId(lng, lat);
  let node = nodes.get(id);
  if (!node) {
    node = { id, lng: snap(lng), lat: snap(lat), edges: [] };
    nodes.set(id, node);
  }
  return node;
}

/* ── Build graph ── */

export function buildGraph() {
  if (built) return;
  const segments = getAllSegments();
  console.log(`[RoadGraph] Building graph from ${segments.length} segments...`);

  const t0 = Date.now();

  for (const seg of segments) {
    const coords = seg.geometry.coordinates;
    if (coords.length < 2) continue;

    const startCoord = coords[0];
    const endCoord = coords[coords.length - 1];

    const startNode = getOrCreateNode(startCoord[0], startCoord[1]);
    const endNode = getOrCreateNode(endCoord[0], endCoord[1]);

    // Use stored length or compute from geometry
    const lengthM = seg.lengthM > 0 ? seg.lengthM : computeLength(coords);

    // Add bidirectional edges
    startNode.edges.push({
      to: endNode.id,
      segmentId: seg.objectid,
      lengthM,
      coords,
    });
    endNode.edges.push({
      to: startNode.id,
      segmentId: seg.objectid,
      lengthM,
      coords: [...coords].reverse(),
    });
  }

  built = true;
  console.log(
    `[RoadGraph] Built: ${nodes.size} nodes, ${segments.length} edges (bidirectional) in ${Date.now() - t0}ms`
  );
}

function computeLength(coords: [number, number][]): number {
  let total = 0;
  for (let i = 1; i < coords.length; i++) {
    total += haversineM(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0]);
  }
  return total;
}

/* ── Find nearest node ── */

export function findNearestNode(lng: number, lat: number, maxDistM = 500): GraphNode | null {
  const id = nodeId(lng, lat);
  const exact = nodes.get(id);
  if (exact) return exact;

  // Search nearby nodes
  let best: GraphNode | null = null;
  let bestDist = maxDistM;

  // Approximate search radius in degrees
  const degRange = (maxDistM / 111000) * 1.5;
  const minLng = snap(lng - degRange);
  const maxLng = snap(lng + degRange);
  const minLat = snap(lat - degRange);
  const maxLat = snap(lat + degRange);

  for (const node of nodes.values()) {
    if (node.lng < minLng || node.lng > maxLng || node.lat < minLat || node.lat > maxLat) continue;
    const d = haversineM(lat, lng, node.lat, node.lng);
    if (d < bestDist) {
      bestDist = d;
      best = node;
    }
  }

  return best;
}

/* ── Dijkstra with risk weights ── */

export interface SafeRouteResult {
  found: boolean;
  coordinates: [number, number][];
  distanceM: number;
  segmentIds: number[];
  riskStats: {
    avgRisk: number;
    maxRisk: number;
    closedSegments: number;
  };
}

/**
 * Find the lowest-risk route between two points.
 * Edge weight = lengthM * (1 + riskScore * RISK_PENALTY)
 * Segments with riskScore >= 0.75 get extra CLOSED_PENALTY.
 */
export function findSafeRoute(
  originLng: number,
  originLat: number,
  destLng: number,
  destLat: number,
  dayOffset: number
): SafeRouteResult {
  buildGraph();

  const startNode = findNearestNode(originLng, originLat);
  const endNode = findNearestNode(destLng, destLat);

  if (!startNode || !endNode) {
    return { found: false, coordinates: [], distanceM: 0, segmentIds: [], riskStats: { avgRisk: 0, maxRisk: 0, closedSegments: 0 } };
  }

  if (startNode.id === endNode.id) {
    return {
      found: true,
      coordinates: [[startNode.lng, startNode.lat]],
      distanceM: 0,
      segmentIds: [],
      riskStats: { avgRisk: 0, maxRisk: 0, closedSegments: 0 },
    };
  }

  // Dijkstra with binary min-heap
  const dist = new Map<NodeId, number>();
  const prev = new Map<NodeId, { nodeId: NodeId; edge: Edge } | null>();
  const visited = new Set<NodeId>();

  // Binary min-heap for O(log n) push/pop
  type PQEntry = { id: NodeId; cost: number };
  const heap: PQEntry[] = [];

  function heapPush(entry: PQEntry) {
    heap.push(entry);
    let i = heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heap[parent].cost <= heap[i].cost) break;
      [heap[parent], heap[i]] = [heap[i], heap[parent]];
      i = parent;
    }
  }

  function heapPop(): PQEntry | undefined {
    if (heap.length === 0) return undefined;
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      while (true) {
        let smallest = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < heap.length && heap[l].cost < heap[smallest].cost) smallest = l;
        if (r < heap.length && heap[r].cost < heap[smallest].cost) smallest = r;
        if (smallest === i) break;
        [heap[smallest], heap[i]] = [heap[i], heap[smallest]];
        i = smallest;
      }
    }
    return top;
  }

  dist.set(startNode.id, 0);
  prev.set(startNode.id, null);
  heapPush({ id: startNode.id, cost: 0 });

  let found = false;

  while (heap.length > 0) {
    const { id: currentId, cost: currentCost } = heapPop()!;

    if (visited.has(currentId)) continue;
    visited.add(currentId);

    if (currentId === endNode.id) {
      found = true;
      break;
    }

    const currentNode = nodes.get(currentId);
    if (!currentNode) continue;

    for (const edge of currentNode.edges) {
      if (visited.has(edge.to)) continue;

      // Compute risk-weighted cost
      const pred = getPrediction(dayOffset, edge.segmentId);
      const riskScore = pred?.riskScore ?? 0;

      let weight = edge.lengthM;
      if (riskScore >= 0.75) {
        weight *= 1 + CLOSED_PENALTY;
      } else {
        weight *= 1 + riskScore * RISK_PENALTY;
      }

      const newCost = currentCost + weight;
      const oldCost = dist.get(edge.to);

      if (oldCost === undefined || newCost < oldCost) {
        dist.set(edge.to, newCost);
        prev.set(edge.to, { nodeId: currentId, edge });
        heapPush({ id: edge.to, cost: newCost });
      }
    }
  }

  if (!found) {
    return { found: false, coordinates: [], distanceM: 0, segmentIds: [], riskStats: { avgRisk: 0, maxRisk: 0, closedSegments: 0 } };
  }

  // Reconstruct path
  const pathEdges: Edge[] = [];
  let cur = endNode.id;

  while (cur !== startNode.id) {
    const entry = prev.get(cur);
    if (!entry) break;
    pathEdges.unshift(entry.edge);
    cur = entry.nodeId;
  }

  // Build continuous coordinate array
  const coordinates: [number, number][] = [];
  let totalDist = 0;
  const segmentIds: number[] = [];
  let totalRisk = 0;
  let maxRisk = 0;
  let closedCount = 0;

  for (const edge of pathEdges) {
    segmentIds.push(edge.segmentId);
    totalDist += edge.lengthM;

    const pred = getPrediction(dayOffset, edge.segmentId);
    const risk = pred?.riskScore ?? 0;
    totalRisk += risk;
    if (risk > maxRisk) maxRisk = risk;
    if (risk >= 0.75) closedCount++;

    // Append coordinates, skipping duplicate junction points
    for (let i = 0; i < edge.coords.length; i++) {
      if (coordinates.length > 0 && i === 0) {
        // Skip first point if it matches the last appended point
        const last = coordinates[coordinates.length - 1];
        if (last[0] === edge.coords[0][0] && last[1] === edge.coords[0][1]) continue;
      }
      coordinates.push(edge.coords[i]);
    }
  }

  return {
    found: true,
    coordinates,
    distanceM: totalDist,
    segmentIds,
    riskStats: {
      avgRisk: pathEdges.length > 0 ? totalRisk / pathEdges.length : 0,
      maxRisk,
      closedSegments: closedCount,
    },
  };
}

export function isGraphBuilt(): boolean {
  return built;
}

export function graphStats(): { nodes: number; built: boolean } {
  return { nodes: nodes.size, built };
}
