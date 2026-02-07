import { getAllSegments, getPrediction, StoredSegment } from "../store";

/* ── Types ── */

type NodeId = string; // "lng,lat" rounded to 5dp (~1.1m)

type Edge = {
  to: NodeId;
  segmentId: number;
  lengthM: number;
  streetname: string;
  coords: [number, number][];
};

type RouteStep = {
  instruction: string;
  streetname: string;
  distanceM: number;
  riskScore: number;
  riskCategory: string;
};

export type SafeRouteResult = {
  geometry: { type: "LineString"; coordinates: [number, number][] };
  distanceM: number;
  durationSec: number;
  steps: RouteStep[];
  routeRisk: { average: number; max: number; category: string };
  matchedSegments: number;
  segments: number[];
};

/* ── Graph storage ── */

const adjacency = new Map<NodeId, Edge[]>();
const nodeCoords = new Map<NodeId, [number, number]>(); // nodeId → [lng, lat]
let mainComponent = new Set<NodeId>(); // largest connected component

function makeNodeId(lng: number, lat: number): NodeId {
  // 5 decimal places ≈ 1.1m precision — merges near-miss segment endpoints
  return `${lng.toFixed(5)},${lat.toFixed(5)}`;
}

function parseNodeId(id: NodeId): [number, number] {
  const cached = nodeCoords.get(id);
  if (cached) return cached;
  const [lng, lat] = id.split(",").map(Number);
  return [lng, lat];
}

function addEdge(from: NodeId, to: NodeId, seg: StoredSegment, coords: [number, number][]) {
  let list = adjacency.get(from);
  if (!list) {
    list = [];
    adjacency.set(from, list);
  }
  list.push({
    to,
    segmentId: seg.objectid,
    lengthM: seg.lengthM > 0 ? seg.lengthM : computeLength(coords),
    streetname: seg.streetname,
    coords,
  });
}

function computeLength(coords: [number, number][]): number {
  let total = 0;
  for (let i = 1; i < coords.length; i++) {
    total += haversineM(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0]);
  }
  return total;
}

/* ── Connected components via BFS ── */

function findComponents(): Set<NodeId>[] {
  const visited = new Set<NodeId>();
  const components: Set<NodeId>[] = [];

  for (const node of adjacency.keys()) {
    if (visited.has(node)) continue;

    const component = new Set<NodeId>();
    const queue = [node];
    while (queue.length > 0) {
      const curr = queue.pop()!;
      if (visited.has(curr)) continue;
      visited.add(curr);
      component.add(curr);

      const edges = adjacency.get(curr);
      if (!edges) continue;
      for (const edge of edges) {
        if (!visited.has(edge.to)) queue.push(edge.to);
      }
    }

    components.push(component);
  }

  return components;
}

/* ── Stitch nearby disconnected endpoints ── */

const STITCH_GRID = 0.0002; // ~20m cells for fast neighbor lookup
const STITCH_MAX_M = 15; // max distance to stitch
let stitchSegId = -1; // virtual segment IDs (negative to avoid collisions)

function stitchNearbyNodes(): number {
  // Build a spatial grid of all node positions
  const grid = new Map<string, NodeId[]>();

  for (const [nodeId, coord] of nodeCoords) {
    const key = `${Math.floor(coord[0] / STITCH_GRID)},${Math.floor(coord[1] / STITCH_GRID)}`;
    let list = grid.get(key);
    if (!list) { list = []; grid.set(key, list); }
    list.push(nodeId);
  }

  // For each node, find nearby nodes and add stitch edges if not already connected
  let stitchCount = 0;
  const alreadyConnected = new Set<string>();

  // Pre-compute existing connections
  for (const [from, edges] of adjacency) {
    for (const e of edges) {
      alreadyConnected.add(`${from}|${e.to}`);
    }
  }

  for (const [nodeId, coord] of nodeCoords) {
    const col = Math.floor(coord[0] / STITCH_GRID);
    const row = Math.floor(coord[1] / STITCH_GRID);

    // Check neighboring grid cells
    for (let dc = -1; dc <= 1; dc++) {
      for (let dr = -1; dr <= 1; dr++) {
        const neighbors = grid.get(`${col + dc},${row + dr}`);
        if (!neighbors) continue;

        for (const otherId of neighbors) {
          if (otherId === nodeId) continue;
          if (alreadyConnected.has(`${nodeId}|${otherId}`)) continue;

          const otherCoord = nodeCoords.get(otherId)!;
          const d = haversineM(coord[1], coord[0], otherCoord[1], otherCoord[0]);

          if (d <= STITCH_MAX_M) {
            const sid = stitchSegId--;
            const coords: [number, number][] = [coord, otherCoord];

            // Add bidirectional stitch edge
            let listA = adjacency.get(nodeId);
            if (!listA) { listA = []; adjacency.set(nodeId, listA); }
            listA.push({ to: otherId, segmentId: sid, lengthM: d, streetname: "", coords });

            let listB = adjacency.get(otherId);
            if (!listB) { listB = []; adjacency.set(otherId, listB); }
            listB.push({ to: nodeId, segmentId: sid, lengthM: d, streetname: "", coords: [otherCoord, coord] });

            alreadyConnected.add(`${nodeId}|${otherId}`);
            alreadyConnected.add(`${otherId}|${nodeId}`);
            stitchCount++;
          }
        }
      }
    }
  }

  return stitchCount;
}

/* ── Build graph from store ── */

export function buildGraph() {
  adjacency.clear();
  nodeCoords.clear();
  stitchSegId = -1;

  let edgeCount = 0;

  for (const seg of getAllSegments()) {
    const coords = seg.geometry.coordinates;
    if (coords.length < 2) continue;

    const startCoord = coords[0];
    const endCoord = coords[coords.length - 1];
    const startNode = makeNodeId(startCoord[0], startCoord[1]);
    const endNode = makeNodeId(endCoord[0], endCoord[1]);

    nodeCoords.set(startNode, startCoord);
    nodeCoords.set(endNode, endCoord);

    // Forward direction
    addEdge(startNode, endNode, seg, coords);
    edgeCount++;

    // Reverse direction (unless one-way)
    if (!seg.oneway) {
      addEdge(endNode, startNode, seg, [...coords].reverse());
      edgeCount++;
    }
  }

  // Stitch nearby disconnected endpoints to fix fragmented data
  const stitched = stitchNearbyNodes();
  edgeCount += stitched * 2;

  // Find the largest connected component so routing always succeeds
  const components = findComponents();
  components.sort((a, b) => b.size - a.size);
  mainComponent = components[0] ?? new Set();

  const totalNodes = nodeCoords.size;
  const pct = totalNodes > 0 ? ((mainComponent.size / totalNodes) * 100).toFixed(1) : "0";

  console.log(`  Road graph: ${totalNodes} nodes, ${edgeCount} edges (${stitched} stitched)`);
  console.log(`  Main component: ${mainComponent.size} nodes (${pct}%), ${components.length} total components`);
}

/* ── MinHeap priority queue ── */

class MinHeap {
  private heap: { node: NodeId; cost: number }[] = [];

  push(node: NodeId, cost: number) {
    this.heap.push({ node, cost });
    this.bubbleUp(this.heap.length - 1);
  }

  pop(): { node: NodeId; cost: number } | undefined {
    if (this.heap.length === 0) return undefined;
    const top = this.heap[0];
    const last = this.heap.pop()!;
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.sinkDown(0);
    }
    return top;
  }

  get size() {
    return this.heap.length;
  }

  private bubbleUp(i: number) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.heap[parent].cost <= this.heap[i].cost) break;
      [this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
      i = parent;
    }
  }

  private sinkDown(i: number) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const left = 2 * i + 1;
      const right = 2 * i + 2;
      if (left < n && this.heap[left].cost < this.heap[smallest].cost) smallest = left;
      if (right < n && this.heap[right].cost < this.heap[smallest].cost) smallest = right;
      if (smallest === i) break;
      [this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
      i = smallest;
    }
  }
}

/* ── Snap point to nearest graph node (main component only) ── */

function snapToNode(lng: number, lat: number, maxDistM = 1000): NodeId | null {
  let bestNode: NodeId | null = null;
  let bestDist = maxDistM;

  for (const nodeId of mainComponent) {
    const coord = nodeCoords.get(nodeId)!;
    const d = haversineM(lat, lng, coord[1], coord[0]);
    if (d < bestDist) {
      bestDist = d;
      bestNode = nodeId;
    }
  }

  return bestNode;
}

/* ── A* pathfinding ── */

const RISK_WEIGHT = 10;
const SPEED_MS = 30 * 1000 / 3600; // 30 km/h in m/s

export function findSafestRoute(
  fromLng: number,
  fromLat: number,
  toLng: number,
  toLat: number,
  dayOffset: number
): SafeRouteResult | null {
  const startNode = snapToNode(fromLng, fromLat);
  const endNode = snapToNode(toLng, toLat);
  if (!startNode || !endNode) return null;
  if (startNode === endNode) return null;

  const endCoord = parseNodeId(endNode);

  // A* search
  const gScore = new Map<NodeId, number>();
  const cameFrom = new Map<NodeId, { node: NodeId; edge: Edge }>();
  const visited = new Set<NodeId>();

  gScore.set(startNode, 0);

  const pq = new MinHeap();
  pq.push(startNode, haversineM(parseNodeId(startNode)[1], parseNodeId(startNode)[0], endCoord[1], endCoord[0]));

  while (pq.size > 0) {
    const current = pq.pop()!;
    const currentNode = current.node;

    if (currentNode === endNode) break;
    if (visited.has(currentNode)) continue;
    visited.add(currentNode);

    const edges = adjacency.get(currentNode);
    if (!edges) continue;

    const currentG = gScore.get(currentNode)!;

    for (const edge of edges) {
      if (visited.has(edge.to)) continue;

      const pred = getPrediction(dayOffset, edge.segmentId);
      const risk = pred?.riskScore ?? 0;
      const edgeCost = edge.lengthM * (1 + RISK_WEIGHT * risk);
      const tentativeG = currentG + edgeCost;

      const existingG = gScore.get(edge.to);
      if (existingG !== undefined && tentativeG >= existingG) continue;

      gScore.set(edge.to, tentativeG);
      cameFrom.set(edge.to, { node: currentNode, edge });

      const neighborCoord = parseNodeId(edge.to);
      const h = haversineM(neighborCoord[1], neighborCoord[0], endCoord[1], endCoord[0]);
      pq.push(edge.to, tentativeG + h);
    }
  }

  // Reconstruct path
  if (!cameFrom.has(endNode)) return null;

  const pathEdges: Edge[] = [];
  let current = endNode;
  while (cameFrom.has(current)) {
    const prev = cameFrom.get(current)!;
    pathEdges.unshift(prev.edge);
    current = prev.node;
  }

  // Build geometry, steps, and statistics
  const allCoords: [number, number][] = [];
  const segmentIds: number[] = [];
  let totalDistM = 0;
  let riskSum = 0;
  let maxRisk = 0;
  const steps: RouteStep[] = [];

  // Group consecutive edges by street name for step generation
  let stepStreet = "";
  let stepDistM = 0;
  let stepRiskSum = 0;
  let stepRiskCount = 0;
  let prevBearing = -1;

  for (let i = 0; i < pathEdges.length; i++) {
    const edge = pathEdges[i];
    const pred = getPrediction(dayOffset, edge.segmentId);
    const risk = pred?.riskScore ?? 0;

    // Append coordinates (skip first point of subsequent edges to avoid duplicates)
    if (i === 0) {
      allCoords.push(...edge.coords);
    } else {
      allCoords.push(...edge.coords.slice(1));
    }

    segmentIds.push(edge.segmentId);
    totalDistM += edge.lengthM;
    riskSum += risk;
    if (risk > maxRisk) maxRisk = risk;

    // Compute bearing for turn detection
    const c = edge.coords;
    const bearing = computeBearing(c[0][1], c[0][0], c[c.length - 1][1], c[c.length - 1][0]);

    const sameStreet = edge.streetname === stepStreet && edge.streetname !== "";
    const bearingDelta = prevBearing >= 0 ? angleDiff(prevBearing, bearing) : 0;
    const isTurn = bearingDelta > 30;

    if (i === 0 || (!sameStreet && edge.streetname !== "") || isTurn) {
      // Flush previous step
      if (i > 0 && stepDistM > 0) {
        const avgStepRisk = stepRiskCount > 0 ? stepRiskSum / stepRiskCount : 0;
        steps.push({
          instruction: buildInstruction(prevBearing, bearing, stepStreet),
          streetname: stepStreet,
          distanceM: Math.round(stepDistM),
          riskScore: Math.round(avgStepRisk * 1000) / 1000,
          riskCategory: categorize(avgStepRisk),
        });
      }
      stepStreet = edge.streetname || stepStreet;
      stepDistM = edge.lengthM;
      stepRiskSum = risk;
      stepRiskCount = 1;
    } else {
      stepDistM += edge.lengthM;
      stepRiskSum += risk;
      stepRiskCount++;
    }

    prevBearing = bearing;
  }

  // Flush last step
  if (stepDistM > 0) {
    const avgStepRisk = stepRiskCount > 0 ? stepRiskSum / stepRiskCount : 0;
    steps.push({
      instruction: `Continue on ${stepStreet || "road"}`,
      streetname: stepStreet,
      distanceM: Math.round(stepDistM),
      riskScore: Math.round(avgStepRisk * 1000) / 1000,
      riskCategory: categorize(avgStepRisk),
    });
  }

  const uniqueSegs = new Set(segmentIds);
  const avgRisk = uniqueSegs.size > 0 ? riskSum / pathEdges.length : 0;

  return {
    geometry: { type: "LineString", coordinates: allCoords },
    distanceM: Math.round(totalDistM),
    durationSec: Math.round(totalDistM / SPEED_MS),
    steps,
    routeRisk: {
      average: Math.round(avgRisk * 1000) / 1000,
      max: Math.round(maxRisk * 1000) / 1000,
      category: categorize(avgRisk),
    },
    matchedSegments: uniqueSegs.size,
    segments: Array.from(uniqueSegs),
  };
}

/* ── Helpers ── */

function haversineM(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371000;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function computeBearing(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const lat1R = lat1 * Math.PI / 180;
  const lat2R = lat2 * Math.PI / 180;
  const y = Math.sin(dLng) * Math.cos(lat2R);
  const x = Math.cos(lat1R) * Math.sin(lat2R) - Math.sin(lat1R) * Math.cos(lat2R) * Math.cos(dLng);
  return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;
}

function angleDiff(a: number, b: number): number {
  const d = Math.abs(b - a) % 360;
  return d > 180 ? 360 - d : d;
}

function buildInstruction(prevBearing: number, bearing: number, street: string): string {
  if (prevBearing < 0) return `Head on ${street || "road"}`;
  const diff = ((bearing - prevBearing) + 360) % 360;
  let turn: string;
  if (diff > 330 || diff < 30) turn = "Continue on";
  else if (diff >= 30 && diff < 150) turn = "Turn right onto";
  else if (diff >= 210 && diff < 330) turn = "Turn left onto";
  else turn = "Continue on";
  return `${turn} ${street || "road"}`;
}

function categorize(score: number): string {
  if (score < 0.05) return "very_low";
  if (score < 0.15) return "low";
  if (score < 0.35) return "moderate";
  if (score < 0.55) return "high";
  return "very_high";
}
