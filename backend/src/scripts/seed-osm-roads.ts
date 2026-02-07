/**
 * Fetch real Pittsburgh road geometries from OpenStreetMap (Overpass API)
 * and insert them into the segments table with demo traffic conditions.
 *
 * Usage:  npx tsx src/scripts/seed-osm-roads.ts
 *
 * This gives us hundreds of real road segments so the map lights up with
 * colored conditions — a quick way to verify the visualization pipeline
 * before ML model data is ready.
 */

import { db } from "../db";

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

// Bounding box: Pittsburgh metro (slightly padded)
const BBOX = "40.40,-80.07,40.48,-79.90";

// Overpass query: fetch major roads in Pittsburgh as ways with geometry
const OVERPASS_QUERY = `
[out:json][timeout:60];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]
    (${BBOX});
);
out body geom;
`;

type OverpassWay = {
  type: "way";
  id: number;
  tags: Record<string, string>;
  geometry: { lat: number; lon: number }[];
};

/** Convert an Overpass way to a WKT LINESTRING. */
function toWkt(nodes: { lat: number; lon: number }[]): string {
  const coords = nodes.map((n) => `${n.lon} ${n.lat}`).join(", ");
  return `LINESTRING(${coords})`;
}

/** Approximate length in meters from an array of lat/lon points. */
function approxLength(nodes: { lat: number; lon: number }[]): number {
  let total = 0;
  for (let i = 1; i < nodes.length; i++) {
    const dlat = (nodes[i].lat - nodes[i - 1].lat) * 111320;
    const dlon = (nodes[i].lon - nodes[i - 1].lon) * 111320 * Math.cos((nodes[i].lat * Math.PI) / 180);
    total += Math.sqrt(dlat * dlat + dlon * dlon);
  }
  return total;
}

/** Assign a demo condition label + score + closure_probability based on road type + randomness. */
function demoCondition(highway: string) {
  // Higher-class roads are more likely to be plowed / open
  const openBias: Record<string, number> = {
    motorway: 0.95,
    trunk: 0.90,
    primary: 0.80,
    secondary: 0.65,
    tertiary: 0.55,
    residential: 0.40,
    unclassified: 0.35,
  };

  const bias = openBias[highway] ?? 0.5;
  const roll = Math.random();

  // Weighted random: bias toward open for major roads
  if (roll < bias * 0.6) {
    // Open
    const score = 80 + Math.floor(Math.random() * 20);
    return { score, label: "open" as const, closureProb: Math.random() * 0.1 };
  } else if (roll < bias * 0.6 + 0.2) {
    // Low risk
    const score = 55 + Math.floor(Math.random() * 20);
    return { score, label: "low_risk" as const, closureProb: 0.1 + Math.random() * 0.2 };
  } else if (roll < bias * 0.6 + 0.35) {
    // Moderate risk
    const score = 30 + Math.floor(Math.random() * 20);
    return { score, label: "moderate_risk" as const, closureProb: 0.3 + Math.random() * 0.3 };
  } else {
    // Closed
    const score = 5 + Math.floor(Math.random() * 20);
    return { score, label: "closed" as const, closureProb: 0.6 + Math.random() * 0.4 };
  }
}

async function main() {
  console.log("Fetching Pittsburgh roads from Overpass API...");
  const res = await fetch(OVERPASS_URL, {
    method: "POST",
    body: `data=${encodeURIComponent(OVERPASS_QUERY)}`,
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });

  if (!res.ok) {
    console.error("Overpass request failed:", res.status, await res.text());
    process.exit(1);
  }

  const data = await res.json();
  const ways: OverpassWay[] = (data.elements ?? []).filter(
    (el: any) => el.type === "way" && el.geometry && el.geometry.length >= 2
  );

  console.log(`Got ${ways.length} road segments from OSM.`);

  // Ensure migration 003 has been applied
  console.log("Ensuring closure_probability column exists...");
  await db.query(`
    ALTER TABLE segment_status_current
      ADD COLUMN IF NOT EXISTS closure_probability REAL
      CHECK (closure_probability IS NULL OR (closure_probability >= 0 AND closure_probability <= 1));
  `);
  await db.query(`
    ALTER TABLE segment_status_current DROP CONSTRAINT IF EXISTS segment_status_current_label_check;
    ALTER TABLE segment_status_current
      ADD CONSTRAINT segment_status_current_label_check
      CHECK (label IN ('open', 'low_risk', 'moderate_risk', 'closed', 'caution'));
  `);

  // Clear previous OSM-seeded data
  console.log("Clearing previous OSM-seeded segments...");
  await db.query(`DELETE FROM segment_status_current WHERE segment_id IN (SELECT id FROM segments WHERE osm_id LIKE 'osm_%')`);
  await db.query(`DELETE FROM segments WHERE osm_id LIKE 'osm_%'`);

  // Insert in batches
  const BATCH = 100;
  let inserted = 0;

  for (let i = 0; i < ways.length; i += BATCH) {
    const batch = ways.slice(i, i + BATCH);

    for (const way of batch) {
      const name = way.tags.name ?? null;
      const highway = way.tags.highway ?? "unclassified";
      const surface = way.tags.surface ?? null;
      const lengthM = approxLength(way.geometry);
      const wkt = toWkt(way.geometry);

      try {
        const { rows } = await db.query(
          `INSERT INTO segments (osm_id, kind, name, highway, surface, length_m, geom)
           VALUES ($1, 'road', $2, $3, $4, $5, ST_GeomFromText($6, 4326))
           RETURNING id`,
          [`osm_${way.id}`, name, highway, surface, lengthM, wkt]
        );

        const segmentId = rows[0].id;
        const cond = demoCondition(highway);

        await db.query(
          `INSERT INTO segment_status_current (segment_id, score, label, reasons, closure_probability)
           VALUES ($1, $2, $3, $4::jsonb, $5)
           ON CONFLICT (segment_id) DO UPDATE SET
             score = EXCLUDED.score,
             label = EXCLUDED.label,
             reasons = EXCLUDED.reasons,
             closure_probability = EXCLUDED.closure_probability,
             updated_at = NOW()`,
          [
            segmentId,
            cond.score,
            cond.label,
            JSON.stringify([{ detail: `Demo: ${cond.label} (${highway})` }]),
            Math.round(cond.closureProb * 100) / 100,
          ]
        );

        inserted++;
      } catch (err) {
        // Skip duplicates or geometry errors
        console.warn(`Skipped way ${way.id}:`, (err as Error).message?.slice(0, 80));
      }
    }

    console.log(`  Inserted ${Math.min(i + BATCH, ways.length)}/${ways.length}...`);
  }

  console.log(`\nDone! Inserted ${inserted} road segments with demo conditions.`);
  console.log("Restart the frontend and zoom into Pittsburgh to see colored roads.");
  process.exit(0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
