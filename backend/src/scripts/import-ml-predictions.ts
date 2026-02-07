/**
 * Import ML model predictions from a GeoJSON file and update road conditions.
 *
 * Usage:  npx tsx src/scripts/import-ml-predictions.ts <path-to-geojson>
 *
 * The GeoJSON should be a FeatureCollection of Point features, each with:
 *   properties.risk_score  (number 0–1, used as closure_probability)
 *
 * Each point is matched to the nearest road segment in the DB (within 100 m).
 */

import fs from "fs";
import path from "path";
import { db } from "../db";

function probabilityToLabel(p: number): "open" | "low_risk" | "moderate_risk" | "closed" {
  if (p < 0.25) return "open";
  if (p < 0.5) return "low_risk";
  if (p < 0.75) return "moderate_risk";
  return "closed";
}

async function main() {
  const filePath = process.argv[2];
  if (!filePath) {
    console.error("Usage: npx tsx src/scripts/import-ml-predictions.ts <path-to-geojson>");
    process.exit(1);
  }

  const absPath = path.resolve(filePath);
  console.log(`Reading ${absPath}...`);
  const raw = fs.readFileSync(absPath, "utf-8");
  const geojson = JSON.parse(raw);

  if (geojson.type !== "FeatureCollection" || !Array.isArray(geojson.features)) {
    console.error("File must be a GeoJSON FeatureCollection.");
    process.exit(1);
  }

  const features: any[] = geojson.features;
  console.log(`Found ${features.length} prediction features.`);

  const client = await db.connect();
  let matched = 0;
  let skipped = 0;

  try {
    await client.query("BEGIN");

    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const geom = feature?.geometry;
      const props = feature?.properties;

      if (!geom || geom.type !== "Point" || !Array.isArray(geom.coordinates) || geom.coordinates.length < 2) {
        skipped++;
        continue;
      }

      const riskScore = typeof props?.risk_score === "number" ? props.risk_score : null;
      if (riskScore === null || riskScore < 0 || riskScore > 1) {
        skipped++;
        continue;
      }

      const [lng, lat] = geom.coordinates;
      const label = probabilityToLabel(riskScore);
      const score = Math.round((1 - riskScore) * 100);
      const streetName = props.streetname ?? props.street_name ?? props.name ?? null;
      const reasons = JSON.stringify([{
        code: "ml_closure_probability",
        weight: Math.round(riskScore * 100),
        detail: `ML risk: ${(riskScore * 100).toFixed(0)}%${streetName ? ` (${streetName})` : ""}${props.risk_category ? ` [${props.risk_category}]` : ""}`
      }]);

      // Find nearest road segment within 100 m
      const { rows } = await client.query(
        `
        SELECT s.id,
               ST_Distance(s.geom::geography, ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography) AS dist_m
        FROM segments s
        WHERE ST_DWithin(s.geom::geography, ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography, 100)
        ORDER BY
          CASE WHEN $3::text IS NOT NULL AND UPPER(s.name) = UPPER($3::text) THEN 0 ELSE 1 END,
          dist_m
        LIMIT 1
        `,
        [lng, lat, streetName]
      );

      if (rows.length === 0) {
        skipped++;
        continue;
      }

      await client.query(
        `
        INSERT INTO segment_status_current (segment_id, score, label, reasons, updated_at, closure_probability)
        VALUES ($1, $2, $3, $4::jsonb, NOW(), $5)
        ON CONFLICT (segment_id)
        DO UPDATE SET
          score = EXCLUDED.score,
          label = EXCLUDED.label,
          reasons = EXCLUDED.reasons,
          updated_at = NOW(),
          closure_probability = EXCLUDED.closure_probability
        `,
        [rows[0].id, score, label, reasons, riskScore]
      );
      matched++;

      if ((i + 1) % 200 === 0) {
        console.log(`  Processed ${i + 1}/${features.length}...`);
      }
    }

    await client.query("COMMIT");
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }

  console.log(`\nDone!`);
  console.log(`  Total predictions: ${features.length}`);
  console.log(`  Matched to road segments: ${matched}`);
  console.log(`  Skipped (no nearby segment or invalid): ${skipped}`);
  console.log(`\nRefresh the frontend to see updated conditions.`);
  process.exit(0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
