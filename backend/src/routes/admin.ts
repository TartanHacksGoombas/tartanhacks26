import { Router, Request, Response, NextFunction } from "express";
import { z } from "zod";
import { db } from "../db";
import { config } from "../config";
import { recomputeScores } from "../services/scoring";

const ingestParamsSchema = z.object({
  provider: z.string().min(1)
});

const eventSchema = z.object({
  source_event_id: z.string().min(1),
  event_type: z.enum(["closure", "winter_condition", "weather_alert", "plow_pass"]),
  severity: z.number().int().min(1).max(5).default(1),
  starts_at: z.string().datetime().optional().nullable(),
  ends_at: z.string().datetime().optional().nullable(),
  props: z.record(z.any()).default({}),
  geom: z.record(z.any()).optional().nullable()
});

const ingestBodySchema = z.object({
  events: z.array(eventSchema).min(1)
});

/** ML model output: closure probability per segment (0 = open, 1 = closed). */
const mlPredictionSchema = z.object({
  segment_id: z.number().int().positive(),
  closure_probability: z.number().min(0).max(1)
});
const mlPredictionsBodySchema = z.object({
  predictions: z.array(mlPredictionSchema).min(1)
});

/** Map closure probability to 4-level label for display (green / yellow / orange / red). */
function probabilityToLabel(p: number): "open" | "low_risk" | "moderate_risk" | "closed" {
  if (p < 0.25) return "open";
  if (p < 0.5) return "low_risk";
  if (p < 0.75) return "moderate_risk";
  return "closed";
}

export const adminRouter = Router();

function requireAdminToken(req: Request, res: Response, next: NextFunction) {
  if (!config.ADMIN_TOKEN) {
    return next();
  }

  const token = req.header("x-admin-token");
  if (token !== config.ADMIN_TOKEN) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  return next();
}

adminRouter.use(requireAdminToken);

adminRouter.post("/ingest/:provider", async (req, res) => {
  try {
    const params = ingestParamsSchema.parse(req.params);
    const body = ingestBodySchema.parse(req.body);

    const client = await db.connect();
    let upserted = 0;

    try {
      await client.query("BEGIN");

      for (const event of body.events) {
        const geom = event.geom ? JSON.stringify(event.geom) : null;

        await client.query(
          `
            INSERT INTO source_events (
              source,
              source_event_id,
              event_type,
              severity,
              starts_at,
              ends_at,
              props,
              geom
            )
            VALUES (
              $1,
              $2,
              $3,
              $4,
              $5::timestamptz,
              $6::timestamptz,
              $7::jsonb,
              CASE
                WHEN $8::text IS NULL THEN NULL
                ELSE ST_SetSRID(ST_GeomFromGeoJSON($8), 4326)
              END
            )
            ON CONFLICT (source, source_event_id)
            DO UPDATE SET
              event_type = EXCLUDED.event_type,
              severity = EXCLUDED.severity,
              starts_at = EXCLUDED.starts_at,
              ends_at = EXCLUDED.ends_at,
              props = EXCLUDED.props,
              geom = EXCLUDED.geom,
              ingested_at = NOW()
          `,
          [
            params.provider,
            event.source_event_id,
            event.event_type,
            event.severity,
            event.starts_at ?? null,
            event.ends_at ?? null,
            JSON.stringify(event.props ?? {}),
            geom
          ]
        );

        upserted += 1;
      }

      await client.query("COMMIT");
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }

    return res.json({ ok: true, provider: params.provider, upserted });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid ingest payload",
        details: error.issues
      });
    }
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Ingest failed"
    });
  }
});

adminRouter.post("/ingest/ml-predictions", async (req, res) => {
  try {
    const body = mlPredictionsBodySchema.parse(req.body);
    const client = await db.connect();
    let updated = 0;

    try {
      await client.query("BEGIN");

      for (const { segment_id, closure_probability } of body.predictions) {
        const label = probabilityToLabel(closure_probability);
        const score = Math.round((1 - closure_probability) * 100);
        const reasons = JSON.stringify([
          { code: "ml_closure_probability", weight: Math.round(closure_probability * 100), detail: `ML closure probability: ${(closure_probability * 100).toFixed(0)}%` }
        ]);

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
          [segment_id, score, label, reasons, closure_probability]
        );
        updated += 1;
      }

      await client.query("COMMIT");
    } catch (e) {
      await client.query("ROLLBACK");
      throw e;
    } finally {
      client.release();
    }

    return res.json({ ok: true, updated });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid ML predictions payload",
        details: error.issues
      });
    }
    return res.status(500).json({
      error: error instanceof Error ? error.message : "ML ingest failed"
    });
  }
});

/**
 * POST /v1/admin/ml-predictions-geojson
 *
 * Accepts the ML model's GeoJSON output directly.  Each Feature is a Point
 * with a `risk_score` (0–1).  We spatial-match each point to the nearest
 * road segment in the DB (within 100 m) and update that segment's
 * closure_probability & label.
 *
 * Body: standard GeoJSON FeatureCollection with Point features.
 * Each feature.properties must include `risk_score` (number 0–1).
 */
adminRouter.post("/ml-predictions-geojson", async (req, res) => {
  try {
    const body = req.body;
    if (!body || body.type !== "FeatureCollection" || !Array.isArray(body.features)) {
      return res.status(400).json({ error: "Body must be a GeoJSON FeatureCollection" });
    }

    const features: any[] = body.features;
    const client = await db.connect();
    let matched = 0;
    let skipped = 0;

    try {
      await client.query("BEGIN");

      for (const feature of features) {
        // Validate the feature
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

        // Find the nearest road segment within 100 m.
        // If the prediction includes a street name, prefer a name-matching segment.
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

        const segmentId = rows[0].id;

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
          [segmentId, score, label, reasons, riskScore]
        );
        matched++;
      }

      await client.query("COMMIT");
    } catch (e) {
      await client.query("ROLLBACK");
      throw e;
    } finally {
      client.release();
    }

    return res.json({
      ok: true,
      total: features.length,
      matched,
      skipped,
      message: `Updated ${matched} road segments from ${features.length} ML predictions (${skipped} skipped — no nearby segment or invalid data).`
    });
  } catch (error) {
    return res.status(500).json({
      error: error instanceof Error ? error.message : "ML GeoJSON ingest failed"
    });
  }
});

adminRouter.post("/recompute-scores", async (_req, res) => {
  try {
    const result = await recomputeScores();
    return res.json({ ok: true, ...result });
  } catch (error) {
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Recompute failed"
    });
  }
});
