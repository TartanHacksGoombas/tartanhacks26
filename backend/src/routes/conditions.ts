import { Router } from "express";
import { z } from "zod";
import { db } from "../db";
import { SegmentKindOrAll } from "../types";

const kindSchema = z.enum(["road", "sidewalk", "all"]).default("road");

function parseBbox(value: string | undefined): [number, number, number, number] | null {
  if (!value) return null;
  const parts = value.split(",").map((part) => Number(part));
  if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) {
    throw new Error("bbox must be minLng,minLat,maxLng,maxLat");
  }
  return parts as [number, number, number, number];
}

function parseLngLat(value: string): [number, number] {
  const parts = value.split(",").map((part) => Number(part));
  if (parts.length !== 2 || parts.some((n) => Number.isNaN(n))) {
    throw new Error("point must be lng,lat");
  }
  return parts as [number, number];
}

const conditionsQuerySchema = z.object({
  kind: kindSchema.optional(),
  bbox: z.string().optional()
});

const closuresQuerySchema = z.object({
  bbox: z.string().optional()
});

const timelineQuerySchema = z.object({
  kind: kindSchema.optional(),
  bbox: z.string().optional(),
  at: z.string().datetime().optional()
});

const routeQuerySchema = z.object({
  from: z.string(),
  to: z.string(),
  kind: z.enum(["road", "sidewalk"]).default("road")
});

const paramsSchema = z.object({
  id: z.coerce.number().int().positive()
});

export const conditionsRouter = Router();

function buildConditionsQuery(kind: SegmentKindOrAll, bbox: [number, number, number, number] | null) {
  const values: Array<number | string> = [];
  const clauses: string[] = [];

  if (kind !== "all") {
    values.push(kind);
    clauses.push(`s.kind = $${values.length}`);
  }

  if (bbox) {
    values.push(bbox[0], bbox[1], bbox[2], bbox[3]);
    clauses.push(
      `ST_Intersects(s.geom, ST_MakeEnvelope($${values.length - 3}, $${values.length - 2}, $${values.length - 1}, $${values.length}, 4326))`
    );
  }

  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";

  const text = `
    SELECT
      s.id,
      s.kind,
      s.name,
      s.highway,
      s.length_m AS "lengthM",
      s.slope_pct AS "slopePct",
      ST_AsGeoJSON(s.geom)::json AS geometry,
      COALESCE(ss.score, 100) AS score,
      COALESCE(ss.label, 'open') AS label,
      COALESCE(ss.reasons, '[]'::jsonb) AS reasons,
      ss.updated_at AS "updatedAt",
      ss.closure_probability AS "closureProbability"
    FROM segments s
    LEFT JOIN segment_status_current ss ON ss.segment_id = s.id
    ${where}
    LIMIT 5000
  `;

  return { text, values };
}

conditionsRouter.get("/conditions", async (req, res) => {
  try {
    const query = conditionsQuerySchema.parse(req.query);
    const kind = kindSchema.parse(query.kind ?? "road");
    const bbox = parseBbox(query.bbox);
    const sql = buildConditionsQuery(kind, bbox);
    const { rows } = await db.query(sql.text, sql.values);

    const featureCollection = {
      type: "FeatureCollection",
      features: rows.map((row) => ({
        type: "Feature",
        id: row.id,
        geometry: row.geometry,
        properties: {
          kind: row.kind,
          name: row.name,
          highway: row.highway,
          score: row.score,
          label: row.label,
          reasons: row.reasons,
          lengthM: row.lengthM,
          slopePct: row.slopePct,
          updatedAt: row.updatedAt,
          closureProbability: row.closureProbability ?? undefined
        }
      }))
    };

    return res.json(featureCollection);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid query parameters",
        details: error.issues
      });
    }
    return res.status(400).json({
      error: error instanceof Error ? error.message : "Failed to fetch conditions"
    });
  }
});

conditionsRouter.get("/segment/:id", async (req, res) => {
  try {
    const params = paramsSchema.parse(req.params);

    const { rows } = await db.query(
      `
        SELECT
          s.id,
          s.kind,
          s.name,
          s.highway,
          s.surface,
          s.length_m AS "lengthM",
          s.slope_pct AS "slopePct",
          ST_AsGeoJSON(s.geom)::json AS geometry,
          COALESCE(ss.score, 100) AS score,
          COALESCE(ss.label, 'open') AS label,
          COALESCE(ss.reasons, '[]'::jsonb) AS reasons,
          ss.updated_at AS "updatedAt",
          ss.closure_probability AS "closureProbability"
        FROM segments s
        LEFT JOIN segment_status_current ss ON ss.segment_id = s.id
        WHERE s.id = $1
        LIMIT 1
      `,
      [params.id]
    );

    if (!rows[0]) {
      return res.status(404).json({ error: "Segment not found" });
    }

    return res.json({
      type: "Feature",
      id: rows[0].id,
      geometry: rows[0].geometry,
      properties: {
        kind: rows[0].kind,
        name: rows[0].name,
        highway: rows[0].highway,
        surface: rows[0].surface,
        score: rows[0].score,
        label: rows[0].label,
        reasons: rows[0].reasons,
        lengthM: rows[0].lengthM,
        slopePct: rows[0].slopePct,
        updatedAt: rows[0].updatedAt,
        closureProbability: rows[0].closureProbability ?? undefined
      }
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid segment id",
        details: error.issues
      });
    }
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to fetch segment"
    });
  }
});

conditionsRouter.get("/closures/active", async (req, res) => {
  try {
    const query = closuresQuerySchema.parse(req.query);
    const bbox = parseBbox(query.bbox);

    const values: Array<number> = [];
    const clauses: string[] = [
      "event_type = 'closure'",
      "(starts_at IS NULL OR starts_at <= NOW())",
      "(ends_at IS NULL OR ends_at >= NOW())",
      "geom IS NOT NULL"
    ];

    if (bbox) {
      values.push(bbox[0], bbox[1], bbox[2], bbox[3]);
      clauses.push(
        `ST_Intersects(geom, ST_MakeEnvelope($${values.length - 3}, $${values.length - 2}, $${values.length - 1}, $${values.length}, 4326))`
      );
    }

    const { rows } = await db.query(
      `
        SELECT
          id,
          source,
          source_event_id AS "sourceEventId",
          severity,
          starts_at AS "startsAt",
          ends_at AS "endsAt",
          props,
          ST_AsGeoJSON(geom)::json AS geometry
        FROM source_events
        WHERE ${clauses.join(" AND ")}
        ORDER BY severity DESC, starts_at DESC NULLS LAST
        LIMIT 2000
      `,
      values
    );

    return res.json({
      type: "FeatureCollection",
      features: rows.map((row) => ({
        type: "Feature",
        id: row.id,
        geometry: row.geometry,
        properties: {
          source: row.source,
          sourceEventId: row.sourceEventId,
          severity: row.severity,
          startsAt: row.startsAt,
          endsAt: row.endsAt,
          props: row.props
        }
      }))
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid query parameters",
        details: error.issues
      });
    }
    return res.status(400).json({
      error: error instanceof Error ? error.message : "Failed to fetch closures"
    });
  }
});

conditionsRouter.get("/timeline", async (req, res) => {
  try {
    const query = timelineQuerySchema.parse(req.query);
    const kind = kindSchema.parse(query.kind ?? "road");
    const at = query.at ? new Date(query.at).toISOString() : new Date().toISOString();
    const bbox = parseBbox(query.bbox);

    const values: Array<number | string> = [at];
    const clauses: string[] = [];

    if (kind !== "all") {
      values.push(kind);
      clauses.push(`s.kind = $${values.length}`);
    }
    if (bbox) {
      values.push(bbox[0], bbox[1], bbox[2], bbox[3]);
      clauses.push(
        `ST_Intersects(s.geom, ST_MakeEnvelope($${values.length - 3}, $${values.length - 2}, $${values.length - 1}, $${values.length}, 4326))`
      );
    }

    const where = clauses.length ? `AND ${clauses.join(" AND ")}` : "";

    const { rows } = await db.query(
      `
        WITH snapshot AS (
          SELECT MAX(ts) AS ts
          FROM segment_status_history
          WHERE ts <= $1::timestamptz
        )
        SELECT
          s.id,
          s.kind,
          s.name,
          ST_AsGeoJSON(s.geom)::json AS geometry,
          h.score,
          h.label,
          h.reasons,
          h.ts AS "snapshotTs"
        FROM segment_status_history h
        JOIN snapshot snap ON h.ts = snap.ts
        JOIN segments s ON s.id = h.segment_id
        WHERE 1 = 1
        ${where}
        LIMIT 5000
      `,
      values
    );

    return res.json({
      snapshotTs: rows[0]?.snapshotTs ?? null,
      type: "FeatureCollection",
      features: rows.map((row) => ({
        type: "Feature",
        id: row.id,
        geometry: row.geometry,
        properties: {
          kind: row.kind,
          name: row.name,
          score: row.score,
          label: row.label,
          reasons: row.reasons
        }
      }))
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid query parameters",
        details: error.issues
      });
    }
    return res.status(400).json({
      error: error instanceof Error ? error.message : "Failed to fetch timeline"
    });
  }
});

conditionsRouter.get("/route-safest", async (req, res) => {
  try {
    const query = routeQuerySchema.parse(req.query);
    const from = parseLngLat(query.from);
    const to = parseLngLat(query.to);

    const { rows } = await db.query(
      `
        WITH route_line AS (
          SELECT ST_MakeLine(
            ST_SetSRID(ST_MakePoint($1, $2), 4326),
            ST_SetSRID(ST_MakePoint($3, $4), 4326)
          ) AS line
        ),
        candidates AS (
          SELECT
            s.id,
            s.kind,
            s.name,
            ST_AsGeoJSON(s.geom)::json AS geometry,
            COALESCE(ss.score, 100) AS score,
            COALESCE(ss.label, 'open') AS label,
            COALESCE(ss.reasons, '[]'::jsonb) AS reasons,
            ss.closure_probability AS "closureProbability",
            ST_Distance(s.geom::geography, (SELECT line FROM route_line)::geography) AS "distanceM",
            ST_LineLocatePoint(
              (SELECT line FROM route_line),
              ST_LineInterpolatePoint(s.geom, 0.5)
            ) AS progress
          FROM segments s
          LEFT JOIN segment_status_current ss ON ss.segment_id = s.id
          WHERE s.kind = $5
            AND ST_DWithin(s.geom::geography, (SELECT line FROM route_line)::geography, 300)
        )
        SELECT id, kind, name, geometry, score, label, reasons, "closureProbability", "distanceM", progress
        FROM candidates
        ORDER BY progress ASC, score DESC
        LIMIT 400
      `,
      [from[0], from[1], to[0], to[1], query.kind]
    );

    return res.json({
      mode: "corridor_mvp",
      from: { lng: from[0], lat: from[1] },
      to: { lng: to[0], lat: to[1] },
      type: "FeatureCollection",
      features: rows.map((row) => ({
        type: "Feature",
        id: row.id,
        geometry: row.geometry,
        properties: {
          kind: row.kind,
          name: row.name,
          score: row.score,
          label: row.label,
          reasons: row.reasons,
          closureProbability: row.closureProbability ?? undefined,
          distanceM: Number(row.distanceM),
          progress: Number(row.progress)
        }
      }))
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: "Invalid query parameters",
        details: error.issues
      });
    }
    return res.status(400).json({
      error: error instanceof Error ? error.message : "Failed to fetch route"
    });
  }
});
