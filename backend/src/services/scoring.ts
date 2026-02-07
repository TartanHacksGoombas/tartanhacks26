import { PoolClient } from "pg";
import { db } from "../db";
import { ScoreReason, SegmentStatusLabel } from "../types";

type FactorRow = {
  segment_id: number;
  has_closure: boolean;
  winter_penalty: number;
  weather_penalty: number;
  has_recent_plow: boolean;
  slope_pct: number | null;
};

type ComputedStatus = {
  segmentId: number;
  score: number;
  label: SegmentStatusLabel;
  reasons: ScoreReason[];
};

function getLabel(score: number): SegmentStatusLabel {
  if (score >= 70) return "open";
  if (score >= 40) return "caution";
  return "closed";
}

function computeStatus(row: FactorRow): ComputedStatus {
  const closurePenalty = row.has_closure ? 80 : 0;
  const plowPenalty = row.has_recent_plow ? 0 : 15;
  const slopePenalty = (row.slope_pct ?? 0) > 8 ? 10 : 0;

  const totalPenalty =
    closurePenalty + row.winter_penalty + row.weather_penalty + plowPenalty + slopePenalty;

  const score = Math.max(0, Math.min(100, 100 - totalPenalty));
  const label = getLabel(score);

  const reasons: ScoreReason[] = [];

  if (closurePenalty > 0) {
    reasons.push({
      code: "closure_overlap",
      weight: closurePenalty,
      detail: "Active closure intersects this segment."
    });
  }
  if (row.winter_penalty > 0) {
    reasons.push({
      code: "winter_severity",
      weight: row.winter_penalty,
      detail: "Winter condition report impacts this segment."
    });
  }
  if (row.weather_penalty > 0) {
    reasons.push({
      code: "weather_alert",
      weight: row.weather_penalty,
      detail: "Active weather alert intersects this segment."
    });
  }
  if (plowPenalty > 0) {
    reasons.push({
      code: "no_recent_plow",
      weight: plowPenalty,
      detail: "No nearby plow activity in the last 6 hours."
    });
  }
  if (slopePenalty > 0) {
    reasons.push({
      code: "steep_grade",
      weight: slopePenalty,
      detail: "Steep segment can be riskier in snow/ice."
    });
  }

  reasons.sort((a, b) => b.weight - a.weight);

  return {
    segmentId: row.segment_id,
    score,
    label,
    reasons: reasons.slice(0, 3)
  };
}

async function loadFactorRows(client: PoolClient): Promise<FactorRow[]> {
  const query = `
    WITH segment_factors AS (
      SELECT
        s.id AS segment_id,
        s.slope_pct,
        EXISTS (
          SELECT 1
          FROM source_events e
          WHERE e.event_type = 'closure'
            AND (e.starts_at IS NULL OR e.starts_at <= NOW())
            AND (e.ends_at IS NULL OR e.ends_at >= NOW())
            AND e.geom IS NOT NULL
            AND ST_Intersects(e.geom, s.geom)
        ) AS has_closure,
        COALESCE((
          SELECT MAX(
            CASE
              WHEN e.severity >= 4 THEN 40
              WHEN e.severity >= 2 THEN 25
              ELSE 10
            END
          )
          FROM source_events e
          WHERE e.event_type = 'winter_condition'
            AND (e.starts_at IS NULL OR e.starts_at <= NOW())
            AND (e.ends_at IS NULL OR e.ends_at >= NOW())
            AND e.geom IS NOT NULL
            AND ST_Intersects(e.geom, s.geom)
        ), 0) AS winter_penalty,
        COALESCE((
          SELECT MAX(
            CASE
              WHEN e.severity >= 4 THEN 40
              WHEN e.severity >= 2 THEN 30
              ELSE 20
            END
          )
          FROM source_events e
          WHERE e.event_type = 'weather_alert'
            AND (e.starts_at IS NULL OR e.starts_at <= NOW())
            AND (e.ends_at IS NULL OR e.ends_at >= NOW())
            AND e.geom IS NOT NULL
            AND ST_Intersects(e.geom, s.geom)
        ), 0) AS weather_penalty,
        EXISTS (
          SELECT 1
          FROM source_events e
          WHERE e.event_type = 'plow_pass'
            AND e.starts_at >= NOW() - INTERVAL '6 hours'
            AND e.geom IS NOT NULL
            AND ST_DWithin(e.geom::geography, s.geom::geography, 60)
        ) AS has_recent_plow
      FROM segments s
    )
    SELECT segment_id, has_closure, winter_penalty, weather_penalty, has_recent_plow, slope_pct
    FROM segment_factors
  `;

  const { rows } = await client.query<FactorRow>(query);
  return rows;
}

export async function recomputeScores(): Promise<{ updated: number }> {
  const client = await db.connect();

  try {
    await client.query("BEGIN");
    const factors = await loadFactorRows(client);
    const now = new Date().toISOString();

    for (const row of factors) {
      const computed = computeStatus(row);

      await client.query(
        `
          INSERT INTO segment_status_current (segment_id, score, label, reasons, updated_at)
          VALUES ($1, $2, $3, $4::jsonb, $5::timestamptz)
          ON CONFLICT (segment_id)
          DO UPDATE SET
            score = EXCLUDED.score,
            label = EXCLUDED.label,
            reasons = EXCLUDED.reasons,
            updated_at = EXCLUDED.updated_at
        `,
        [
          computed.segmentId,
          computed.score,
          computed.label,
          JSON.stringify(computed.reasons),
          now
        ]
      );

      await client.query(
        `
          INSERT INTO segment_status_history (ts, segment_id, score, label, reasons)
          VALUES ($1::timestamptz, $2, $3, $4, $5::jsonb)
          ON CONFLICT (ts, segment_id) DO NOTHING
        `,
        [now, computed.segmentId, computed.score, computed.label, JSON.stringify(computed.reasons)]
      );
    }

    await client.query("COMMIT");
    return { updated: factors.length };
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
