import { Router } from "express";
import { z } from "zod";
import { getNearestSegments, getPrediction } from "../store";

const routeRiskSchema = z.object({
  coordinates: z.array(z.tuple([z.number(), z.number()])).min(2),
  day_offset: z.number().int().min(0).max(6).default(0),
});

export const routeRiskRouter = Router();

routeRiskRouter.post("/route-risk", async (req, res) => {
  try {
    const body = routeRiskSchema.parse(req.body);
    const { coordinates, day_offset } = body;

    // Sample coordinates every ~50m along the route to avoid over-querying
    const sampledCoords = sampleCoordinates(coordinates, 50);

    // Find nearest road segments for each sampled coordinate
    const matched = new Map<number, { riskScore: number; riskCategory: string; name: string; geometry: any }>();

    for (const [lng, lat] of sampledCoords) {
      const nearest = getNearestSegments(lng, lat, 50);
      if (nearest.length === 0) continue;

      const seg = nearest[0].seg;
      if (matched.has(seg.objectid)) continue;

      const pred = getPrediction(day_offset, seg.objectid);
      matched.set(seg.objectid, {
        riskScore: pred?.riskScore ?? 0,
        riskCategory: pred?.riskCategory ?? "very_low",
        name: seg.streetname,
        geometry: seg.geometry,
      });
    }

    const segments = Array.from(matched.values());
    const riskScores = segments.map((s) => s.riskScore);
    const avgRisk = riskScores.length > 0 ? riskScores.reduce((a, b) => a + b, 0) / riskScores.length : 0;
    const maxRisk = riskScores.length > 0 ? Math.max(...riskScores) : 0;

    // Per-day risk summary (for all cached days)
    const riskByDay: { day: number; avgRisk: number; category: string }[] = [];
    for (let d = 0; d <= 6; d++) {
      const dayScores: number[] = [];
      for (const [objectid] of matched) {
        const pred = getPrediction(d, objectid);
        if (pred) dayScores.push(pred.riskScore);
      }
      if (dayScores.length > 0) {
        const dayAvg = dayScores.reduce((a, b) => a + b, 0) / dayScores.length;
        riskByDay.push({ day: d, avgRisk: Math.round(dayAvg * 1000) / 1000, category: categorize(dayAvg) });
      }
    }

    return res.json({
      routeRisk: {
        average: Math.round(avgRisk * 1000) / 1000,
        max: Math.round(maxRisk * 1000) / 1000,
        category: categorize(avgRisk),
      },
      matchedSegments: segments.length,
      riskByDay,
      segments: {
        type: "FeatureCollection",
        features: segments.map((s, i) => ({
          type: "Feature",
          id: i,
          geometry: s.geometry,
          properties: {
            name: s.name,
            riskScore: s.riskScore,
            riskCategory: s.riskCategory,
            label: riskCatToLabel(s.riskCategory),
          },
        })),
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: "Invalid request", details: error.issues });
    }
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Route risk failed",
    });
  }
});

function categorize(score: number): string {
  if (score < 0.05) return "very_low";
  if (score < 0.15) return "low";
  if (score < 0.35) return "moderate";
  if (score < 0.55) return "high";
  return "very_high";
}

function riskCatToLabel(cat: string): string {
  switch (cat) {
    case "very_low": return "open";
    case "low": return "low_risk";
    case "moderate": return "moderate_risk";
    case "high": return "high_risk";
    default: return "closed";
  }
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

/** Sample coordinates along a polyline at approximately every `intervalM` meters. */
function sampleCoordinates(coords: [number, number][], intervalM: number): [number, number][] {
  if (coords.length <= 2) return coords;

  const sampled: [number, number][] = [coords[0]];
  let accDist = 0;

  for (let i = 1; i < coords.length; i++) {
    const [lng1, lat1] = coords[i - 1];
    const [lng2, lat2] = coords[i];
    const d = haversineM(lat1, lng1, lat2, lng2);
    accDist += d;

    if (accDist >= intervalM) {
      sampled.push(coords[i]);
      accDist = 0;
    }
  }

  // Always include last point
  const last = coords[coords.length - 1];
  if (sampled[sampled.length - 1] !== last) {
    sampled.push(last);
  }

  return sampled;
}
