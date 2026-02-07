import { Router } from "express";
import { z } from "zod";
import { findSafeRoute, buildGraph } from "../services/road-graph";

const safeRouteSchema = z.object({
  origin: z.tuple([z.number(), z.number()]), // [lng, lat]
  destination: z.tuple([z.number(), z.number()]), // [lng, lat]
  day_offset: z.number().int().min(0).max(6).default(0),
});

export const safeRouteRouter = Router();

safeRouteRouter.post("/safe-route", async (req, res) => {
  try {
    const body = safeRouteSchema.parse(req.body);
    const { origin, destination, day_offset } = body;

    // Ensure graph is built (lazy, first call may take ~100ms)
    buildGraph();

    const t0 = Date.now();
    const result = findSafeRoute(origin[0], origin[1], destination[0], destination[1], day_offset);
    const elapsed = Date.now() - t0;

    if (!result.found) {
      return res.status(404).json({
        error: "No safe route found between the given points",
        detail: "The origin or destination may be too far from the road network",
      });
    }

    return res.json({
      found: true,
      geometry: {
        type: "LineString",
        coordinates: result.coordinates,
      },
      distanceM: Math.round(result.distanceM),
      segments: result.segmentIds.length,
      riskStats: {
        avgRisk: Math.round(result.riskStats.avgRisk * 1000) / 1000,
        maxRisk: Math.round(result.riskStats.maxRisk * 1000) / 1000,
        closedSegments: result.riskStats.closedSegments,
      },
      elapsedMs: elapsed,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: "Invalid request", details: error.issues });
    }
    console.error("[safe-route] Error:", error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Safe route failed",
    });
  }
});
