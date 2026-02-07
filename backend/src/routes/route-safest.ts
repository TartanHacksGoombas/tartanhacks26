import { Router } from "express";
import { z } from "zod";
import { findSafestRoute } from "../services/graph";

const querySchema = z.object({
  from: z.string().regex(/^-?\d+\.?\d*,-?\d+\.?\d*$/, "from must be lng,lat"),
  to: z.string().regex(/^-?\d+\.?\d*,-?\d+\.?\d*$/, "to must be lng,lat"),
  day_offset: z.coerce.number().int().min(0).max(6).default(0),
});

export const routeSafestRouter = Router();

routeSafestRouter.get("/route-safest", (req, res) => {
  try {
    const params = querySchema.parse(req.query);

    const [fromLng, fromLat] = params.from.split(",").map(Number);
    const [toLng, toLat] = params.to.split(",").map(Number);

    const result = findSafestRoute(fromLng, fromLat, toLng, toLat, params.day_offset);

    if (!result) {
      return res.status(404).json({
        error: "No route found. Points may be too far from the road network.",
      });
    }

    return res.json(result);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: "Invalid request", details: error.issues });
    }
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Route failed",
    });
  }
});
