import { Router } from "express";
import { z } from "zod";
import { getSegmentsInBbox, getPrediction, StoredSegment, Prediction } from "../store";
import { SegmentStatusLabel } from "../types";

const conditionsQuerySchema = z.object({
  bbox: z.string().optional(),
  kind: z.enum(["road", "all"]).default("road"),
  day_offset: z.coerce.number().int().min(0).max(6).default(0),
});

export const conditionsRouter = Router();

/** Map WPRDC domi_class to OSM-like highway type for frontend layer filtering. */
function domiClassToHighway(domiClass: string): string {
  const cls = (domiClass ?? "").toLowerCase();
  if (cls.includes("principal") || cls.includes("major arterial")) return "primary";
  if (cls.includes("minor arterial")) return "secondary";
  if (cls.includes("collector")) return "tertiary";
  if (cls.includes("alley")) return "service";
  return "residential";
}

/** Map ML risk_category to frontend SegmentStatusLabel. */
function riskCategoryToLabel(riskCategory: string): SegmentStatusLabel {
  switch (riskCategory) {
    case "very_low": return "open";
    case "low": return "low_risk";
    case "moderate": return "moderate_risk";
    case "high": return "high_risk";
    case "very_high": return "closed";
    default: return "open";
  }
}

function parseBbox(value: string): [number, number, number, number] {
  const parts = value.split(",").map(Number);
  if (parts.length !== 4 || parts.some(isNaN)) {
    throw new Error("bbox must be minLng,minLat,maxLng,maxLat");
  }
  return parts as [number, number, number, number];
}

function buildFeature(seg: StoredSegment, pred: Prediction | undefined) {
  const riskScore = pred?.riskScore ?? 0;
  const riskCategory = pred?.riskCategory ?? "very_low";
  const label = riskCategoryToLabel(riskCategory);
  const score = Math.round((1 - riskScore) * 100);
  const highway = domiClassToHighway(seg.domiClass);

  return {
    type: "Feature" as const,
    id: seg.objectid,
    geometry: seg.geometry,
    properties: {
      kind: "road" as const,
      name: seg.streetname || null,
      highway,
      score,
      label,
      reasons: [] as { code: string; weight: number; detail: string }[],
      lengthM: Math.round(seg.lengthM),
      updatedAt: null as string | null,
      closureProbability: riskScore,
      riskCategory,
    },
  };
}

conditionsRouter.get("/conditions", async (req, res) => {
  try {
    const query = conditionsQuerySchema.parse(req.query);

    if (!query.bbox) {
      return res.status(400).json({ error: "bbox parameter required" });
    }

    const bbox = parseBbox(query.bbox);
    const segments = getSegmentsInBbox(bbox);
    const dayOffset = query.day_offset;

    const features = segments.map((seg) => {
      const pred = getPrediction(dayOffset, seg.objectid);
      return buildFeature(seg, pred);
    });

    return res.json({
      type: "FeatureCollection",
      features,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: "Invalid query parameters", details: error.issues });
    }
    return res.status(400).json({
      error: error instanceof Error ? error.message : "Failed to fetch conditions",
    });
  }
});

conditionsRouter.get("/segment/:id", async (req, res) => {
  const id = Number(req.params.id);
  if (isNaN(id)) return res.status(400).json({ error: "Invalid segment id" });

  const dayOffset = Number(req.query.day_offset ?? 0);
  // Whole Pittsburgh bbox
  const segments = getSegmentsInBbox([-80.2, 40.2, -79.7, 40.7]);
  const seg = segments.find((s) => s.objectid === id);

  if (!seg) return res.status(404).json({ error: "Segment not found" });

  const pred = getPrediction(dayOffset, seg.objectid);
  return res.json(buildFeature(seg, pred));
});
