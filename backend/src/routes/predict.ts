import { Router } from "express";
import { z } from "zod";
import { runPredictions, getPredictorStatus, WeatherParams } from "../services/predictor";
import { getPredictionDays, getSegmentCount, getPredictionCount } from "../store";

const predictBodySchema = z.object({
  weather: z.object({
    snowfall_cm: z.number().min(0).max(200),
    min_temp_c: z.number().min(-50).max(30),
    max_wind_kmh: z.number().min(0).max(200),
    duration_days: z.number().int().min(1).max(7),
  }),
});

export const predictRouter = Router();

predictRouter.post("/predict", async (req, res) => {
  try {
    const body = predictBodySchema.parse(req.body);
    const weather: WeatherParams = body.weather;

    const result = await runPredictions(weather);

    return res.json({
      ok: true,
      days: result.days,
      segments: result.segments,
      duration_ms: result.durationMs,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: "Invalid request body", details: error.issues });
    }
    const msg = error instanceof Error ? error.message : "Prediction failed";
    const status = msg.includes("already in progress") ? 409 : 500;
    return res.status(status).json({ error: msg });
  }
});

predictRouter.get("/predict/status", (_req, res) => {
  const { running, lastWeather, lastRunAt } = getPredictorStatus();
  const days = getPredictionDays();

  return res.json({
    running,
    lastWeather,
    lastRunAt,
    segmentCount: getSegmentCount(),
    cachedDays: days.map((d) => ({ day: d, predictions: getPredictionCount(d) })),
  });
});
