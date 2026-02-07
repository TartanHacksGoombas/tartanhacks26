import { execFile } from "child_process";
import path from "path";
import { config } from "../config";
import { setPredictions, Prediction } from "../store";

export type WeatherParams = {
  snowfall_cm: number;
  min_temp_c: number;
  max_wind_kmh: number;
  duration_days: number;
};

let running = false;
let lastWeather: WeatherParams | null = null;
let lastRunAt: string | null = null;

export function getPredictorStatus() {
  return { running, lastWeather, lastRunAt };
}

export function runPredictions(weather: WeatherParams): Promise<{ days: number; segments: number; durationMs: number }> {
  if (running) {
    return Promise.reject(new Error("Prediction already in progress"));
  }

  running = true;
  const start = Date.now();

  const scriptPath = path.join(config.PIPELINE_DIR, "predict_batch.py");
  const args = [
    scriptPath,
    "--snowfall_cm", String(weather.snowfall_cm),
    "--min_temp_c", String(weather.min_temp_c),
    "--max_wind_kmh", String(weather.max_wind_kmh),
    "--duration_days", String(weather.duration_days),
    "--max_day_offset", "6",
  ];

  return new Promise((resolve, reject) => {
    execFile("python3", args, {
      maxBuffer: 50 * 1024 * 1024, // 50MB
      timeout: 120_000, // 2 minutes
    }, (error, stdout, stderr) => {
      running = false;

      if (stderr) {
        console.log("[predictor] stderr:", stderr.trim().split("\n").slice(-5).join("\n"));
      }

      if (error) {
        console.error("[predictor] Error:", error.message);
        return reject(new Error(`Prediction failed: ${error.message}`));
      }

      try {
        const result = JSON.parse(stdout);
        const days = Object.keys(result.days);
        let totalSegments = 0;

        for (const dayStr of days) {
          const dayOffset = parseInt(dayStr, 10);
          const preds = result.days[dayStr] as Array<{ objectid: number; risk_score: number; risk_category: string }>;
          const predMap = new Map<number, Prediction>();

          for (const p of preds) {
            predMap.set(p.objectid, {
              riskScore: p.risk_score,
              riskCategory: p.risk_category,
            });
          }

          setPredictions(dayOffset, predMap);
          totalSegments = preds.length;
        }

        lastWeather = weather;
        lastRunAt = new Date().toISOString();

        const durationMs = Date.now() - start;
        console.log(`[predictor] Completed: ${days.length} days, ${totalSegments} segments, ${durationMs}ms`);

        resolve({
          days: days.length,
          segments: totalSegments,
          durationMs,
        });
      } catch (parseError) {
        console.error("[predictor] Failed to parse output:", stdout.slice(0, 200));
        reject(new Error("Failed to parse prediction output"));
      }
    });
  });
}
