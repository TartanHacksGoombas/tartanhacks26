import cors from "cors";
import express from "express";
import { config } from "./config";
import { loadStore } from "./store";
import { buildGraph } from "./services/graph";
import { conditionsRouter } from "./routes/conditions";
import { predictRouter } from "./routes/predict";
import { routeRiskRouter } from "./routes/route-risk";
import { routeSafestRouter } from "./routes/route-safest";
import { mapsProxyRouter } from "./routes/maps-proxy";

const app = express();

app.use(cors({ origin: config.CORS_ORIGIN }));
app.use(express.json({ limit: "2mb" }));

app.get("/healthz", (_req, res) => {
  res.json({ ok: true, service: "winterwise-backend" });
});

app.use("/v1", conditionsRouter);
app.use("/v1", predictRouter);
app.use("/v1", routeRiskRouter);
app.use("/v1", routeSafestRouter);
app.use("/v1/maps", mapsProxyRouter);

app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error("Unhandled error:", error);
  res.status(500).json({ error: "Internal server error" });
});

async function start() {
  await loadStore(config.DATA_DIR);
  console.log("Building road network graph...");
  buildGraph();
  app.listen(config.PORT, () => {
    console.log(`API listening on http://localhost:${config.PORT}`);
  });
}

start().catch((error) => {
  console.error("Failed to start API:", error);
  process.exit(1);
});
