import dotenv from "dotenv";
import path from "path";
import { z } from "zod";

dotenv.config();

const PROJECT_ROOT = path.resolve(__dirname, "../..");

const schema = z.object({
  PORT: z.coerce.number().default(4000),
  CORS_ORIGIN: z.string().default("http://localhost:5173"),
  DATA_DIR: z.string().default(path.join(PROJECT_ROOT, "data")),
  PIPELINE_DIR: z.string().default(path.join(PROJECT_ROOT, "pipeline")),
});

const parsed = schema.safeParse(process.env);

if (!parsed.success) {
  console.error("Invalid environment configuration:", parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = parsed.data;
