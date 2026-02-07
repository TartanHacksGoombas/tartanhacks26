import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const schema = z.object({
  PORT: z.coerce.number().default(4000),
  DATABASE_URL: z.string().min(1, "postgresql://postgres:tartanhacks26@db.hzgcxotuammcbyiagylq.supabase.co:5432/postgres"),
  ADMIN_TOKEN: z.string().default("f"),
  CORS_ORIGIN: z.string().default("http://localhost:5173")
});

const parsed = schema.safeParse(process.env);

if (!parsed.success) {
  console.error("Invalid environment configuration:", parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = parsed.data;
