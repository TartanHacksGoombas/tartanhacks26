/**
 * Ingest active NWS (api.weather.gov) alerts for Pittsburgh zone PAZ021
 * into source_events as weather_alert (and winter_condition for winter-specific).
 *
 * Usage:
 *   BACKEND_URL=http://localhost:4000 ADMIN_TOKEN=your-token npx tsx scripts/ingest-weather-alerts.ts
 *
 * Then: POST /v1/admin/recompute-scores
 */

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:4000";
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? "change-me";
const NWS_ZONE = "PAZ021"; // Pittsburgh / Allegheny County

type NWSAlertFeature = {
  type: "Feature";
  id: string;
  properties: {
    event?: string;
    severity?: string;
    headline?: string;
    description?: string;
    onset?: string;
    expires?: string;
    [k: string]: unknown;
  };
  geometry?: { type: string; coordinates: unknown };
};

type NWSAlertsResponse = {
  type: "FeatureCollection";
  features: NWSAlertFeature[];
};

const winterKeywords = /winter|snow|ice|freez|wind chill|blizzard|frost/i;

async function fetchAlerts(): Promise<NWSAlertsResponse> {
  const url = `https://api.weather.gov/alerts/active?zone=${NWS_ZONE}`;
  const res = await fetch(url, {
    headers: {
      "User-Agent": "SnowRoutePittsburgh/1.0 (hackathon; contact: your-email@example.com)",
      Accept: "application/geo+json"
    }
  });
  if (!res.ok) throw new Error(`NWS API ${res.status}: ${await res.text()}`);
  return (await res.json()) as NWSAlertsResponse;
}

function severityToNumber(severity: string | undefined): number {
  if (!severity) return 1;
  switch (severity.toLowerCase()) {
    case "extreme":
      return 5;
    case "severe":
      return 4;
    case "moderate":
      return 3;
    case "minor":
      return 2;
    default:
      return 1;
  }
}

async function main() {
  console.log("Fetching NWS alerts for", NWS_ZONE, "...");
  const data = await fetchAlerts();
  const features = data.features ?? [];
  if (features.length === 0) {
    console.log("No active alerts.");
    return;
  }

  const events = features
    .filter((f) => f.geometry && f.properties?.event)
    .map((f) => {
      const props = f.properties!;
      const isWinter = winterKeywords.test(String(props.event ?? "") + String(props.headline ?? ""));
      return {
        source_event_id: `nws-${f.id}`,
        event_type: isWinter ? ("winter_condition" as const) : ("weather_alert" as const),
        severity: severityToNumber(props.severity),
        starts_at: props.onset ?? null,
        ends_at: props.expires ?? null,
        props: {
          event: props.event,
          headline: props.headline,
          description: (props.description ?? "").slice(0, 500)
        },
        geom: f.geometry
      };
    });

  if (events.length === 0) {
    console.log("No alerts with geometry; skipping ingest.");
    return;
  }

  const res = await fetch(`${BACKEND_URL}/v1/admin/ingest/weather.gov`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-admin-token": ADMIN_TOKEN
    },
    body: JSON.stringify({ events })
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Ingest ${res.status}: ${text}`);
  }

  const out = (await res.json()) as { ok: boolean; upserted: number };
  console.log("Ingested", out.upserted, "alerts. Run: POST /v1/admin/recompute-scores");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
