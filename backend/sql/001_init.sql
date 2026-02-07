CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS segments (
  id BIGSERIAL PRIMARY KEY,
  osm_id TEXT,
  kind TEXT NOT NULL CHECK (kind IN ('road', 'sidewalk')),
  name TEXT,
  highway TEXT,
  surface TEXT,
  slope_pct REAL DEFAULT 0,
  length_m REAL NOT NULL,
  geom geometry(LineString, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_geom ON segments USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_segments_kind ON segments(kind);

CREATE TABLE IF NOT EXISTS source_events (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  source_event_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  severity SMALLINT DEFAULT 1,
  starts_at TIMESTAMPTZ,
  ends_at TIMESTAMPTZ,
  props JSONB NOT NULL DEFAULT '{}'::jsonb,
  geom geometry(Geometry, 4326),
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (source, source_event_id)
);

CREATE INDEX IF NOT EXISTS idx_source_events_geom ON source_events USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_source_events_type ON source_events(event_type);
CREATE INDEX IF NOT EXISTS idx_source_events_active ON source_events(starts_at, ends_at);

CREATE TABLE IF NOT EXISTS segment_status_current (
  segment_id BIGINT PRIMARY KEY REFERENCES segments(id) ON DELETE CASCADE,
  score SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
  label TEXT NOT NULL CHECK (label IN ('open', 'caution', 'closed')),
  reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS segment_status_history (
  ts TIMESTAMPTZ NOT NULL,
  segment_id BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
  score SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
  label TEXT NOT NULL CHECK (label IN ('open', 'caution', 'closed')),
  reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
  PRIMARY KEY (ts, segment_id)
);
