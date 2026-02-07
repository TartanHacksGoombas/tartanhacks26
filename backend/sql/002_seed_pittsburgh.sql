-- Sample road segments in Pittsburgh so the map shows colored lines.
-- Run this in Supabase SQL Editor after 001_init.sql.
-- Coordinates: downtown Pittsburgh (center ~ -79.9959, 40.4406).
-- Safe to run multiple times: removes sample segments first.

DELETE FROM segment_status_current WHERE segment_id IN (SELECT id FROM segments WHERE name IN ('Fifth Ave', 'Forbes Ave', 'Bigelow Blvd', 'Schenley Dr', 'Forbes Ave sidewalk'));
DELETE FROM segments WHERE name IN ('Fifth Ave', 'Forbes Ave', 'Bigelow Blvd', 'Schenley Dr', 'Forbes Ave sidewalk');

INSERT INTO segments (kind, name, highway, length_m, geom)
VALUES
  ('road', 'Fifth Ave', 'primary', 800, ST_GeomFromText('LINESTRING(-80.02 40.44, -80.01 40.441, -79.995 40.442, -79.98 40.44)', 4326)),
  ('road', 'Forbes Ave', 'primary', 600, ST_GeomFromText('LINESTRING(-80.015 40.438, -80.00 40.439, -79.985 40.438)', 4326)),
  ('road', 'Bigelow Blvd', 'secondary', 450, ST_GeomFromText('LINESTRING(-79.96 40.443, -79.955 40.445, -79.945 40.444)', 4326)),
  ('road', 'Schenley Dr', 'tertiary', 350, ST_GeomFromText('LINESTRING(-79.94 40.441, -79.935 40.438, -79.93 40.435)', 4326)),
  ('sidewalk', 'Forbes Ave sidewalk', 'footway', 200, ST_GeomFromText('LINESTRING(-80.005 40.4395, -79.995 40.4398, -79.99 40.4395)', 4326));

-- Set status for each segment (open / caution / closed) so they get colors.
INSERT INTO segment_status_current (segment_id, score, label, reasons)
SELECT id, 85, 'open', '[]'::jsonb FROM segments WHERE name = 'Fifth Ave'
ON CONFLICT (segment_id) DO UPDATE SET score = EXCLUDED.score, label = EXCLUDED.label, reasons = EXCLUDED.reasons, updated_at = NOW();

INSERT INTO segment_status_current (segment_id, score, label, reasons)
SELECT id, 55, 'caution', '[{"detail": "Partial snow coverage"}]'::jsonb FROM segments WHERE name = 'Forbes Ave'
ON CONFLICT (segment_id) DO UPDATE SET score = EXCLUDED.score, label = EXCLUDED.label, reasons = EXCLUDED.reasons, updated_at = NOW();

INSERT INTO segment_status_current (segment_id, score, label, reasons)
SELECT id, 25, 'closed', '[{"detail": "Icy conditions"}]'::jsonb FROM segments WHERE name = 'Bigelow Blvd'
ON CONFLICT (segment_id) DO UPDATE SET score = EXCLUDED.score, label = EXCLUDED.label, reasons = EXCLUDED.reasons, updated_at = NOW();

INSERT INTO segment_status_current (segment_id, score, label, reasons)
SELECT id, 90, 'open', '[]'::jsonb FROM segments WHERE name = 'Schenley Dr'
ON CONFLICT (segment_id) DO UPDATE SET score = EXCLUDED.score, label = EXCLUDED.label, reasons = EXCLUDED.reasons, updated_at = NOW();

INSERT INTO segment_status_current (segment_id, score, label, reasons)
SELECT id, 70, 'caution', '[]'::jsonb FROM segments WHERE name = 'Forbes Ave sidewalk'
ON CONFLICT (segment_id) DO UPDATE SET score = EXCLUDED.score, label = EXCLUDED.label, reasons = EXCLUDED.reasons, updated_at = NOW();
