-- ML closure probability: 4-level labels (green / yellow / orange / red) and optional probability.
-- Run after 001_init.sql. Safe to run if already applied.

-- Add closure_probability (0–1) from ML model; NULL when status is rule-based.
ALTER TABLE segment_status_current
  ADD COLUMN IF NOT EXISTS closure_probability REAL CHECK (closure_probability IS NULL OR (closure_probability >= 0 AND closure_probability <= 1));

-- Allow 4-level labels: open (green), low_risk (yellow), moderate_risk (orange), closed (red). Keep 'caution' for backward compat.
ALTER TABLE segment_status_current DROP CONSTRAINT IF EXISTS segment_status_current_label_check;
ALTER TABLE segment_status_current
  ADD CONSTRAINT segment_status_current_label_check
  CHECK (label IN ('open', 'low_risk', 'moderate_risk', 'closed', 'caution'));

ALTER TABLE segment_status_history DROP CONSTRAINT IF EXISTS segment_status_history_label_check;
ALTER TABLE segment_status_history
  ADD CONSTRAINT segment_status_history_label_check
  CHECK (label IN ('open', 'low_risk', 'moderate_risk', 'closed', 'caution'));
