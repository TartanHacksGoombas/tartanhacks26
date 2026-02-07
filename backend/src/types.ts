export type SegmentKind = "road" | "sidewalk";
export type SegmentKindOrAll = SegmentKind | "all";

/** 4-level display: open (green), low_risk (yellow), moderate_risk (orange), closed (red). 'caution' kept for backward compat (treated as moderate_risk). */
export type SegmentStatusLabel = "open" | "low_risk" | "moderate_risk" | "closed" | "caution";

export type ScoreReason = {
  code: string;
  weight: number;
  detail: string;
};
