import type { SegmentStatusLabel } from "../types";

/** Map ML risk_category to frontend SegmentStatusLabel. */
export function riskCategoryToLabel(riskCategory: string): SegmentStatusLabel {
  switch (riskCategory) {
    case "very_low": return "open";
    case "low": return "low_risk";
    case "moderate": return "moderate_risk";
    case "high":
    case "very_high": return "closed";
    default: return "open";
  }
}
