/** Shared color constants for road-segment risk levels. */

/** Hex colors for ML risk categories (used by NavigationPanel route-risk badges). */
export const RISK_COLORS: Record<string, string> = {
  very_low: "#16a34a",
  low: "#eab308",
  moderate: "#f97316",
  high: "#dc2626",
  very_high: "#7f1d1d",
};

export const RISK_COLOR_DEFAULT = "#64748b";

export function riskColor(category: string): string {
  return RISK_COLORS[category] ?? RISK_COLOR_DEFAULT;
}

/** Legend items for the sidebar and map (label → Tailwind bg class). */
export const LEGEND_ITEMS = [
  { label: "Open", colorClass: "bg-green-600" },
  { label: "Low risk", colorClass: "bg-yellow-500" },
  { label: "Moderate", colorClass: "bg-orange-500" },
  { label: "High risk", colorClass: "bg-red-600" },
] as const;

/** MapLibre expression that maps the `label` property to a line color. */
export const CONDITION_COLOR_EXPR = [
  "match", ["get", "label"],
  "open", "#16a34a",
  "low_risk", "#eab308",
  "moderate_risk", "#f97316",
  "closed", "#dc2626",
  RISK_COLOR_DEFAULT,
] as const;

/** Marker colors for the navigation A / B pins. */
export const MARKER_COLOR_START = "#16a34a";
export const MARKER_COLOR_END = "#dc2626";
