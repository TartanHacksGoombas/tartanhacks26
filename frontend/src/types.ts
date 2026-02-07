export type SegmentKind = "road" | "sidewalk" | "all";

/** 4-level display: open (green), low_risk (yellow), moderate_risk (orange), closed (red). 'caution' = legacy, shown as orange. */
export type SegmentLabel = "open" | "low_risk" | "moderate_risk" | "closed" | "caution";

export type ScoreReason = {
  code: string;
  weight: number;
  detail: string;
};

export type ConditionFeature = {
  type: "Feature";
  id: number;
  geometry: {
    type: "LineString";
    coordinates: [number, number][];
  };
  properties: {
    kind: "road" | "sidewalk";
    name: string | null;
    score: number;
    label: SegmentLabel;
    reasons: ScoreReason[];
    updatedAt: string | null;
    /** 0–1 from ML model; only present when status is from ML predictions. */
    closureProbability?: number;
  };
};

export type ConditionFeatureCollection = {
  type: "FeatureCollection";
  features: ConditionFeature[];
};
