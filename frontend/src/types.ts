export type SegmentKind = "road" | "all";

export type SegmentLabel = "open" | "low_risk" | "moderate_risk" | "high_risk" | "closed";

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
    kind: "road";
    name: string | null;
    score: number;
    label: SegmentLabel;
    reasons: ScoreReason[];
    updatedAt: string | null;
    closureProbability?: number;
    riskCategory?: string;
  };
};

export type ConditionFeatureCollection = {
  type: "FeatureCollection";
  features: ConditionFeature[];
};

export type WeatherParams = {
  snowfall_cm: number;
  min_temp_c: number;
  max_wind_kmh: number;
  duration_days: number;
};

export type RouteRiskResult = {
  routeRisk: {
    average: number;
    max: number;
    category: string;
  };
  matchedSegments: number;
  riskByDay: { day: number; avgRisk: number; category: string }[];
  segments: ConditionFeatureCollection;
};
