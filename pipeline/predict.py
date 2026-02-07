"""
Generate per-segment road closure risk predictions for a given weather scenario.

Two input modes:
    1. NWS forecast: Parse data/weather_forecast_pgh.json from collect_weather_realtime.py
    2. CLI args: --snowfall_cm 10 --min_temp_c -5 --max_wind_kmh 30 --duration_days 2

Scoring strategy:
    - Prefer the LambdaRank model (ranker_lgbm.txt) which includes weather
      features and interaction terms directly.
    - Fall back to the regression model (model_lgbm.txt) if the ranker is not
      available.
    - Raw ranker scores are converted to [0,1] via a sigmoid-style calibration
      (no artificial min-max normalization).
    - Output is deduplicated to one row per objectid.

Outputs:
    - data/predictions_latest.csv
    - data/predictions_latest.json (GeoJSON-compatible)

Requires: pip install pandas numpy lightgbm
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Input files — prefer ranker, fall back to regression
RANKER_MODEL_FILE = os.path.join(DATA_DIR, "ranker_lgbm.txt")
RANKER_META_FILE = os.path.join(DATA_DIR, "ranker_metadata.json")
REGRESSION_MODEL_FILE = os.path.join(DATA_DIR, "model_lgbm.txt")
REGRESSION_META_FILE = os.path.join(DATA_DIR, "model_metadata.json")
DATASET_CSV = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")
FORECAST_JSON = os.path.join(DATA_DIR, "weather_forecast_pgh.json")

# Output files
OUTPUT_CSV = os.path.join(DATA_DIR, "predictions_latest.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "predictions_latest.json")

# Risk categories (applied to calibrated [0,1] scores)
RISK_THRESHOLDS = [
    (0.0, 0.05, "very_low"),
    (0.05, 0.15, "low"),
    (0.15, 0.35, "moderate"),
    (0.35, 0.55, "high"),
    (0.55, 1.01, "very_high"),
]


def categorize_risk(score):
    """Map a 0-1 risk score to a category label."""
    for low, high, label in RISK_THRESHOLDS:
        if low <= score < high:
            return label
    return "very_high"


def parse_nws_forecast(forecast_path):
    """Extract weather features from NWS forecast JSON."""
    with open(forecast_path, "r") as f:
        forecast = json.load(f)

    periods = []
    if "properties" in forecast and "periods" in forecast["properties"]:
        periods = forecast["properties"]["periods"]
    elif "periods" in forecast:
        periods = forecast["periods"]
    elif isinstance(forecast, list):
        periods = forecast

    if not periods:
        print("  Warning: No forecast periods found, using defaults")
        return None

    temps, winds = [], []
    snow_keywords = ["snow", "blizzard", "winter storm", "ice"]
    has_snow = False

    for period in periods[:6]:
        temp = period.get("temperature")
        if temp is not None:
            unit = period.get("temperatureUnit", "F")
            if unit == "F":
                temp = (temp - 32) * 5 / 9
            temps.append(temp)

        wind_str = period.get("windSpeed", "")
        if isinstance(wind_str, str):
            import re
            nums = re.findall(r"\d+", wind_str)
            if nums:
                max_wind = max(int(n) for n in nums) * 1.60934
                winds.append(max_wind)
        elif isinstance(wind_str, (int, float)):
            winds.append(wind_str * 1.60934)

        detail = period.get("detailedForecast", "") or period.get("shortForecast", "")
        if any(kw in detail.lower() for kw in snow_keywords):
            has_snow = True

    if not has_snow:
        print("  Warning: No snow in forecast. Predictions may show very low risk.")

    snowfall_cm = 5.0 if has_snow else 0.5
    for period in periods[:6]:
        detail = (period.get("detailedForecast", "") or "").lower()
        import re
        inches_match = re.findall(r"(\d+)\s*(?:to\s*(\d+))?\s*inch", detail)
        if inches_match:
            max_inches = max(int(m[1]) if m[1] else int(m[0]) for m in inches_match)
            snowfall_cm = max(snowfall_cm, max_inches * 2.54)

    return {
        "total_snowfall_cm": round(snowfall_cm, 1),
        "max_daily_snowfall_cm": round(snowfall_cm, 1),
        "min_temp_c": round(min(temps), 1) if temps else -5.0,
        "max_wind_kmh": round(max(winds), 1) if winds else 20.0,
        "duration_days": 1,
        "precip_total_mm": round(snowfall_cm * 1.0, 1),
    }


def encode_static_features(df):
    """Encode static features identically to build_training_data.py."""
    sys.path.insert(0, _SCRIPT_DIR)
    from build_training_data import encode_static_features as _encode
    return _encode(df)


def add_weather_and_interactions(pred_df, weather, temporal=None):
    """Broadcast weather features and compute interaction terms.

    Args:
        pred_df: DataFrame with encoded static features
        weather: dict of weather scenario values
        temporal: optional dict with temporal features (day_offset, days_since_peak_snowfall,
                  is_post_storm, cumulative_snowfall_cm, cumulative_thaw_degree_hours,
                  daily_snowfall_cm). Only used when model expects temporal features.
    """
    # Weather features (same for every segment)
    pred_df["total_snowfall_cm"] = np.float32(weather["total_snowfall_cm"])
    pred_df["max_daily_snowfall_cm"] = np.float32(weather["max_daily_snowfall_cm"])
    pred_df["min_temp_c"] = np.float32(weather["min_temp_c"])
    pred_df["max_wind_kmh"] = np.float32(weather["max_wind_kmh"])
    pred_df["duration_days"] = np.float32(weather["duration_days"])
    pred_df["precip_total_mm"] = np.float32(weather["precip_total_mm"])

    # Interaction features (must match build_training_data.py)
    snow = weather["total_snowfall_cm"]
    temp = weather["min_temp_c"]
    wind = weather["max_wind_kmh"]

    pred_df["snow_x_steep_slope"] = np.float32(
        snow * pred_df.get("near_steep_slope", 0))
    if "incline_pct" in pred_df.columns:
        pred_df["snow_x_incline"] = np.float32(
            snow * pred_df["incline_pct"].abs())
    else:
        pred_df["snow_x_incline"] = np.float32(0)
    pred_df["snow_x_elevation"] = np.float32(
        snow * pred_df.get("elevation_m", 0) / 1000.0)
    pred_df["temp_x_bridge"] = np.float32(
        (-temp) * pred_df.get("is_bridge", 0))
    rw = pd.to_numeric(pred_df.get("roadwidth", 0), errors="coerce").fillna(0)
    pred_df["wind_x_roadwidth"] = np.float32(wind / (rw + 1))

    # Temporal features (Part B: for models trained with day-level granularity)
    if temporal:
        day_offset = temporal.get("day_offset", 0)
        pred_df["day_offset"] = np.float32(day_offset)
        pred_df["days_since_peak_snowfall"] = np.float32(
            temporal.get("days_since_peak_snowfall", 0))
        pred_df["is_post_storm"] = np.float32(
            temporal.get("is_post_storm", 0))
        pred_df["cumulative_snowfall_cm"] = np.float32(
            temporal.get("cumulative_snowfall_cm", snow))
        pred_df["cumulative_thaw_degree_hours"] = np.float32(
            temporal.get("cumulative_thaw_degree_hours", 0))
        pred_df["daily_snowfall_cm"] = np.float32(
            temporal.get("daily_snowfall_cm", snow))
        # Temporal interaction features
        pred_df["thaw_x_steep_slope"] = np.float32(
            temporal.get("cumulative_thaw_degree_hours", 0)
            * pred_df.get("near_steep_slope", 0))
        pred_df["post_storm_x_elevation"] = np.float32(
            temporal.get("is_post_storm", 0)
            * pred_df.get("elevation_m", 0) / 1000.0)
        pred_df["day_offset_x_snow"] = np.float32(
            day_offset * snow)

    return pred_df


def compute_weather_severity(weather):
    """Map weather scenario to a [0.1, 1.0] severity score."""
    snowfall_f = min(weather["total_snowfall_cm"] / 25.0, 1.0)
    temp_f = min(max(0, -weather["min_temp_c"]) / 15.0, 1.0)
    wind_f = min(weather["max_wind_kmh"] / 60.0, 1.0)
    duration_f = min(weather["duration_days"] / 3.0, 1.0)
    severity = 0.45 * snowfall_f + 0.20 * temp_f + 0.20 * wind_f + 0.15 * duration_f
    return max(0.1, min(1.0, severity))


def calibrate_scores(raw_scores, weather_severity, calibration=None):
    """Convert raw ranker scores to [0,1] with weather-aware calibration.

    Strategy:
      1. Use per-prediction percentile sigmoid to get a segment RANKING in [0,1].
      2. Apply weather_severity to compress the distribution: light snow pushes
         most roads toward 0 (low risk), blizzard spreads them out.
    """
    n = len(raw_scores)
    if n == 0:
        return raw_scores

    # Step 1: Sigmoid based on this prediction's score distribution
    p75 = np.percentile(raw_scores, 75)
    p99 = np.percentile(raw_scores, 99)
    spread = max(p99 - p75, 1e-6)
    z = (raw_scores - p75) / spread * 3.0
    ranking_score = 1.0 / (1.0 + np.exp(-z))

    # Step 2: Apply weather severity as a power transform.
    # severity=1.0 (blizzard)  → exponent ~0.5 → pushes scores UP (more high-risk)
    # severity=0.1 (light snow) → exponent ~3.0 → pushes scores DOWN (fewer high-risk)
    # Exponent: higher severity → lower exponent → more spread toward high end
    exponent = 1.0 / max(weather_severity, 0.1)
    exponent = max(0.3, min(exponent, 5.0))  # clamp for stability
    calibrated = np.power(ranking_score, exponent)

    return np.clip(calibrated, 0, 1)


def compute_effective_weather(weather, recent_snowfall_cm, hours_since_snow,
                              temp_avg_since_snow_c=None):
    """Adjust weather dict to account for lingering snow from recent storms.

    Uses temperature-dependent exponential decay + plowing linear decay to
    estimate how much recent snow is still on the ground.

    Args:
        weather: dict with current weather scenario
        recent_snowfall_cm: total snowfall in recent past
        hours_since_snow: hours since last significant snowfall stopped
        temp_avg_since_snow_c: avg temp since snow stopped (default: use min_temp_c)

    Returns:
        Updated weather dict with effective snowfall merged in.
    """
    if recent_snowfall_cm <= 0 or hours_since_snow is None:
        return weather

    # Use provided avg temp, or fall back to the current scenario's min temp
    temp_c = temp_avg_since_snow_c if temp_avg_since_snow_c is not None else weather["min_temp_c"]

    # Thermal decay rate: colder = slower melt
    #   -10C: k=0.005/hr (half-life ~139h, snow preserved)
    #     0C: k=0.02/hr  (half-life ~35h)
    #    +5C: k=0.085/hr (half-life ~8h)
    k = max(0.005, 0.02 + 0.013 * max(0, temp_c))
    thermal_factor = np.exp(-k * hours_since_snow)

    # Plow linear decay: city plows clear ~1%/hr, floor at 10%
    plow_factor = max(0.1, 1.0 - 0.01 * hours_since_snow)

    effective_recent = recent_snowfall_cm * thermal_factor * plow_factor

    weather = weather.copy()
    current_snow = weather["total_snowfall_cm"]
    weather["total_snowfall_cm"] = round(current_snow + effective_recent, 2)
    weather["max_daily_snowfall_cm"] = round(
        max(weather["max_daily_snowfall_cm"], effective_recent), 2)
    weather["duration_days"] = max(weather["duration_days"],
                                   int(np.ceil(hours_since_snow / 24)) + 1)
    weather["precip_total_mm"] = round(
        weather["precip_total_mm"] + effective_recent, 2)

    return weather, {
        "recent_snowfall_cm": recent_snowfall_cm,
        "hours_since_snow": hours_since_snow,
        "temp_avg_since_snow_c": temp_c,
        "thermal_decay_rate": round(k, 4),
        "thermal_factor": round(thermal_factor, 4),
        "plow_factor": round(plow_factor, 4),
        "effective_recent_snow_cm": round(effective_recent, 2),
        "effective_total_snowfall_cm": round(current_snow + effective_recent, 2),
    }


def main():
    import lightgbm as lgb

    parser = argparse.ArgumentParser(
        description="Predict road closure risk for a weather scenario"
    )
    parser.add_argument("--snowfall_cm", type=float, default=None)
    parser.add_argument("--min_temp_c", type=float, default=None)
    parser.add_argument("--max_wind_kmh", type=float, default=None)
    parser.add_argument("--duration_days", type=int, default=None)
    parser.add_argument("--precip_mm", type=float, default=None)
    parser.add_argument("--use_forecast", action="store_true")
    # Lingering snow arguments (Part A: effective weather at prediction time)
    parser.add_argument("--recent_snowfall_cm", type=float, default=None,
                        help="Total snowfall in recent past (e.g. 20cm fell yesterday)")
    parser.add_argument("--hours_since_snow", type=float, default=None,
                        help="Hours since last significant snowfall stopped")
    parser.add_argument("--temp_avg_since_snow_c", type=float, default=None,
                        help="Avg temperature since snow stopped (for melt estimation)")
    # Temporal features (Part B: for models trained with temporal features)
    parser.add_argument("--day_offset", type=int, default=None,
                        help="Day offset within storm (0 = first day of snow)")
    parser.add_argument("--days_since_peak", type=int, default=None,
                        help="Days since peak snowfall day")
    parser.add_argument("--is_post_storm", action="store_true",
                        help="Whether this is after the last snow day")
    parser.add_argument("--cumulative_thaw_degree_hours", type=float, default=None,
                        help="Cumulative degree-hours above 0C since last snow")
    args = parser.parse_args()

    print("Road Closure Risk Prediction")
    print("=" * 60)

    # ── Choose model ──────────────────────────────────────────────────────
    use_ranker = os.path.exists(RANKER_MODEL_FILE) and os.path.exists(RANKER_META_FILE)

    if use_ranker:
        model_file = RANKER_MODEL_FILE
        meta_file = RANKER_META_FILE
        model_label = "LambdaRank ranker"
    elif os.path.exists(REGRESSION_MODEL_FILE):
        model_file = REGRESSION_MODEL_FILE
        meta_file = REGRESSION_META_FILE
        model_label = "Regression (fallback)"
    else:
        print("Error: No model found. Run train_ranker.py or train_model.py first.")
        sys.exit(1)

    print(f"\nLoading {model_label}...")
    model = lgb.Booster(model_file=model_file)
    with open(meta_file, "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    print(f"  Model: {model_label} ({model.num_trees()} trees, {len(feature_cols)} features)")

    # ── Weather input ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Weather scenario...")
    print(f"{'='*60}")

    weather = None
    if args.use_forecast or (args.snowfall_cm is None and os.path.exists(FORECAST_JSON)):
        print("  Mode: NWS forecast")
        if os.path.exists(FORECAST_JSON):
            weather = parse_nws_forecast(FORECAST_JSON)
            if weather:
                print(f"  Parsed forecast from {FORECAST_JSON}")
        else:
            print(f"  Warning: {FORECAST_JSON} not found")

    if weather is None or args.snowfall_cm is not None:
        print("  Mode: CLI arguments")
        weather = {
            "total_snowfall_cm": args.snowfall_cm or 5.0,
            "max_daily_snowfall_cm": args.snowfall_cm or 5.0,
            "min_temp_c": args.min_temp_c or -5.0,
            "max_wind_kmh": args.max_wind_kmh or 20.0,
            "duration_days": args.duration_days or 1,
            "precip_total_mm": args.precip_mm or (args.snowfall_cm or 5.0),
        }

    # ── Lingering snow adjustment ────────────────────────────────────────
    lingering_info = None
    if args.recent_snowfall_cm and args.recent_snowfall_cm > 0:
        hours = args.hours_since_snow if args.hours_since_snow is not None else 12.0
        weather, lingering_info = compute_effective_weather(
            weather, args.recent_snowfall_cm, hours, args.temp_avg_since_snow_c
        )
        print(f"\n  Lingering snow adjustment:")
        print(f"    Recent snowfall: {lingering_info['recent_snowfall_cm']} cm")
        print(f"    Hours since snow: {lingering_info['hours_since_snow']}")
        print(f"    Avg temp since snow: {lingering_info['temp_avg_since_snow_c']}C")
        print(f"    Thermal factor: {lingering_info['thermal_factor']}")
        print(f"    Plow factor: {lingering_info['plow_factor']}")
        print(f"    Effective recent snow: {lingering_info['effective_recent_snow_cm']} cm")
        print(f"    Effective total snow: {lingering_info['effective_total_snowfall_cm']} cm")

    print(f"\n  Weather scenario:")
    for k, v in weather.items():
        print(f"    {k}: {v}")

    # ── Load and encode segments ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading road segments...")
    print(f"{'='*60}")

    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found. Run build_dataset.py first.")
        sys.exit(1)

    dataset = pd.read_csv(DATASET_CSV)

    # Deduplicate by objectid BEFORE encoding (consistent with training)
    dataset = dataset.drop_duplicates(subset=["objectid"], keep="first").reset_index(drop=True)
    print(f"  Loaded {len(dataset)} unique segments")

    static = encode_static_features(dataset)
    print(f"  Encoded {len(static)} segments")

    # ── Build prediction matrix with weather ──────────────────────────────
    print(f"\n{'='*60}")
    print("Building prediction matrix...")
    print(f"{'='*60}")

    pred_df = static.copy()

    # Build temporal features dict if the model expects them
    temporal = None
    has_temporal_features = any(c in feature_cols for c in [
        "day_offset", "days_since_peak_snowfall", "is_post_storm",
        "cumulative_snowfall_cm", "cumulative_thaw_degree_hours", "daily_snowfall_cm",
    ])
    if has_temporal_features:
        # Derive temporal values from CLI args or lingering snow info
        day_offset = args.day_offset if args.day_offset is not None else 0
        if args.recent_snowfall_cm and args.hours_since_snow:
            day_offset = max(day_offset, int(np.ceil(args.hours_since_snow / 24)))
        temporal = {
            "day_offset": day_offset,
            "days_since_peak_snowfall": args.days_since_peak if args.days_since_peak is not None else day_offset,
            "is_post_storm": int(args.is_post_storm) if args.is_post_storm else (1 if args.recent_snowfall_cm and weather.get("total_snowfall_cm", 0) == 0 else 0),
            "cumulative_snowfall_cm": weather["total_snowfall_cm"],
            "cumulative_thaw_degree_hours": args.cumulative_thaw_degree_hours if args.cumulative_thaw_degree_hours is not None else 0,
            "daily_snowfall_cm": weather.get("max_daily_snowfall_cm", weather["total_snowfall_cm"]),
        }
        print(f"\n  Temporal features: {temporal}")

    pred_df = add_weather_and_interactions(pred_df, weather, temporal=temporal)

    # Ensure all feature columns present
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
            print(f"  Warning: Missing feature '{col}', filled with 0")

    X = pred_df[feature_cols]
    print(f"  Prediction matrix: {X.shape}")

    # ── Predict ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating predictions...")
    print(f"{'='*60}")

    raw_scores = model.predict(X)
    print(f"  Raw score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")

    # Compute weather severity for calibration
    severity = compute_weather_severity(weather)
    print(f"  Weather severity: {severity:.3f}")

    # Calibrate: ranking sigmoid + weather-based power transform
    scores = calibrate_scores(raw_scores, severity)
    print(f"  Calibrated range: [{scores.min():.4f}, {scores.max():.4f}]")

    pred_df["risk_score"] = np.round(scores, 6)
    pred_df["risk_category"] = pred_df["risk_score"].apply(categorize_risk)

    # ── Build output ──────────────────────────────────────────────────────
    # Map back metadata from original dataset (same length after dedup)
    orig_idx = static.attrs.get("_original_index")
    if orig_idx is not None:
        src = dataset.iloc[orig_idx].reset_index(drop=True)
    else:
        src = dataset

    output_cols = ["objectid"]
    if "streetname" in src.columns:
        pred_df["streetname"] = src["streetname"].values
        output_cols.append("streetname")
    output_cols.extend(["mid_lat", "mid_lng", "risk_score", "risk_category"])
    if "domi_class" in src.columns:
        pred_df["domi_class"] = src["domi_class"].values
        output_cols.append("domi_class")
    if "is_snow_emergency_route" in src.columns:
        pred_df["is_snow_emergency_route"] = src["is_snow_emergency_route"].values
        output_cols.append("is_snow_emergency_route")

    output_df = pred_df[output_cols].sort_values("risk_score", ascending=False)

    # ── Save CSV ──────────────────────────────────────────────────────────
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  CSV saved to {OUTPUT_CSV}")

    # ── Save GeoJSON ──────────────────────────────────────────────────────
    features = []
    for _, row in output_df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["mid_lng"]) if pd.notna(row["mid_lng"]) else 0,
                    float(row["mid_lat"]) if pd.notna(row["mid_lat"]) else 0,
                ],
            },
            "properties": {
                "objectid": int(row["objectid"]) if pd.notna(row["objectid"]) else 0,
                "streetname": str(row.get("streetname", "")),
                "risk_score": float(row["risk_score"]),
                "risk_category": str(row["risk_category"]),
                "domi_class": str(row.get("domi_class", "")),
                "is_snow_emergency_route": bool(
                    str(row.get("is_snow_emergency_route", "")).lower() in ["true", "1"]
                ),
            },
        }
        features.append(feature)

    model_basename = os.path.basename(model_file)
    meta = {
        "generated_at": datetime.now().isoformat(),
        "weather_scenario": weather,
        "model": model_label,
        "model_file": model_basename,
        "segment_count": len(output_df),
        "risk_distribution": {
            cat: int((output_df["risk_category"] == cat).sum())
            for _, _, cat in RISK_THRESHOLDS
        },
    }
    if lingering_info:
        meta["lingering_snow"] = lingering_info
    if temporal:
        meta["temporal_features"] = temporal

    geojson = {
        "type": "FeatureCollection",
        "metadata": meta,
        "features": features,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  GeoJSON saved to {OUTPUT_JSON}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {model_label}")
    print(f"  Total segments: {len(output_df)}")
    print(f"  Risk score range: [{output_df['risk_score'].min():.4f}, "
          f"{output_df['risk_score'].max():.4f}]")
    print(f"  Mean risk: {output_df['risk_score'].mean():.4f}")
    print(f"\n  Risk distribution:")
    for _, _, cat in RISK_THRESHOLDS:
        count = (output_df["risk_category"] == cat).sum()
        pct = count / len(output_df) * 100
        print(f"    {cat:12s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n  Top 10 highest risk roads:")
    for _, row in output_df.head(10).iterrows():
        name = row.get("streetname", "Unknown")
        print(f"    {name:30s}  score={row['risk_score']:.4f}  "
              f"cat={row['risk_category']}")


if __name__ == "__main__":
    main()
