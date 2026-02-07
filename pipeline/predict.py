"""
Generate per-segment road closure risk predictions for a given weather scenario.

Two input modes:
    1. NWS forecast: Parse data/weather_forecast_pgh.json from collect_weather_realtime.py
    2. CLI args: --snowfall_cm 10 --min_temp_c -5 --max_wind_kmh 30 --duration_days 2

Scoring: Load model + static features → broadcast weather → predict → clamp [0,1]
         → categorize (very_low/low/moderate/high/very_high).

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

# Input files
MODEL_FILE = os.path.join(DATA_DIR, "model_lgbm.txt")
METADATA_FILE = os.path.join(DATA_DIR, "model_metadata.json")
DATASET_CSV = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")
FORECAST_JSON = os.path.join(DATA_DIR, "weather_forecast_pgh.json")

# Output files
OUTPUT_CSV = os.path.join(DATA_DIR, "predictions_latest.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "predictions_latest.json")

# Risk categories
RISK_THRESHOLDS = [
    (0.0, 0.15, "very_low"),
    (0.15, 0.35, "low"),
    (0.35, 0.55, "moderate"),
    (0.55, 0.75, "high"),
    (0.75, 1.01, "very_high"),
]


def categorize_risk(score):
    """Map a 0-1 risk score to a category label."""
    for low, high, label in RISK_THRESHOLDS:
        if low <= score < high:
            return label
    return "very_high"


def parse_nws_forecast(forecast_path):
    """Extract weather features from NWS forecast JSON.

    Looks for snowfall, temperature, and wind data in the forecast periods.
    Returns a dict of weather features.
    """
    with open(forecast_path, "r") as f:
        forecast = json.load(f)

    # Handle both raw NWS API format and our simplified format
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

    # Aggregate across forecast periods (next 48-72h)
    temps = []
    winds = []
    snow_keywords = ["snow", "blizzard", "winter storm", "ice"]

    has_snow = False
    for period in periods[:6]:  # ~3 days of data
        # Temperature
        temp = period.get("temperature")
        if temp is not None:
            unit = period.get("temperatureUnit", "F")
            if unit == "F":
                temp = (temp - 32) * 5 / 9
            temps.append(temp)

        # Wind
        wind_str = period.get("windSpeed", "")
        if isinstance(wind_str, str):
            # Parse "15 mph" or "10 to 20 mph"
            import re
            nums = re.findall(r"\d+", wind_str)
            if nums:
                # Convert mph to km/h
                max_wind = max(int(n) for n in nums) * 1.60934
                winds.append(max_wind)
        elif isinstance(wind_str, (int, float)):
            winds.append(wind_str * 1.60934)

        # Snow detection
        detail = period.get("detailedForecast", "") or period.get("shortForecast", "")
        if any(kw in detail.lower() for kw in snow_keywords):
            has_snow = True

    if not has_snow:
        print("  Warning: No snow in forecast. Predictions may show very low risk.")

    # Estimate snowfall from forecast text (rough heuristic)
    snowfall_cm = 5.0 if has_snow else 0.5  # Conservative default
    for period in periods[:6]:
        detail = (period.get("detailedForecast", "") or "").lower()
        # Look for "3 to 5 inches" patterns
        import re
        inches_match = re.findall(r"(\d+)\s*(?:to\s*(\d+))?\s*inch", detail)
        if inches_match:
            max_inches = max(int(m[1]) if m[1] else int(m[0]) for m in inches_match)
            snowfall_cm = max(snowfall_cm, max_inches * 2.54)

    weather = {
        "total_snowfall_cm": round(snowfall_cm, 1),
        "max_daily_snowfall_cm": round(snowfall_cm, 1),  # Assume single-day event
        "min_temp_c": round(min(temps), 1) if temps else -5.0,
        "max_wind_kmh": round(max(winds), 1) if winds else 20.0,
        "duration_days": 1,
        "precip_total_mm": round(snowfall_cm * 1.0, 1),  # Rough SWE estimate
    }

    return weather


def encode_static_features(df):
    """Encode static features identically to build_training_data.py."""
    # Import encoding maps from build_training_data
    sys.path.insert(0, _SCRIPT_DIR)
    from build_training_data import encode_static_features as _encode
    return _encode(df)


def main():
    import lightgbm as lgb

    parser = argparse.ArgumentParser(
        description="Predict road closure risk for a weather scenario"
    )
    parser.add_argument("--snowfall_cm", type=float, default=None,
                        help="Total snowfall in cm")
    parser.add_argument("--min_temp_c", type=float, default=None,
                        help="Minimum temperature in Celsius")
    parser.add_argument("--max_wind_kmh", type=float, default=None,
                        help="Maximum wind speed in km/h")
    parser.add_argument("--duration_days", type=int, default=None,
                        help="Storm duration in days")
    parser.add_argument("--precip_mm", type=float, default=None,
                        help="Total precipitation in mm")
    parser.add_argument("--use_forecast", action="store_true",
                        help="Use NWS forecast JSON instead of CLI args")
    args = parser.parse_args()

    print("Road Closure Risk Prediction")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Run train_model.py first.")
        sys.exit(1)

    model = lgb.Booster(model_file=MODEL_FILE)
    print(f"  Model loaded: {model.num_trees()} trees")

    # Load metadata for feature column order
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found. Run train_model.py first.")
        sys.exit(1)

    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    print(f"  Feature columns: {len(feature_cols)}")

    # ── Determine weather input ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Weather scenario...")
    print(f"{'='*60}")

    weather = None

    # Mode 1: NWS forecast
    if args.use_forecast or (args.snowfall_cm is None and os.path.exists(FORECAST_JSON)):
        print("  Mode: NWS forecast")
        if os.path.exists(FORECAST_JSON):
            weather = parse_nws_forecast(FORECAST_JSON)
            if weather:
                print(f"  Parsed forecast from {FORECAST_JSON}")
        else:
            print(f"  Warning: {FORECAST_JSON} not found")

    # Mode 2: CLI args (override or fallback)
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

    print(f"\n  Weather scenario:")
    for k, v in weather.items():
        print(f"    {k}: {v}")

    # ── Load and encode static features ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading road segments...")
    print(f"{'='*60}")

    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found. Run build_dataset.py first.")
        sys.exit(1)

    dataset = pd.read_csv(DATASET_CSV)
    print(f"  Loaded {len(dataset)} segments")

    static = encode_static_features(dataset)
    print(f"  Encoded {len(static)} segments with {len(static.columns)} features")

    # ── Build prediction matrix (static features only) ──────────────────────
    print(f"\n{'='*60}")
    print("Building prediction matrix...")
    print(f"{'='*60}")

    pred_df = static.copy()

    # Ensure all feature columns are present and in correct order
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
            print(f"  Warning: Missing feature '{col}', filled with 0")

    X = pred_df[feature_cols]
    print(f"  Prediction matrix: {X.shape}")

    # ── Predict segment-level risk ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating predictions...")
    print(f"{'='*60}")

    # Model predicts segment-level risk (weather was excluded from training).
    # Outputs are narrow-band because within-storm label variance is small,
    # so we rescale to [0, 1] via min-max normalization before weather scaling.
    raw_scores = model.predict(X)

    # Min-max normalize raw scores to [0, 1] segment risk
    raw_min = raw_scores.min()
    raw_max = raw_scores.max()
    if raw_max > raw_min:
        segment_risk = (raw_scores - raw_min) / (raw_max - raw_min)
    else:
        segment_risk = np.full_like(raw_scores, 0.5)

    print(f"  Raw model output range: [{raw_min:.4f}, {raw_max:.4f}]")
    print(f"  Normalized segment risk: [{segment_risk.min():.4f}, {segment_risk.max():.4f}]")

    # Weather severity score: maps weather conditions to a [0.1, 1.0] scale
    # where 1.0 = extreme storm. This controls the "ceiling" of risk scores.
    snowfall_factor = min(weather["total_snowfall_cm"] / 25.0, 1.0)  # 25cm = max
    temp_factor = min(max(0, -weather["min_temp_c"] - 0) / 15.0, 1.0)  # -15C = max
    wind_factor = min(weather["max_wind_kmh"] / 60.0, 1.0)  # 60 km/h = max
    duration_factor = min(weather["duration_days"] / 3.0, 1.0)  # 3 days = max

    weather_severity = (
        0.45 * snowfall_factor +
        0.20 * temp_factor +
        0.20 * wind_factor +
        0.15 * duration_factor
    )
    weather_severity = max(0.1, min(1.0, weather_severity))

    print(f"  Weather severity: {weather_severity:.3f}")
    print(f"    snowfall={snowfall_factor:.2f}, temp={temp_factor:.2f}, "
          f"wind={wind_factor:.2f}, duration={duration_factor:.2f}")

    # Final risk: segment_risk scaled by weather severity
    # Low weather → scores compressed toward 0; extreme weather → full range
    scores = segment_risk * weather_severity
    scores = np.clip(scores, 0, 1)

    pred_df["risk_score"] = np.round(scores, 6)
    pred_df["risk_category"] = pred_df["risk_score"].apply(categorize_risk)

    # ── Build output ──────────────────────────────────────────────────────────
    # Merge back street name and classification from the original dataset.
    # encode_static_features() drops rows with missing coords and resets the
    # index, so pred_df's index no longer matches dataset's.  Use the
    # preserved original-index mapping to look up the correct source rows.
    orig_idx = static.attrs.get("_original_index")
    if orig_idx is not None:
        src = dataset.iloc[orig_idx].reset_index(drop=True)
    else:
        # Fallback: indices happen to align (no rows were dropped)
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

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  CSV saved to {OUTPUT_CSV}")

    # ── Save GeoJSON ──────────────────────────────────────────────────────────
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

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "weather_scenario": weather,
            "segment_count": len(output_df),
            "model_file": os.path.basename(MODEL_FILE),
            "risk_distribution": {
                cat: int((output_df["risk_category"] == cat).sum())
                for _, _, cat in RISK_THRESHOLDS
            },
        },
        "features": features,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  GeoJSON saved to {OUTPUT_JSON}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
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
    top10 = output_df.head(10)
    for _, row in top10.iterrows():
        name = row.get("streetname", "Unknown")
        print(f"    {name:30s}  score={row['risk_score']:.4f}  "
              f"cat={row['risk_category']}")


if __name__ == "__main__":
    main()
