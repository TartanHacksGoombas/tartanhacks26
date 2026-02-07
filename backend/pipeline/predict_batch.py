"""
Multi-day batch prediction for the web backend.

Loads the model ONCE, encodes static features ONCE, then generates predictions
for days 0..max_day_offset with appropriate temporal features for each day.

Output: JSON to stdout with structure:
{
  "weather": {...},
  "days": {
    "0": [{"objectid": 123, "risk_score": 0.45, "risk_category": "moderate"}, ...],
    "1": [...],
    ...
  }
}

Called by the Express backend via child_process.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")

sys.path.insert(0, _SCRIPT_DIR)
from predict import (
    encode_static_features,
    add_weather_and_interactions,
    calibrate_scores,
    categorize_risk,
    compute_weather_severity,
    compute_effective_weather,
    RISK_THRESHOLDS,
)


def main():
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description="Batch predict road risk for multiple days")
    parser.add_argument("--snowfall_cm", type=float, required=True)
    parser.add_argument("--min_temp_c", type=float, default=-5.0)
    parser.add_argument("--max_wind_kmh", type=float, default=20.0)
    parser.add_argument("--duration_days", type=int, default=1)
    parser.add_argument("--precip_mm", type=float, default=None)
    parser.add_argument("--max_day_offset", type=int, default=6)
    args = parser.parse_args()

    # Suppress all non-JSON output by redirecting prints to stderr
    _print = lambda *a, **kw: print(*a, file=sys.stderr, **kw)

    # Load model
    ranker_model = os.path.join(DATA_DIR, "ranker_lgbm.txt")
    ranker_meta = os.path.join(DATA_DIR, "ranker_metadata.json")
    regression_model = os.path.join(DATA_DIR, "model_lgbm.txt")
    regression_meta = os.path.join(DATA_DIR, "model_metadata.json")

    if os.path.exists(ranker_model) and os.path.exists(ranker_meta):
        model_file, meta_file = ranker_model, ranker_meta
    elif os.path.exists(regression_model):
        model_file, meta_file = regression_model, regression_meta
    else:
        _print("Error: No model found")
        sys.exit(1)

    model = lgb.Booster(model_file=model_file)
    with open(meta_file) as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_columns"]
    _print(f"Loaded model with {len(feature_cols)} features")

    # Check if model expects temporal features
    has_temporal = any(c in feature_cols for c in [
        "day_offset", "days_since_peak_snowfall", "is_post_storm",
        "cumulative_snowfall_cm", "cumulative_thaw_degree_hours", "daily_snowfall_cm",
    ])

    # Load and encode static features (once)
    # Redirect stdout to stderr during encoding (encode_static_features prints to stdout)
    dataset_csv = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")
    dataset = pd.read_csv(dataset_csv)
    dataset = dataset.drop_duplicates(subset=["objectid"], keep="first").reset_index(drop=True)
    _orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    static = encode_static_features(dataset)
    sys.stdout = _orig_stdout
    objectids = dataset["objectid"].values
    _print(f"Encoded {len(static)} segments")

    # Weather scenario
    weather = {
        "total_snowfall_cm": args.snowfall_cm,
        "max_daily_snowfall_cm": args.snowfall_cm,
        "min_temp_c": args.min_temp_c,
        "max_wind_kmh": args.max_wind_kmh,
        "duration_days": args.duration_days,
        "precip_total_mm": args.precip_mm if args.precip_mm is not None else args.snowfall_cm,
    }

    result = {"weather": weather, "days": {}}

    for day_offset in range(args.max_day_offset + 1):
        pred_df = static.copy()

        # Compute temporal features for this day
        # Days 0..duration-1: during storm (snow accumulating)
        # Days duration..max: post-storm (lingering, decaying)
        is_during_storm = day_offset < args.duration_days
        daily_snow = args.snowfall_cm / max(args.duration_days, 1) if is_during_storm else 0
        cumulative_snow = (
            args.snowfall_cm * min(day_offset + 1, args.duration_days) / max(args.duration_days, 1)
        )
        # Peak snowfall is the day with most snow (day 0 for single-day storms)
        peak_day = 0
        days_since_peak = day_offset - peak_day

        # Post-storm: temperature-based thaw
        post_storm_hours = max(0, (day_offset - args.duration_days + 1) * 24) if not is_during_storm else 0
        # Thaw degree hours: hours above 0C
        thaw_temp = max(0, args.min_temp_c + 3)  # rough avg temp estimate
        cumulative_thaw = thaw_temp * post_storm_hours

        # For post-storm days, apply lingering snow decay to weather
        day_weather = weather.copy()
        if not is_during_storm and day_offset > 0:
            hours_since = (day_offset - args.duration_days + 1) * 24
            result_tuple = compute_effective_weather(
                {"total_snowfall_cm": 0, "max_daily_snowfall_cm": 0,
                 "min_temp_c": args.min_temp_c, "max_wind_kmh": args.max_wind_kmh,
                 "duration_days": 1, "precip_total_mm": 0},
                args.snowfall_cm, hours_since, args.min_temp_c
            )
            if isinstance(result_tuple, tuple):
                day_weather = result_tuple[0]
            else:
                day_weather = result_tuple

        temporal = None
        if has_temporal:
            temporal = {
                "day_offset": day_offset,
                "days_since_peak_snowfall": days_since_peak,
                "is_post_storm": 0 if is_during_storm else 1,
                "cumulative_snowfall_cm": cumulative_snow,
                "cumulative_thaw_degree_hours": cumulative_thaw,
                "daily_snowfall_cm": daily_snow,
            }

        pred_df = add_weather_and_interactions(pred_df, day_weather, temporal=temporal)

        for col in feature_cols:
            if col not in pred_df.columns:
                pred_df[col] = 0

        X = pred_df[feature_cols]
        raw_scores = model.predict(X)

        severity = compute_weather_severity(day_weather)
        scores = calibrate_scores(raw_scores, severity)

        day_preds = []
        for i in range(len(scores)):
            score = float(np.round(scores[i], 6))
            day_preds.append({
                "objectid": int(objectids[i]),
                "risk_score": score,
                "risk_category": categorize_risk(score),
            })

        result["days"][str(day_offset)] = day_preds
        _print(f"Day {day_offset}: severity={severity:.3f}, score range=[{scores.min():.3f}, {scores.max():.3f}]")

    # Output JSON to stdout (only JSON goes to stdout)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
