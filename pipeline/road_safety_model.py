"""
XGBoost regressor to predict a composite Winter Road Safety Score (0-100).

Uses the prediction-ready dataset built from centerlines, OSM, elevation,
slopes, bridges, landslides, crashes, 311 complaints, plow activity,
PennDOT routes, DOMI closures, and snow emergency routes.

The safety score is computed from real data — no synthetic labels.
Higher score = MORE dangerous road in winter conditions.

Requires: pip install xgboost scikit-learn pandas matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")


def load_and_prepare_data():
    """Load the prediction-ready dataset and prepare features + safety score."""
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Columns: {', '.join(df.columns)}")

    # --- Compute composite safety score from REAL data ---
    print(f"\n{'=' * 60}")
    print("COMPUTING COMPOSITE SAFETY SCORE")
    print(f"{'=' * 60}")

    df = compute_safety_score(df)

    # --- Feature columns (inputs to the model) ---
    # These are road characteristics the model uses to PREDICT the score.
    # We exclude the score components themselves to avoid data leakage.
    feature_cols = [
        # Road attributes
        "domi_class", "paveclass", "speedlimit", "num_lanes", "roadwidth",
        "oneway", "highway_type_osm", "surface_osm",
        "mid_lat", "mid_lng",
        # Topographic risk
        "elevation_m", "near_steep_slope", "near_landslide",
        # Structural risk
        "is_bridge", "bridge_age_years",
        # Priority/classification
        "penndot_aadt", "penndot_pavement_idx",
        "is_snow_emergency_route",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat = df[feature_cols].copy()

    # --- Parse boolean columns ---
    bool_cols = ["near_steep_slope", "near_landslide", "is_bridge", "is_snow_emergency_route"]
    for col in bool_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)

    # --- Parse numeric columns ---
    numeric_cols = [
        "speedlimit", "num_lanes", "roadwidth", "elevation_m",
        "bridge_age_years", "penndot_aadt", "penndot_pavement_idx",
        "mid_lat", "mid_lng",
    ]
    for col in numeric_cols:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    # --- Encode categorical columns ---
    label_encoders = {}
    cat_cols = ["domi_class", "paveclass", "oneway", "highway_type_osm", "surface_osm"]
    for col in cat_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna("unknown").astype(str)
            le = LabelEncoder()
            df_feat[col] = le.fit_transform(df_feat[col])
            label_encoders[col] = le

    df_feat = df_feat.fillna(0)

    target = df["safety_score"].values

    print(f"\nFeature matrix: {df_feat.shape[0]} rows x {df_feat.shape[1]} features")
    print(f"Features: {', '.join(df_feat.columns)}")
    print(f"\nSafety score stats:")
    print(f"  Mean:   {target.mean():.1f}")
    print(f"  Median: {np.median(target):.1f}")
    print(f"  Min:    {target.min():.1f}")
    print(f"  Max:    {target.max():.1f}")
    print(f"  Std:    {target.std():.1f}")

    return df, df_feat, target, label_encoders


def compute_safety_score(df):
    """
    Compute a 0-100 composite winter road safety score from real data.

    Components (weighted):
      30% — Crash risk:        winter_crash_count + 10*winter_crash_fatal
      20% — Closure history:   domi_closure_count + 3*domi_full_closure_count
      15% — Community reports: snow_complaint_count
      10% — Plow coverage:    inverse of plow_coverage_score (less plow = worse)
      10% — Terrain risk:     steep slope + landslide + elevation
       5% — Structural risk:  bridge + bridge age
       5% — Road geometry:    narrow + low speed limit
       5% — Pavement quality: inverse of penndot_pavement_idx
    """
    scaler = MinMaxScaler(feature_range=(0, 100))

    def safe_numeric(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    def safe_bool(col):
        if col in df.columns:
            return (df[col].astype(str).str.lower() == "true").astype(float)
        return pd.Series(0, index=df.index)

    # --- Crash risk (30%) ---
    crash_raw = safe_numeric("winter_crash_count") + 10 * safe_numeric("winter_crash_fatal")
    crash_score = scaler.fit_transform(crash_raw.values.reshape(-1, 1)).flatten()

    # --- Closure history (20%) ---
    closure_raw = safe_numeric("domi_closure_count") + 3 * safe_numeric("domi_full_closure_count")
    closure_score = scaler.fit_transform(closure_raw.values.reshape(-1, 1)).flatten()

    # --- Community reports (15%) ---
    complaint_raw = safe_numeric("snow_complaint_count")
    complaint_score = scaler.fit_transform(complaint_raw.values.reshape(-1, 1)).flatten()

    # --- Plow coverage (10%) — inverse: less plow = higher risk ---
    plow = safe_numeric("plow_coverage_score")
    plow_max = plow.max() if plow.max() > 0 else 1
    plow_score = scaler.fit_transform((plow_max - plow).values.reshape(-1, 1)).flatten()

    # --- Terrain risk (10%) ---
    terrain_raw = (
        safe_bool("near_steep_slope") * 40 +
        safe_bool("near_landslide") * 50 +
        np.clip((safe_numeric("elevation_m") - 250) / 150, 0, 1) * 10
    )
    terrain_score = np.clip(terrain_raw, 0, 100)

    # --- Structural risk (5%) ---
    struct_raw = (
        safe_bool("is_bridge") * 50 +
        np.clip(safe_numeric("bridge_age_years") / 100, 0, 1) * 50
    )
    struct_score = np.clip(struct_raw, 0, 100)

    # --- Road geometry (5%) ---
    width = safe_numeric("roadwidth")
    width_risk = np.clip((40 - width) / 30, 0, 1) * 50  # narrower = riskier
    speed = safe_numeric("speedlimit")
    speed_risk = np.clip((speed - 25) / 40, 0, 1) * 50  # faster = riskier
    geo_score = np.clip(width_risk + speed_risk, 0, 100)

    # --- Pavement quality (5%) — lower index = worse ---
    pave_idx = safe_numeric("penndot_pavement_idx")
    pave_max = pave_idx.max() if pave_idx.max() > 0 else 1
    pave_inv = pave_max - pave_idx
    # Only score roads that actually have pavement data
    has_pave = pave_idx > 0
    pave_score = np.zeros(len(df))
    if has_pave.any():
        pave_score[has_pave] = scaler.fit_transform(
            pave_inv[has_pave].values.reshape(-1, 1)
        ).flatten()

    # --- Weighted composite ---
    df["safety_score"] = (
        0.30 * crash_score +
        0.20 * closure_score +
        0.15 * complaint_score +
        0.10 * plow_score +
        0.10 * terrain_score +
        0.05 * struct_score +
        0.05 * geo_score +
        0.05 * pave_score
    )

    # Store sub-scores for analysis
    df["_crash_score"] = crash_score
    df["_closure_score"] = closure_score
    df["_complaint_score"] = complaint_score
    df["_plow_score"] = plow_score
    df["_terrain_score"] = terrain_score

    print(f"\nScore component breakdown (mean):")
    print(f"  Crash risk (30%):        {crash_score.mean():.1f}")
    print(f"  Closure history (20%):   {closure_score.mean():.1f}")
    print(f"  Snow complaints (15%):   {complaint_score.mean():.1f}")
    print(f"  Plow coverage (10%):     {plow_score.mean():.1f}")
    print(f"  Terrain risk (10%):      {terrain_score.mean():.1f}")
    print(f"  Structural risk (5%):    {struct_score.mean():.1f}")
    print(f"  Road geometry (5%):      {geo_score.mean():.1f}")
    print(f"  Pavement quality (5%):   {pave_score.mean():.1f}")

    return df


def train_model(df_feat, target):
    """Train XGBoost regressor and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        df_feat, target, test_size=0.2, random_state=42,
    )

    print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    print(f"\n{'=' * 60}")
    print("MODEL EVALUATION")
    print(f"{'=' * 60}")

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.2f} (avg error in safety score points)")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f} (1.0 = perfect, 0.0 = baseline)")

    cv_scores = cross_val_score(model, df_feat, target, cv=5,
                                 scoring="neg_mean_absolute_error")
    print(f"\n5-Fold CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    return model, X_test, y_test, y_pred


def plot_results(model, df, df_feat, y_test, y_pred):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Feature importance
    ax = axes[0, 0]
    importance = model.feature_importances_
    feat_names = df_feat.columns
    sorted_idx = np.argsort(importance)[-15:]
    ax.barh(range(len(sorted_idx)), importance[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx])
    ax.set_title("Feature Importance (Top 15)")
    ax.set_xlabel("Importance")

    # 2. Predicted vs Actual
    ax = axes[0, 1]
    ax.scatter(y_test, y_pred, alpha=0.3, s=5)
    ax.plot([0, 100], [0, 100], "r--", label="Perfect")
    ax.set_xlabel("Actual Safety Score")
    ax.set_ylabel("Predicted Safety Score")
    ax.set_title("Predicted vs Actual")
    ax.legend()

    # 3. Safety score distribution
    ax = axes[1, 0]
    ax.hist(df["safety_score"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Safety Score (0=Safe, 100=Dangerous)")
    ax.set_ylabel("Number of Road Segments")
    ax.set_title("Safety Score Distribution")
    ax.axvline(df["safety_score"].mean(), color="r", linestyle="--",
               label=f"Mean: {df['safety_score'].mean():.1f}")
    ax.legend()

    # 4. Residuals
    ax = axes[1, 1]
    residuals = y_test - y_pred
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (MAE={mean_absolute_error(y_test, y_pred):.2f})")
    ax.axvline(0, color="r", linestyle="--")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "road_safety_model_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plots to {plot_path}")
    plt.show()


def print_top_dangerous_roads(df, n=20):
    """Print the most dangerous roads by safety score."""
    print(f"\n{'=' * 60}")
    print(f"TOP {n} MOST DANGEROUS ROADS (Winter Snow/Ice)")
    print(f"{'=' * 60}")

    top = df.nlargest(n, "safety_score")
    for i, (_, row) in enumerate(top.iterrows()):
        name = row.get("streetname", "Unknown")
        score = row["safety_score"]
        crash = row.get("_crash_score", 0)
        closure = row.get("_closure_score", 0)
        print(f"  {i+1:2d}. {name:30s} Score: {score:5.1f}  "
              f"(crash={crash:.0f}, closure={closure:.0f})")


def main():
    # Load and prepare
    df, df_feat, target, label_encoders = load_and_prepare_data()

    # Train
    model, X_test, y_test, y_pred = train_model(df_feat, target)

    # Plot
    plot_results(model, df, df_feat, y_test, y_pred)

    # Top dangerous roads
    print_top_dangerous_roads(df)

    # Export model
    print(f"\n{'=' * 60}")
    print("EXPORTING MODEL")
    print(f"{'=' * 60}")
    model.save_model(os.path.join(OUTPUT_DIR, "road_safety_model.json"))
    print(f"Saved {os.path.join(OUTPUT_DIR, 'road_safety_model.json')}")
    model.save_model(os.path.join(OUTPUT_DIR, "road_safety_model.ubj"))
    print(f"Saved {os.path.join(OUTPUT_DIR, 'road_safety_model.ubj')}")

    # Save scored dataset
    scored_path = os.path.join(DATA_DIR, "roads_scored.csv")
    df.to_csv(scored_path, index=False)
    print(f"Saved scored roads to {scored_path}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
