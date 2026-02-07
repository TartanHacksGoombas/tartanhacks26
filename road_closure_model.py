"""
XGBoost model to predict road closure probability.

Uses the prediction-ready dataset built from centerlines, OSM, elevation,
slopes, bridges, landslides, crashes, 311 complaints, plow activity, and
PennDOT route data.

NOTE: No real "is_closed" labels exist yet. Synthetic labels are generated
using the rich feature set as a proxy. Replace with real closure data
(e.g., PennDOT 511, Waze) for production use.

Requires: pip install xgboost scikit-learn pandas matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb


def load_and_prepare_data(csv_path="dataset_prediction_ready.csv"):
    """Load the prediction-ready dataset and prepare features."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {', '.join(df.columns)}")

    # --- Feature columns from the joined dataset ---
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
        # Historical risk
        "winter_crash_count", "winter_crash_fatal",
        "snow_complaint_count", "plow_coverage_score",
    ]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat = df[feature_cols].copy()

    # --- Parse boolean columns ---
    bool_cols = ["near_steep_slope", "near_landslide", "is_bridge"]
    for col in bool_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)

    # --- Parse numeric columns that may have string values ---
    numeric_cols = [
        "speedlimit", "num_lanes", "roadwidth", "elevation_m",
        "bridge_age_years", "penndot_aadt", "penndot_pavement_idx",
        "winter_crash_count", "winter_crash_fatal",
        "snow_complaint_count", "plow_coverage_score",
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

    # Fill remaining NaNs
    df_feat = df_feat.fillna(0)

    print(f"\nFeature matrix: {df_feat.shape[0]} rows x {df_feat.shape[1]} features")
    print(f"Features: {', '.join(df_feat.columns)}")

    return df, df_feat, label_encoders


def generate_synthetic_labels(df):
    """
    Generate synthetic 'is_closed' labels using the rich feature set.

    Closure is more likely for:
    - Steep slopes near landslides
    - Bridges (especially old ones)
    - Low plow coverage
    - High crash/complaint history
    - Narrow residential roads
    Replace with real closure data for production use.
    """
    np.random.seed(42)
    n = len(df)
    probs = np.full(n, 0.05)

    # Road class: local/residential roads close more often
    if "domi_class" in df.columns:
        dc = df["domi_class"].fillna("").astype(str).str.lower()
        probs[dc == "local"] += 0.08
        probs[dc.isin(["arterial", "highway"])] -= 0.03

    # Steep slopes
    if "near_steep_slope" in df.columns:
        steep = df["near_steep_slope"].astype(str).str.lower() == "true"
        probs[steep] += 0.12

    # Landslides
    if "near_landslide" in df.columns:
        slide = df["near_landslide"].astype(str).str.lower() == "true"
        probs[slide] += 0.20

    # Bridges
    if "is_bridge" in df.columns:
        bridge = df["is_bridge"].astype(str).str.lower() == "true"
        probs[bridge] += 0.15

    # Old bridges
    if "bridge_age_years" in df.columns:
        age = pd.to_numeric(df["bridge_age_years"], errors="coerce").fillna(0)
        probs[age > 50] += 0.10

    # Low plow coverage = higher closure risk
    if "plow_coverage_score" in df.columns:
        plow = pd.to_numeric(df["plow_coverage_score"], errors="coerce").fillna(0)
        probs[plow == 0] += 0.06

    # High crash history
    if "winter_crash_count" in df.columns:
        crashes = pd.to_numeric(df["winter_crash_count"], errors="coerce").fillna(0)
        probs[crashes > 5] += 0.08
        probs[crashes > 20] += 0.10

    # High 311 snow complaints
    if "snow_complaint_count" in df.columns:
        complaints = pd.to_numeric(df["snow_complaint_count"], errors="coerce").fillna(0)
        probs[complaints > 10] += 0.05

    # Narrow roads
    if "roadwidth" in df.columns:
        width = pd.to_numeric(df["roadwidth"], errors="coerce").fillna(30)
        probs[width < 20] += 0.08

    # Elevation (high elevation = more snow)
    if "elevation_m" in df.columns:
        elev = pd.to_numeric(df["elevation_m"], errors="coerce").fillna(0)
        probs[elev > 350] += 0.04

    probs = np.clip(probs, 0.01, 0.90)
    labels = np.random.binomial(1, probs)

    print(f"\nSynthetic labels: {labels.sum()} closed / {n} total "
          f"({labels.mean():.1%} closure rate)")
    return labels


def train_model(df_feat, labels):
    """Train XGBoost classifier and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        df_feat, labels, test_size=0.2, random_state=42, stratify=labels,
    )

    print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"Train closure rate: {y_train.mean():.1%}")
    print(f"Test closure rate:  {y_test.mean():.1%}")

    # --- Train XGBoost ---
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 60}")
    print("MODEL EVALUATION")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=["Open", "Closed"]))

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    print(f"\nROC AUC:     {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}  (lower is better, 0 = perfect calibration)")

    # --- Cross-validation ---
    cv_scores = cross_val_score(model, df_feat, labels, cv=5, scoring="roc_auc")
    print(f"\n5-Fold CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model, X_test, y_test, y_prob


def calibrate_model(model, df_feat, labels):
    """Apply probability calibration (Platt scaling + isotonic)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df_feat, labels, test_size=0.2, random_state=42, stratify=labels,
    )

    print(f"\n{'=' * 60}")
    print("PROBABILITY CALIBRATION")
    print(f"{'=' * 60}")

    # Platt scaling (sigmoid)
    cal_sigmoid = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    cal_sigmoid.fit(X_train, y_train)
    y_prob_sigmoid = cal_sigmoid.predict_proba(X_test)[:, 1]

    # Isotonic regression
    cal_isotonic = CalibratedClassifierCV(model, method="isotonic", cv=3)
    cal_isotonic.fit(X_train, y_train)
    y_prob_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]

    # Raw XGBoost probabilities
    y_prob_raw = model.predict_proba(X_test)[:, 1]

    # Compare Brier scores
    brier_raw = brier_score_loss(y_test, y_prob_raw)
    brier_sig = brier_score_loss(y_test, y_prob_sigmoid)
    brier_iso = brier_score_loss(y_test, y_prob_isotonic)

    print(f"Brier Score (raw XGBoost):       {brier_raw:.4f}")
    print(f"Brier Score (Platt scaling):     {brier_sig:.4f}")
    print(f"Brier Score (isotonic):          {brier_iso:.4f}")

    return y_prob_raw, y_prob_sigmoid, y_prob_isotonic, y_test


def plot_results(model, df_feat, y_prob, y_test,
                 y_prob_raw, y_prob_sigmoid, y_prob_isotonic):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Feature importance
    ax = axes[0]
    importance = model.feature_importances_
    feat_names = df_feat.columns
    sorted_idx = np.argsort(importance)[-15:]  # top 15
    ax.barh(range(len(sorted_idx)), importance[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx])
    ax.set_title("Feature Importance (Top 15)")
    ax.set_xlabel("Importance")

    # 2. ROC Curve
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # 3. Calibration curves
    ax = axes[2]
    for probs, label in [
        (y_prob_raw, "Raw XGBoost"),
        (y_prob_sigmoid, "Platt Scaling"),
        (y_prob_isotonic, "Isotonic"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()

    plt.tight_layout()
    plt.savefig("road_closure_model_results.png", dpi=150)
    print("\nSaved plots to road_closure_model_results.png")
    plt.show()


def main():
    # Load data
    df, df_feat, label_encoders = load_and_prepare_data()

    # Generate synthetic labels (replace with real data!)
    labels = generate_synthetic_labels(df)

    # Train and evaluate
    model, X_test, y_test, y_prob = train_model(df_feat, labels)

    # Calibrate probabilities
    y_prob_raw, y_prob_sigmoid, y_prob_isotonic, y_test_cal = calibrate_model(
        model, df_feat, labels
    )

    # Plot
    plot_results(
        model, df_feat, y_prob, y_test,
        y_prob_raw, y_prob_sigmoid, y_prob_isotonic,
    )

    # --- Export model ---
    print(f"\n{'=' * 60}")
    print("EXPORTING MODEL")
    print(f"{'=' * 60}")
    model.save_model("road_closure_model.json")
    print("Saved road_closure_model.json (human-readable)")
    model.save_model("road_closure_model.ubj")
    print("Saved road_closure_model.ubj (binary, smaller)")

    # --- Verdict on probability fitness ---
    brier = brier_score_loss(y_test, y_prob)
    print(f"\n{'=' * 60}")
    print("VERDICT: Is XGBoost good for closure probabilities?")
    print(f"{'=' * 60}")
    print(f"""
XGBoost CAN output probabilities, but they need calibration:

- Raw probabilities tend to be overconfident (pushed toward 0 or 1)
- Platt scaling or isotonic regression improves calibration
- Brier score measures calibration quality (0 = perfect)
- Your Brier score: {brier:.4f}

For this use case (road closure prediction), XGBoost is a solid
choice IF you have real labeled data. The synthetic labels here
are just for pipeline demo purposes.

To improve with real data:
  1. Pull live road closure data (PennDOT 511, Waze, Google Traffic)
  2. Join with these road attributes as features
  3. Add temporal features (time of day, season, weather)
  4. Use CalibratedClassifierCV for well-calibrated probabilities
""")


if __name__ == "__main__":
    main()
