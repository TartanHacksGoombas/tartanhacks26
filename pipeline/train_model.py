"""
Train a LightGBM regression model to predict road closure risk scores.

Strategy:
    - Train on 90% of data (random split by storm_id to avoid segment leakage)
    - Validate on 10% holdout for early stopping
    - Evaluate temporal generalization on held-out 2023-24 + 2024-25 seasons
    - Strong regularization to prevent overfitting to historical noise

Evaluation: RMSE, MAE, R², Spearman correlation, Precision/Recall@0.5

Outputs:
    - data/model_lgbm.txt (LightGBM model file)
    - data/model_metadata.json (hyperparameters, metrics, split info)
    - data/feature_importance.csv (top features)

Requires: pip install pandas numpy lightgbm scikit-learn
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Input
TRAINING_PARQUET = os.path.join(DATA_DIR, "training_matrix.parquet")

# Outputs
MODEL_FILE = os.path.join(DATA_DIR, "model_lgbm.txt")
METADATA_FILE = os.path.join(DATA_DIR, "model_metadata.json")
IMPORTANCE_CSV = os.path.join(DATA_DIR, "feature_importance.csv")

# Seasons held out for temporal evaluation
TEST_SEASONS = ["2023-24", "2024-25"]

# LightGBM hyperparameters — strong regularization for noisy proxy labels
PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 200,
    "min_data_in_leaf": 200,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "max_depth": 7,
    "verbose": -1,
}

NUM_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 50

# Random holdout fraction for early stopping (split by storm_id)
VAL_FRACTION = 0.1


def split_data(df, feature_cols, label_col="risk_score", season_col="season"):
    """Split data: random storm-level holdout for validation, temporal test set.

    - Test: 2023-24 + 2024-25 seasons (temporal holdout)
    - Val: 10% of remaining storms (random, for early stopping)
    - Train: remaining 90% of non-test storms
    """
    test_mask = df[season_col].isin(TEST_SEASONS)
    non_test = df[~test_mask]

    # Split non-test by storm_id (not by row) to avoid segment leakage
    storm_ids = non_test["storm_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(storm_ids)
    n_val = max(1, int(len(storm_ids) * VAL_FRACTION))
    val_storm_ids = set(storm_ids[:n_val])
    train_storm_ids = set(storm_ids[n_val:])

    train_mask = (~test_mask) & (df["storm_id"].isin(train_storm_ids))
    val_mask = (~test_mask) & (df["storm_id"].isin(val_storm_ids))

    splits = {
        "train": (df[train_mask][feature_cols], df[train_mask][label_col]),
        "val": (df[val_mask][feature_cols], df[val_mask][label_col]),
        "test": (df[test_mask][feature_cols], df[test_mask][label_col]),
    }

    train_seasons = sorted(df[train_mask][season_col].unique())
    val_seasons = sorted(df[val_mask][season_col].unique())
    test_seasons = sorted(df[test_mask][season_col].unique())

    print(f"  Train: {train_mask.sum():,} rows ({len(train_storm_ids)} storms, "
          f"seasons: {', '.join(train_seasons)})")
    print(f"  Val:   {val_mask.sum():,} rows ({len(val_storm_ids)} storms, "
          f"seasons: {', '.join(val_seasons)})")
    print(f"  Test:  {test_mask.sum():,} rows (seasons: {', '.join(test_seasons)})")

    # Warn if any split is empty
    for name, (X, y) in splits.items():
        if len(X) == 0:
            print(f"  WARNING: {name} split is empty!")

    return splits, {
        "train_storms": len(train_storm_ids),
        "val_storms": len(val_storm_ids),
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "test_seasons": test_seasons,
    }


def evaluate_model(model, X, y, label=""):
    """Compute evaluation metrics on a dataset."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    preds = model.predict(X)
    preds = np.clip(preds, 0, 1)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    spearman_corr, spearman_p = spearmanr(y, preds)

    # Binary precision/recall at threshold 0.5
    y_bin = (y >= 0.5).astype(int)
    pred_bin = (preds >= 0.5).astype(int)

    tp = ((y_bin == 1) & (pred_bin == 1)).sum()
    fp = ((y_bin == 0) & (pred_bin == 1)).sum()
    fn = ((y_bin == 1) & (pred_bin == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "rmse": round(float(rmse), 6),
        "mae": round(float(mae), 6),
        "r2": round(float(r2), 6),
        "spearman_corr": round(float(spearman_corr), 6),
        "spearman_p": float(spearman_p),
        "precision_at_0.5": round(float(precision), 4),
        "recall_at_0.5": round(float(recall), 4),
        "n_samples": int(len(y)),
    }

    if label:
        print(f"\n  {label}:")
    print(f"    RMSE:      {rmse:.6f}")
    print(f"    MAE:       {mae:.6f}")
    print(f"    R²:        {r2:.6f}")
    print(f"    Spearman:  {spearman_corr:.6f} (p={spearman_p:.2e})")
    print(f"    P@0.5:     {precision:.4f}")
    print(f"    R@0.5:     {recall:.4f}")

    return metrics


def get_feature_importance(model, feature_names):
    """Extract and sort feature importance."""
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["importance_pct"] = (imp_df["importance"] / imp_df["importance"].sum() * 100).round(2)
    return imp_df


def main():
    import lightgbm as lgb

    print("Training LightGBM model...")
    print("=" * 60)

    # ── Load training matrix ──────────────────────────────────────────────────
    print("\nLoading training matrix...")
    if not os.path.exists(TRAINING_PARQUET):
        print(f"Error: {TRAINING_PARQUET} not found. Run build_training_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(TRAINING_PARQUET)
    print(f"  Shape: {df.shape}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # ── Identify feature columns ──────────────────────────────────────────────
    # Exclude weather features from model training — they are the same for all
    # segments in a given storm and dominate the model, preventing it from
    # learning segment-level risk differentiation. Weather severity is applied
    # as a multiplier at prediction time instead.
    meta_cols = {"objectid", "storm_id", "risk_score", "season"}
    weather_cols = {"total_snowfall_cm", "max_daily_snowfall_cm", "min_temp_c",
                    "max_wind_kmh", "duration_days", "precip_total_mm"}
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in weather_cols]
    print(f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    # ── Quick data quality check ──────────────────────────────────────────────
    print(f"\n  Risk score stats by season:")
    for season in sorted(df["season"].unique()):
        s = df[df["season"] == season]["risk_score"]
        print(f"    {season}: mean={s.mean():.4f} std={s.std():.4f} "
              f"n={len(s):,} storms={df[df['season']==season]['storm_id'].nunique()}")

    # ── Split data ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Splitting data (storm-level random val + temporal test)...")
    print(f"{'='*60}")
    splits, split_info = split_data(df, feature_cols)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    if len(X_train) == 0:
        print("Error: Training set is empty. Check season labels in data.")
        sys.exit(1)

    # ── Create LightGBM datasets ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training LightGBM...")
    print(f"{'='*60}")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain) if len(X_val) > 0 else None

    # ── Train ─────────────────────────────────────────────────────────────────
    callbacks = [
        lgb.log_evaluation(period=100),
    ]
    if dval is not None:
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS))

    valid_sets = [dtrain]
    valid_names = ["train"]
    if dval is not None:
        valid_sets.append(dval)
        valid_names.append("val")

    start_time = time.time()

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    train_time = time.time() - start_time
    best_iter = model.best_iteration if model.best_iteration > 0 else NUM_ROUNDS
    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Best iteration: {best_iter}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluation...")
    print(f"{'='*60}")

    metrics = {}
    metrics["train"] = evaluate_model(model, X_train, y_train, "Training set")
    if len(X_val) > 0:
        metrics["val"] = evaluate_model(model, X_val, y_val, "Validation set")
    if len(X_test) > 0:
        metrics["test"] = evaluate_model(model, X_test, y_test, "Test set (temporal)")
    else:
        print("\n  Test set: empty (no data for test seasons)")

    # ── Feature importance ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Feature importance (top 15)...")
    print(f"{'='*60}")

    imp_df = get_feature_importance(model, feature_cols)
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:35s} : {row['importance_pct']:6.2f}%")

    imp_df.to_csv(IMPORTANCE_CSV, index=False)
    print(f"\n  Saved full importance to {IMPORTANCE_CSV}")

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")

    model.save_model(MODEL_FILE)
    print(f"  Model saved to {MODEL_FILE}")

    # Save metadata
    metadata = {
        "model_type": "lightgbm_regression",
        "params": PARAMS,
        "num_boost_round": NUM_ROUNDS,
        "best_iteration": best_iter,
        "train_time_seconds": round(train_time, 1),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "split": {
            **split_info,
            "test_seasons": TEST_SEASONS,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "metrics": metrics,
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {METADATA_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training rows: {len(X_train):,}")
    print(f"  Best iteration: {best_iter}")
    if "test" in metrics:
        print(f"  Test RMSE: {metrics['test']['rmse']:.6f}")
        print(f"  Test Spearman: {metrics['test']['spearman_corr']:.6f}")
    if "val" in metrics:
        print(f"  Val RMSE: {metrics['val']['rmse']:.6f}")
        print(f"  Val Spearman: {metrics['val']['spearman_corr']:.6f}")


if __name__ == "__main__":
    main()
