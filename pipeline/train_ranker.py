"""
Train a LightGBM LambdaRank model grouped by storm_id.

The ranking objective directly optimises the order of road segments within each
storm — which is exactly what the downstream routing app needs (identify the K
most dangerous roads, not predict exact risk scores).

Weather features are now *included* so the model can learn weather × segment
interactions (e.g., steep roads become disproportionately dangerous in heavy
snow).

Evaluation: per-storm NDCG@{50,200}, mean Spearman, Recall@{200,500}.

Outputs:
    - data/ranker_lgbm.txt   (LightGBM model file)
    - data/ranker_metadata.json
    - data/ranker_feature_importance.csv

Requires: pip install pandas numpy lightgbm scikit-learn scipy pyarrow
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
MODEL_FILE = os.path.join(DATA_DIR, "ranker_lgbm.txt")
METADATA_FILE = os.path.join(DATA_DIR, "ranker_metadata.json")
IMPORTANCE_CSV = os.path.join(DATA_DIR, "ranker_feature_importance.csv")

# Seasons held out for temporal evaluation
TEST_SEASONS = ["2023-24", "2024-25"]

# LightGBM LambdaRank hyperparameters
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [50, 200],
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

NUM_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.1
MAX_DOCS_PER_QUERY = 9000  # LightGBM LambdaRank hard limit is 10,000 per query


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_group_sizes(storm_ids):
    """Return an array of group sizes (one per query/storm) in order."""
    current_sid = storm_ids.iloc[0]
    sizes = []
    count = 0
    for sid in storm_ids:
        if sid == current_sid:
            count += 1
        else:
            sizes.append(count)
            current_sid = sid
            count = 1
    sizes.append(count)
    return np.array(sizes, dtype=np.int32)


def subsample_queries(df, label_col="risk_score", max_per_query=MAX_DOCS_PER_QUERY, seed=42):
    """Subsample each query (storm) to fit LightGBM's per-query row limit.

    Strategy: keep as many positive segments as possible (capped at
    max_per_query), then fill remaining slots with negatives.
    """
    rng = np.random.RandomState(seed)
    dfs = []
    for sid, group in df.groupby("storm_id"):
        if len(group) <= max_per_query:
            dfs.append(group)
            continue
        positive = group[group[label_col] > 0]
        negative = group[group[label_col] == 0]

        if len(positive) >= max_per_query:
            # Too many positives — sample down, no negatives
            dfs.append(positive.sample(n=max_per_query, random_state=rng))
        else:
            # Keep all positives, fill with random negatives
            n_neg = max_per_query - len(positive)
            if len(negative) > n_neg:
                negative = negative.sample(n=n_neg, random_state=rng)
            dfs.append(pd.concat([positive, negative]))

    return pd.concat(dfs).sort_values("storm_id").reset_index(drop=True)


def discretize_labels(df, label_col="risk_score", n_grades=5):
    """Convert continuous risk_score to integer relevance grades for LambdaRank.

    Grades:
        0 = no risk (score == 0)
        1-4 = quartile bins of nonzero risk scores
    """
    scores = df[label_col].values
    nonzero = scores[scores > 0]
    if len(nonzero) == 0:
        df["relevance"] = 0
        return df

    # Compute quartile thresholds among nonzero scores
    thresholds = np.percentile(nonzero, [25, 50, 75])
    grades = np.zeros(len(scores), dtype=np.int32)
    for i, s in enumerate(scores):
        if s <= 0:
            grades[i] = 0
        elif s <= thresholds[0]:
            grades[i] = 1
        elif s <= thresholds[1]:
            grades[i] = 2
        elif s <= thresholds[2]:
            grades[i] = 3
        else:
            grades[i] = n_grades - 1  # = 4

    df["relevance"] = grades
    grade_counts = {g: int((grades == g).sum()) for g in range(n_grades)}
    print(f"  Relevance grade distribution: {grade_counts}")
    return df


def split_data(df, feature_cols, label_col="risk_score", season_col="season"):
    """Split: temporal test on 2023-25, random storm-level val on rest."""
    test_mask = df[season_col].isin(TEST_SEASONS)
    non_test = df[~test_mask]

    storm_ids = non_test["storm_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(storm_ids)
    n_val = max(1, int(len(storm_ids) * VAL_FRACTION))
    val_storm_ids = set(storm_ids[:n_val])
    train_storm_ids = set(storm_ids[n_val:])

    train_mask = (~test_mask) & (df["storm_id"].isin(train_storm_ids))
    val_mask = (~test_mask) & (df["storm_id"].isin(val_storm_ids))

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub = df[mask].copy()
        # Subsample large query groups for LambdaRank
        sub = subsample_queries(sub, label_col=label_col)
        X = sub[feature_cols]
        y_rank = sub["relevance"]  # integer relevance grades for LambdaRank
        y_score = sub[label_col]   # continuous scores for custom eval
        groups = build_group_sizes(sub["storm_id"])
        splits[name] = (X, y_rank, y_score, groups, sub)
        n_storms = sub["storm_id"].nunique()
        seasons = sorted(sub[season_col].unique())
        print(f"  {name:5s}: {len(sub):>9,} rows, {n_storms:>3d} storms "
              f"(seasons: {', '.join(seasons)})")

    return splits, {
        "train_storms": int(len(train_storm_ids)),
        "val_storms": int(n_val),
        "test_seasons": TEST_SEASONS,
    }


def ndcg_at_k(y_true, y_pred, k):
    """Compute NDCG@k for a single query."""
    order = np.argsort(-y_pred)[:k]
    dcg = np.sum((2 ** y_true[order] - 1) / np.log2(np.arange(2, k + 2)[:len(order)]))
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum((2 ** y_true[ideal_order] - 1) / np.log2(np.arange(2, k + 2)[:len(ideal_order)]))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(y_true, y_pred, k):
    """Fraction of the true top-k that appear in the predicted top-k."""
    true_top = set(np.argsort(-y_true)[:k])
    pred_top = set(np.argsort(-y_pred)[:k])
    return len(true_top & pred_top) / max(len(true_top), 1)


def evaluate_ranking(model, X, y, groups, label=""):
    """Compute per-storm ranking metrics."""
    from scipy.stats import spearmanr

    preds = model.predict(X)
    preds = np.clip(preds, 0, None)

    # Per-storm metrics
    ndcg_50_list, ndcg_200_list = [], []
    recall_200_list, recall_500_list = [], []
    spearman_list = []

    offset = 0
    for size in groups:
        y_g = y.values[offset:offset + size]
        p_g = preds[offset:offset + size]
        offset += size

        if len(y_g) < 2:
            continue

        ndcg_50_list.append(ndcg_at_k(y_g, p_g, min(50, len(y_g))))
        ndcg_200_list.append(ndcg_at_k(y_g, p_g, min(200, len(y_g))))
        recall_200_list.append(recall_at_k(y_g, p_g, min(200, len(y_g))))
        recall_500_list.append(recall_at_k(y_g, p_g, min(500, len(y_g))))

        corr, _ = spearmanr(y_g, p_g)
        if not np.isnan(corr):
            spearman_list.append(corr)

    metrics = {
        "mean_ndcg_50": round(float(np.mean(ndcg_50_list)), 6) if ndcg_50_list else 0.0,
        "mean_ndcg_200": round(float(np.mean(ndcg_200_list)), 6) if ndcg_200_list else 0.0,
        "mean_recall_200": round(float(np.mean(recall_200_list)), 6) if recall_200_list else 0.0,
        "mean_recall_500": round(float(np.mean(recall_500_list)), 6) if recall_500_list else 0.0,
        "mean_spearman": round(float(np.mean(spearman_list)), 6) if spearman_list else 0.0,
        "n_storms": len(ndcg_50_list),
        "n_samples": int(len(y)),
    }

    if label:
        print(f"\n  {label}:")
    print(f"    NDCG@50:      {metrics['mean_ndcg_50']:.6f}")
    print(f"    NDCG@200:     {metrics['mean_ndcg_200']:.6f}")
    print(f"    Recall@200:   {metrics['mean_recall_200']:.6f}")
    print(f"    Recall@500:   {metrics['mean_recall_500']:.6f}")
    print(f"    Spearman:     {metrics['mean_spearman']:.6f}")
    return metrics


def get_feature_importance(model, feature_names):
    """Extract and sort feature importance."""
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    total = imp_df["importance"].sum()
    imp_df["importance_pct"] = (imp_df["importance"] / total * 100).round(2) if total > 0 else 0
    return imp_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import lightgbm as lgb

    print("Training LightGBM LambdaRank model...")
    print("=" * 60)

    # ── Load training matrix ──────────────────────────────────────────────
    print("\nLoading training matrix...")
    if not os.path.exists(TRAINING_PARQUET):
        print(f"Error: {TRAINING_PARQUET} not found. Run build_training_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(TRAINING_PARQUET)
    print(f"  Shape: {df.shape}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # ── Identify feature columns ──────────────────────────────────────────
    # Include weather features so the model can learn weather × segment
    # interactions (unlike the old regression model which excluded weather).
    meta_cols = {"objectid", "storm_id", "risk_score", "season",
                 "label_any_incident", "label_top_1pct", "label_top_5pct",
                 "relevance"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    # ── Discretize labels for LambdaRank ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Discretizing risk scores into relevance grades...")
    print(f"{'='*60}")
    df = discretize_labels(df)

    # ── Split data ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Splitting data (storm-level random val + temporal test)...")
    print(f"{'='*60}")
    splits, split_info = split_data(df, feature_cols)

    X_train, y_train, y_train_score, g_train, _ = splits["train"]
    X_val, y_val, y_val_score, g_val, _ = splits["val"]
    X_test, y_test, y_test_score, g_test, _ = splits["test"]

    if len(X_train) == 0:
        print("Error: Training set is empty.")
        sys.exit(1)

    # ── Create LightGBM datasets ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training LambdaRank...")
    print(f"{'='*60}")

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train)
    dval = lgb.Dataset(X_val, label=y_val, group=g_val, reference=dtrain) if len(X_val) > 0 else None

    callbacks = [lgb.log_evaluation(period=100)]
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

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluation (per-storm ranking metrics)...")
    print(f"{'='*60}")

    metrics = {}
    # Evaluate against continuous risk_score (better for Spearman/Recall metrics)
    metrics["train"] = evaluate_ranking(model, X_train, y_train_score, g_train, "Training set")
    if len(X_val) > 0:
        metrics["val"] = evaluate_ranking(model, X_val, y_val_score, g_val, "Validation set")
    if len(X_test) > 0:
        metrics["test"] = evaluate_ranking(model, X_test, y_test_score, g_test, "Test set (temporal)")

    # ── Feature importance ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Feature importance (top 15)...")
    print(f"{'='*60}")

    imp_df = get_feature_importance(model, feature_cols)
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:35s} : {row['importance_pct']:6.2f}%")

    imp_df.to_csv(IMPORTANCE_CSV, index=False)
    print(f"\n  Saved full importance to {IMPORTANCE_CSV}")

    # ── Save model ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")

    model.save_model(MODEL_FILE)
    print(f"  Model saved to {MODEL_FILE}")

    # Compute calibration statistics from training predictions
    # (used at prediction time for absolute score calibration)
    train_preds = model.predict(X_train)
    cal_p50 = float(np.percentile(train_preds, 50))
    cal_p75 = float(np.percentile(train_preds, 75))
    cal_p99 = float(np.percentile(train_preds, 99))
    cal_mean = float(np.mean(train_preds))
    cal_std = float(np.std(train_preds))
    print(f"  Calibration stats: mean={cal_mean:.4f}, std={cal_std:.4f}, "
          f"p75={cal_p75:.4f}, p99={cal_p99:.4f}")

    metadata = {
        "model_type": "lightgbm_lambdarank",
        "params": {k: v for k, v in PARAMS.items()},
        "num_boost_round": NUM_ROUNDS,
        "best_iteration": best_iter,
        "train_time_seconds": round(train_time, 1),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "calibration": {
            "p50": round(cal_p50, 6),
            "p75": round(cal_p75, 6),
            "p99": round(cal_p99, 6),
            "mean": round(cal_mean, 6),
            "std": round(cal_std, 6),
        },
        "split": {
            **split_info,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "metrics": metrics,
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {METADATA_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RANKER SUMMARY")
    print(f"{'='*60}")
    print(f"  Features: {len(feature_cols)} (including weather)")
    print(f"  Training rows: {len(X_train):,}")
    print(f"  Best iteration: {best_iter}")
    if "test" in metrics:
        print(f"  Test NDCG@50:    {metrics['test']['mean_ndcg_50']:.6f}")
        print(f"  Test NDCG@200:   {metrics['test']['mean_ndcg_200']:.6f}")
        print(f"  Test Recall@200: {metrics['test']['mean_recall_200']:.6f}")
        print(f"  Test Spearman:   {metrics['test']['mean_spearman']:.6f}")


if __name__ == "__main__":
    main()
