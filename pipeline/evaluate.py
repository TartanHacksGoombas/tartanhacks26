"""
Evaluate both the regression model and the LambdaRank ranker on the temporal
holdout set (2023-24 + 2024-25 seasons).

Reports per-storm ranking metrics that map directly to routing value:
  - NDCG@{50,200,500}
  - Recall@{200,500}  (does the predicted top-K overlap the true top-K?)
  - Mean per-storm Spearman correlation
  - PR-AUC and Recall@K for binary label_any_incident
  - Traditional RMSE/MAE/R² for regression baseline

Outputs: data/model_report.json

Requires: pip install pandas numpy lightgbm scikit-learn scipy pyarrow
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")

TRAINING_PARQUET = os.path.join(DATA_DIR, "training_matrix.parquet")
REGRESSION_MODEL = os.path.join(DATA_DIR, "model_lgbm.txt")
REGRESSION_META = os.path.join(DATA_DIR, "model_metadata.json")
RANKER_MODEL = os.path.join(DATA_DIR, "ranker_lgbm.txt")
RANKER_META = os.path.join(DATA_DIR, "ranker_metadata.json")
REPORT_FILE = os.path.join(DATA_DIR, "model_report.json")

TEST_SEASONS = ["2023-24", "2024-25"]


# ── Metrics ──────────────────────────────────────────────────────────────────

def ndcg_at_k(y_true, y_pred, k):
    """NDCG@k for a single query (storm)."""
    n = min(k, len(y_true))
    order = np.argsort(-y_pred)[:n]
    dcg = np.sum((2 ** y_true[order] - 1) / np.log2(np.arange(2, n + 2)))
    ideal_order = np.argsort(-y_true)[:n]
    idcg = np.sum((2 ** y_true[ideal_order] - 1) / np.log2(np.arange(2, n + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(y_true, y_pred, k):
    """Fraction of the true top-k that appear in predicted top-k."""
    n = min(k, len(y_true))
    true_top = set(np.argsort(-y_true)[:n])
    pred_top = set(np.argsort(-y_pred)[:n])
    return len(true_top & pred_top) / max(len(true_top), 1)


def per_storm_ranking_metrics(y_all, preds_all, storm_ids):
    """Compute per-storm ranking metrics averaged across storms."""
    from scipy.stats import spearmanr

    unique_storms = storm_ids.unique()
    results = {
        "ndcg_50": [], "ndcg_200": [], "ndcg_500": [],
        "recall_200": [], "recall_500": [],
        "spearman": [],
    }

    for sid in unique_storms:
        mask = storm_ids == sid
        y_g = y_all[mask].values
        p_g = preds_all[mask]
        if len(y_g) < 2:
            continue

        results["ndcg_50"].append(ndcg_at_k(y_g, p_g, min(50, len(y_g))))
        results["ndcg_200"].append(ndcg_at_k(y_g, p_g, min(200, len(y_g))))
        results["ndcg_500"].append(ndcg_at_k(y_g, p_g, min(500, len(y_g))))
        results["recall_200"].append(recall_at_k(y_g, p_g, min(200, len(y_g))))
        results["recall_500"].append(recall_at_k(y_g, p_g, min(500, len(y_g))))

        corr, _ = spearmanr(y_g, p_g)
        if not np.isnan(corr):
            results["spearman"].append(corr)

    out = {}
    for key, vals in results.items():
        out[f"mean_{key}"] = round(float(np.mean(vals)), 6) if vals else 0.0
    out["n_storms"] = len(results["ndcg_50"])
    return out


def binary_metrics(y_binary, preds, label_name="label_any_incident"):
    """PR-AUC and recall@K for a binary label."""
    from sklearn.metrics import average_precision_score

    n_pos = int(y_binary.sum())
    if n_pos == 0:
        return {"pr_auc": 0.0, "n_positive": 0, "recall_at_200": 0.0, "recall_at_500": 0.0}

    pr_auc = float(average_precision_score(y_binary, preds))

    # Recall at K: of the segments with label=1, how many are in predicted top-K?
    top_200 = set(np.argsort(-preds)[:200])
    top_500 = set(np.argsort(-preds)[:500])
    pos_idx = set(np.where(y_binary == 1)[0])

    return {
        "pr_auc": round(pr_auc, 6),
        "n_positive": n_pos,
        "recall_at_200": round(len(pos_idx & top_200) / n_pos, 6),
        "recall_at_500": round(len(pos_idx & top_500) / n_pos, 6),
    }


def regression_metrics(y_true, preds):
    """Traditional RMSE / MAE / R²."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    preds_clipped = np.clip(preds, 0, 1)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds_clipped)))
    mae = float(mean_absolute_error(y_true, preds_clipped))
    r2 = float(r2_score(y_true, preds_clipped))
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import lightgbm as lgb

    print("Model Evaluation Report")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    if not os.path.exists(TRAINING_PARQUET):
        print(f"Error: {TRAINING_PARQUET} not found. Run build_training_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(TRAINING_PARQUET)
    test_mask = df["season"].isin(TEST_SEASONS)
    test_df = df[test_mask].copy()

    if test_df.empty:
        print("Error: No test data for seasons", TEST_SEASONS)
        sys.exit(1)

    print(f"  Test set: {len(test_df):,} rows, "
          f"{test_df['storm_id'].nunique()} storms, "
          f"seasons: {sorted(test_df['season'].unique())}")

    report = {"test_seasons": TEST_SEASONS, "n_test_rows": len(test_df)}

    # ── Identify feature columns ──────────────────────────────────────────
    meta_cols = {"objectid", "storm_id", "risk_score", "season",
                 "label_any_incident", "label_top_1pct", "label_top_5pct"}

    # ── Evaluate regression model ─────────────────────────────────────────
    if os.path.exists(REGRESSION_MODEL) and os.path.exists(REGRESSION_META):
        print(f"\n{'='*60}")
        print("Regression model (model_lgbm.txt)")
        print(f"{'='*60}")

        with open(REGRESSION_META) as f:
            reg_meta = json.load(f)
        reg_features = reg_meta["feature_columns"]

        model_reg = lgb.Booster(model_file=REGRESSION_MODEL)
        X_reg = test_df[reg_features]
        preds_reg = model_reg.predict(X_reg)
        preds_reg = np.clip(preds_reg, 0, None)

        reg_report = {}
        reg_report["regression"] = regression_metrics(test_df["risk_score"], preds_reg)
        reg_report["ranking"] = per_storm_ranking_metrics(
            test_df["risk_score"], preds_reg, test_df["storm_id"])

        if "label_any_incident" in test_df.columns:
            reg_report["binary_any_incident"] = binary_metrics(
                test_df["label_any_incident"].fillna(0).astype(int), preds_reg)

        report["regression_model"] = reg_report

        print("  Regression metrics:")
        for k, v in reg_report["regression"].items():
            print(f"    {k}: {v}")
        print("  Ranking metrics:")
        for k, v in reg_report["ranking"].items():
            print(f"    {k}: {v}")
        if "binary_any_incident" in reg_report:
            print("  Binary (label_any_incident):")
            for k, v in reg_report["binary_any_incident"].items():
                print(f"    {k}: {v}")
    else:
        print("\n  Regression model not found, skipping.")

    # ── Evaluate ranker model ─────────────────────────────────────────────
    if os.path.exists(RANKER_MODEL) and os.path.exists(RANKER_META):
        print(f"\n{'='*60}")
        print("Ranker model (ranker_lgbm.txt)")
        print(f"{'='*60}")

        with open(RANKER_META) as f:
            ranker_meta = json.load(f)
        ranker_features = ranker_meta["feature_columns"]

        # Ensure all features present
        X_rank = test_df.reindex(columns=ranker_features, fill_value=0)

        model_rank = lgb.Booster(model_file=RANKER_MODEL)
        preds_rank = model_rank.predict(X_rank)
        preds_rank = np.clip(preds_rank, 0, None)

        rank_report = {}
        rank_report["ranking"] = per_storm_ranking_metrics(
            test_df["risk_score"], preds_rank, test_df["storm_id"])

        if "label_any_incident" in test_df.columns:
            rank_report["binary_any_incident"] = binary_metrics(
                test_df["label_any_incident"].fillna(0).astype(int), preds_rank)

        report["ranker_model"] = rank_report

        print("  Ranking metrics:")
        for k, v in rank_report["ranking"].items():
            print(f"    {k}: {v}")
        if "binary_any_incident" in rank_report:
            print("  Binary (label_any_incident):")
            for k, v in rank_report["binary_any_incident"].items():
                print(f"    {k}: {v}")
    else:
        print("\n  Ranker model not found, skipping.")

    # ── Save report ───────────────────────────────────────────────────────
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Report saved to {REPORT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
