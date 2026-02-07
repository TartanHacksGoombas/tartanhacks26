"""
LSTM regressor to predict the composite Winter Road Safety Score (0-100).

Same task and dataset as road_safety_model.py (XGBoost), but using a small LSTM.
Since the dataset is tabular (no time series), each sample's feature vector is
treated as a pseudo-sequence (seq_len = num_features, input_size = 1).

Evaluation: Stratified 5-Fold CV with MAE, RMSE, R².
Inference: load saved model and score any road segment.

Requires: pip install torch scikit-learn pandas numpy matplotlib
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")

# Model hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 50
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Data loading (mirrors road_safety_model.py)
# ---------------------------------------------------------------------------

def compute_safety_score(df):
    """
    Compute a 0-100 composite winter road safety score from real data.
    Identical logic to road_safety_model.py.
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

    crash_raw = safe_numeric("winter_crash_count") + 10 * safe_numeric("winter_crash_fatal")
    crash_score = scaler.fit_transform(crash_raw.values.reshape(-1, 1)).flatten()

    closure_raw = safe_numeric("domi_closure_count") + 3 * safe_numeric("domi_full_closure_count")
    closure_score = scaler.fit_transform(closure_raw.values.reshape(-1, 1)).flatten()

    complaint_raw = safe_numeric("snow_complaint_count")
    complaint_score = scaler.fit_transform(complaint_raw.values.reshape(-1, 1)).flatten()

    plow = safe_numeric("plow_coverage_score")
    plow_max = plow.max() if plow.max() > 0 else 1
    plow_score = scaler.fit_transform((plow_max - plow).values.reshape(-1, 1)).flatten()

    terrain_raw = (
        safe_bool("near_steep_slope") * 40 +
        safe_bool("near_landslide") * 50 +
        np.clip((safe_numeric("elevation_m") - 250) / 150, 0, 1) * 10
    )
    terrain_score = np.clip(terrain_raw, 0, 100)

    struct_raw = (
        safe_bool("is_bridge") * 50 +
        np.clip(safe_numeric("bridge_age_years") / 100, 0, 1) * 50
    )
    struct_score = np.clip(struct_raw, 0, 100)

    width = safe_numeric("roadwidth")
    width_risk = np.clip((40 - width) / 30, 0, 1) * 50
    speed = safe_numeric("speedlimit")
    speed_risk = np.clip((speed - 25) / 40, 0, 1) * 50
    geo_score = np.clip(width_risk + speed_risk, 0, 100)

    pave_idx = safe_numeric("penndot_pavement_idx")
    pave_max = pave_idx.max() if pave_idx.max() > 0 else 1
    pave_inv = pave_max - pave_idx
    has_pave = pave_idx > 0
    pave_score = np.zeros(len(df))
    if has_pave.any():
        pave_score[has_pave] = scaler.fit_transform(
            pave_inv[has_pave].values.reshape(-1, 1)
        ).flatten()

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


def load_and_prepare_data():
    """Load the prediction-ready dataset and prepare features + safety score."""
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    print(f"\n{'=' * 60}")
    print("COMPUTING COMPOSITE SAFETY SCORE")
    print(f"{'=' * 60}")
    df = compute_safety_score(df)

    feature_cols = [
        "domi_class", "paveclass", "speedlimit", "num_lanes", "roadwidth",
        "oneway", "highway_type_osm", "surface_osm",
        "mid_lat", "mid_lng",
        "elevation_m", "near_steep_slope", "near_landslide",
        "is_bridge", "bridge_age_years",
        "penndot_aadt", "penndot_pavement_idx",
        "is_snow_emergency_route",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat = df[feature_cols].copy()

    # Boolean columns
    bool_cols = ["near_steep_slope", "near_landslide", "is_bridge", "is_snow_emergency_route"]
    for col in bool_cols:
        if col in df_feat.columns:
            df_feat[col] = (
                df_feat[col].astype(str).str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0).astype(int)
            )

    # Numeric columns
    numeric_cols = [
        "speedlimit", "num_lanes", "roadwidth", "elevation_m",
        "bridge_age_years", "penndot_aadt", "penndot_pavement_idx",
        "mid_lat", "mid_lng",
    ]
    for col in numeric_cols:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    # Categorical columns — encode and store mappings for inference
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


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class RoadSafetyLSTM(nn.Module):
    """
    Small LSTM for tabular regression.

    Architecture: the raw feature vector is projected through a linear
    embedding layer, then chunked into a short sequence of `seq_len` steps
    (each of width `embed_dim // seq_len`).  The LSTM processes that
    sequence and the final hidden state feeds a regression head.

    This lets the LSTM see meaningful multi-feature "tokens" at each step
    rather than individual scalars, enabling it to learn interactions.
    """

    SEQ_LEN = 4  # number of pseudo time-steps

    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        # Embedding: project raw features into a space divisible by SEQ_LEN
        embed_dim = hidden_size * self.SEQ_LEN  # e.g. 256
        self.embed = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim // self.SEQ_LEN,  # e.g. 64
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, num_features)
        x = self.embed(x)                             # (batch, embed_dim)
        x = x.view(x.size(0), self.SEQ_LEN, -1)       # (batch, SEQ_LEN, embed_dim // SEQ_LEN)
        _, (h_n, _) = self.lstm(x)                     # h_n: (num_layers, batch, hidden)
        out = self.head(h_n[-1])                       # (batch, 1)
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# 5-Fold CV
# ---------------------------------------------------------------------------

def run_cv(df_feat_np, target, num_features):
    """5-fold cross-validation for the LSTM regressor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_feat_np)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

        X_train, X_val = df_feat_np[train_idx], df_feat_np[val_idx]
        y_train, y_val = target[train_idx], target[val_idx]

        # Per-fold standard scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = RoadSafetyLSTM(
            num_features=num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # Validation loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    val_loss += criterion(model(X_b), y_b).item() * len(y_b)
            val_loss /= len(val_ds)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}")

            if patience_counter >= 10:
                print(f"  Early stop at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        y_pred = predict(model, val_loader, device)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        print(f"  Fold {fold+1}  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.4f}")

        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    print(f"\n{'=' * 60}")
    print("LSTM 5-FOLD CV RESULTS")
    print(f"{'=' * 60}")
    print(f"MAE:  {np.mean(fold_maes):.2f} (+/- {np.std(fold_maes):.2f})")
    print(f"RMSE: {np.mean(fold_rmses):.2f} (+/- {np.std(fold_rmses):.2f})")
    print(f"R²:   {np.mean(fold_r2s):.4f} (+/- {np.std(fold_r2s):.4f})")

    return all_y_true, all_y_pred, fold_maes, fold_rmses, fold_r2s


# ---------------------------------------------------------------------------
# Train final model on all data (for export + inference)
# ---------------------------------------------------------------------------

def train_final_model(df_feat_np, target, num_features):
    """Train on full dataset and save for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_feat_np)

    ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = RoadSafetyLSTM(
        num_features=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    print(f"\n{'=' * 60}")
    print("TRAINING FINAL MODEL (full dataset)")
    print(f"{'=' * 60}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}  loss={loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, scaler


def save_model(model, scaler, feature_cols, label_encoders):
    """Save model weights, scaler params, and feature metadata for inference."""
    # PyTorch weights
    weights_path = os.path.join(OUTPUT_DIR, "road_safety_lstm.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

    # Metadata needed for inference (scaler params, feature names, label encoders)
    meta = {
        "feature_cols": list(feature_cols),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "label_encoders": {
            col: list(le.classes_) for col, le in label_encoders.items()
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, "road_safety_lstm_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved inference metadata to {meta_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model_for_inference():
    """Load saved LSTM model and metadata for inference."""
    meta_path = os.path.join(OUTPUT_DIR, "road_safety_lstm_meta.json")
    weights_path = os.path.join(OUTPUT_DIR, "road_safety_lstm.pt")

    with open(meta_path) as f:
        meta = json.load(f)

    model = RoadSafetyLSTM(
        num_features=len(meta["feature_cols"]),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["scaler_mean"])
    scaler.scale_ = np.array(meta["scaler_scale"])
    scaler.n_features_in_ = len(meta["feature_cols"])

    label_encoders = {}
    for col, classes in meta["label_encoders"].items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        label_encoders[col] = le

    return model, scaler, meta["feature_cols"], label_encoders


def predict_safety_scores(df, model=None, scaler=None, feature_cols=None,
                          label_encoders=None):
    """
    Score road segments using the saved LSTM model.

    Args:
        df: DataFrame with the same columns as dataset_prediction_ready.csv
        model, scaler, feature_cols, label_encoders: pass all or none (auto-loads)

    Returns:
        numpy array of predicted safety scores (0-100)
    """
    if model is None:
        model, scaler, feature_cols, label_encoders = load_model_for_inference()

    df_feat = df[feature_cols].copy()

    # Boolean columns
    bool_cols = ["near_steep_slope", "near_landslide", "is_bridge", "is_snow_emergency_route"]
    for col in bool_cols:
        if col in df_feat.columns:
            df_feat[col] = (
                df_feat[col].astype(str).str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0).astype(int)
            )

    # Numeric columns
    numeric_cols = [
        "speedlimit", "num_lanes", "roadwidth", "elevation_m",
        "bridge_age_years", "penndot_aadt", "penndot_pavement_idx",
        "mid_lat", "mid_lng",
    ]
    for col in numeric_cols:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    # Categorical columns
    cat_cols = ["domi_class", "paveclass", "oneway", "highway_type_osm", "surface_osm"]
    for col in cat_cols:
        if col in df_feat.columns and col in label_encoders:
            df_feat[col] = df_feat[col].fillna("unknown").astype(str)
            le = label_encoders[col]
            # Handle unseen categories gracefully
            known = set(le.classes_)
            df_feat[col] = df_feat[col].apply(
                lambda v: v if v in known else "unknown"
            )
            df_feat[col] = le.transform(df_feat[col])

    df_feat = df_feat.fillna(0)

    X = scaler.transform(df_feat.values.astype(np.float32))
    X_tensor = torch.tensor(X, dtype=torch.float32)

    ds = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    preds = predict(model, loader, "cpu")
    return np.clip(preds, 0, 100)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(df, y_true, y_pred, fold_maes):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=5)
    ax.plot([0, 100], [0, 100], "r--", label="Perfect")
    ax.set_xlabel("Actual Safety Score")
    ax.set_ylabel("Predicted Safety Score")
    ax.set_title("Predicted vs Actual (Aggregated CV)")
    ax.legend()

    # 2. Per-fold MAE
    ax = axes[0, 1]
    ax.bar(range(1, len(fold_maes) + 1), fold_maes)
    ax.axhline(np.mean(fold_maes), color="red", linestyle="--",
               label=f"Mean={np.mean(fold_maes):.2f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Fold MAE")
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
    residuals = y_true - y_pred
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_title(f"Residual Distribution (MAE={mae:.2f})")
    ax.axvline(0, color="r", linestyle="--")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "road_safety_lstm_results.png")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load and prepare
    df, df_feat, target, label_encoders = load_and_prepare_data()
    df_feat_np = df_feat.values.astype(np.float32)
    num_features = df_feat_np.shape[1]
    feature_cols = list(df_feat.columns)

    # 5-fold CV evaluation
    y_true, y_pred, fold_maes, fold_rmses, fold_r2s = run_cv(
        df_feat_np, target, num_features,
    )

    # Plot CV results
    plot_results(df, y_true, y_pred, fold_maes)

    # Train final model on all data
    model, scaler = train_final_model(df_feat_np, target, num_features)

    # Save model + metadata
    save_model(model, scaler, feature_cols, label_encoders)

    # Score all roads with the final model and save
    print(f"\n{'=' * 60}")
    print("SCORING ALL ROADS WITH LSTM")
    print(f"{'=' * 60}")
    lstm_scores = predict_safety_scores(
        df, model=model, scaler=scaler,
        feature_cols=feature_cols, label_encoders=label_encoders,
    )
    df["lstm_safety_score"] = lstm_scores

    scored_path = os.path.join(DATA_DIR, "roads_scored_lstm.csv")
    df.to_csv(scored_path, index=False)
    print(f"Saved scored roads to {scored_path}")

    # Top dangerous roads
    print_top_dangerous_roads(df)

    # Summary
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
LSTM 5-Fold CV Results:
  MAE:  {overall_mae:.2f}
  RMSE: {overall_rmse:.2f}
  R²:   {overall_r2:.4f}

Compare with XGBoost results from road_safety_model.py.
Tree-based models typically outperform LSTMs on tabular data,
but the LSTM provides an alternative perspective.

Saved:
  models/road_safety_lstm.pt          — model weights
  models/road_safety_lstm_meta.json   — scaler + feature metadata
  models/road_safety_lstm_results.png — evaluation plots
  data/roads_scored_lstm.csv          — all roads with LSTM scores
""")


if __name__ == "__main__":
    main()
