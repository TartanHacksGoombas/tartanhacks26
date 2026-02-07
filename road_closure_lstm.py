"""
LSTM model to predict road closure probability.

Uses the same prediction-ready dataset and synthetic labels as road_closure_model.py.
Since the dataset is tabular (no real time series), each sample's features are treated
as a pseudo-sequence (sequence_length = num_features, input_size = 1).

Evaluation: Stratified 5-Fold CV with ROC AUC, Brier score, classification report.
(LOOCV is impractical for neural networks on ~28k samples.)

Requires: pip install torch scikit-learn pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    brier_score_loss,
)


# ---------------------------------------------------------------------------
# Data loading (mirrors road_closure_model.py)
# ---------------------------------------------------------------------------

def load_and_prepare_data(csv_path="dataset_prediction_ready.csv"):
    """Load the prediction-ready dataset and prepare features."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    feature_cols = [
        "domi_class", "paveclass", "speedlimit", "num_lanes", "roadwidth",
        "oneway", "highway_type_osm", "surface_osm",
        "mid_lat", "mid_lng",
        "elevation_m", "near_steep_slope", "near_landslide",
        "is_bridge", "bridge_age_years",
        "penndot_aadt", "penndot_pavement_idx",
        "winter_crash_count", "winter_crash_fatal",
        "snow_complaint_count", "plow_coverage_score",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_feat = df[feature_cols].copy()

    # Boolean columns
    for col in ["near_steep_slope", "near_landslide", "is_bridge"]:
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
        "winter_crash_count", "winter_crash_fatal",
        "snow_complaint_count", "plow_coverage_score",
        "mid_lat", "mid_lng",
    ]
    for col in numeric_cols:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    # Categorical columns
    cat_cols = ["domi_class", "paveclass", "oneway", "highway_type_osm", "surface_osm"]
    for col in cat_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna("unknown").astype(str)
            le = LabelEncoder()
            df_feat[col] = le.fit_transform(df_feat[col])

    df_feat = df_feat.fillna(0)
    print(f"Feature matrix: {df_feat.shape[0]} rows x {df_feat.shape[1]} features")
    return df, df_feat


def generate_synthetic_labels(df):
    """Same synthetic label logic as road_closure_model.py."""
    np.random.seed(42)
    n = len(df)
    probs = np.full(n, 0.05)

    if "domi_class" in df.columns:
        dc = df["domi_class"].fillna("").astype(str).str.lower()
        probs[dc == "local"] += 0.08
        probs[dc.isin(["arterial", "highway"])] -= 0.03

    if "near_steep_slope" in df.columns:
        steep = df["near_steep_slope"].astype(str).str.lower() == "true"
        probs[steep] += 0.12

    if "near_landslide" in df.columns:
        slide = df["near_landslide"].astype(str).str.lower() == "true"
        probs[slide] += 0.20

    if "is_bridge" in df.columns:
        bridge = df["is_bridge"].astype(str).str.lower() == "true"
        probs[bridge] += 0.15

    if "bridge_age_years" in df.columns:
        age = pd.to_numeric(df["bridge_age_years"], errors="coerce").fillna(0)
        probs[age > 50] += 0.10

    if "plow_coverage_score" in df.columns:
        plow = pd.to_numeric(df["plow_coverage_score"], errors="coerce").fillna(0)
        probs[plow == 0] += 0.06

    if "winter_crash_count" in df.columns:
        crashes = pd.to_numeric(df["winter_crash_count"], errors="coerce").fillna(0)
        probs[crashes > 5] += 0.08
        probs[crashes > 20] += 0.10

    if "snow_complaint_count" in df.columns:
        complaints = pd.to_numeric(df["snow_complaint_count"], errors="coerce").fillna(0)
        probs[complaints > 10] += 0.05

    if "roadwidth" in df.columns:
        width = pd.to_numeric(df["roadwidth"], errors="coerce").fillna(30)
        probs[width < 20] += 0.08

    if "elevation_m" in df.columns:
        elev = pd.to_numeric(df["elevation_m"], errors="coerce").fillna(0)
        probs[elev > 350] += 0.04

    probs = np.clip(probs, 0.01, 0.90)
    labels = np.random.binomial(1, probs)
    print(f"Synthetic labels: {labels.sum()} closed / {n} total "
          f"({labels.mean():.1%} closure rate)")
    return labels


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class RoadClosureLSTM(nn.Module):
    """
    Small LSTM for tabular data.

    Each sample's feature vector (length F) is reshaped into a sequence of
    F time steps with input_size=1, so the LSTM processes features one at a
    time. The final hidden state is passed through a linear head.
    """

    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_features = num_features
        self.lstm = nn.LSTM(
            input_size=1,
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
        # x: (batch, num_features) -> (batch, num_features, 1)
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, batch, hidden)
        out = self.head(h_n[-1])            # last layer hidden state
        return out.squeeze(-1)              # (batch,)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Stratified 5-Fold CV
# ---------------------------------------------------------------------------

def run_cv(df_feat_np, labels, num_features, n_splits=5, epochs=30,
           batch_size=512, lr=1e-3, hidden_size=64, num_layers=2):
    """Stratified k-fold cross-validation for the LSTM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_aucs = []
    fold_briers = []
    all_y_true = []
    all_y_prob = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_feat_np, labels)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_train, X_val = df_feat_np[train_idx], df_feat_np[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Per-fold standard scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Class weight for imbalance
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = RoadClosureLSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
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

        y_prob = predict(model, val_loader, device)
        auc = roc_auc_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)
        fold_aucs.append(auc)
        fold_briers.append(brier)
        print(f"  Fold {fold+1} ROC AUC: {auc:.4f}  Brier: {brier:.4f}")

        all_y_true.append(y_val)
        all_y_prob.append(y_prob)

    all_y_true = np.concatenate(all_y_true)
    all_y_prob = np.concatenate(all_y_prob)

    print(f"\n{'=' * 60}")
    print("LSTM 5-FOLD CV RESULTS")
    print(f"{'=' * 60}")
    print(f"ROC AUC:     {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Brier Score: {np.mean(fold_briers):.4f} (+/- {np.std(fold_briers):.4f})")

    # Overall classification report at 0.5 threshold
    y_pred = (all_y_prob >= 0.5).astype(int)
    print(f"\nOverall classification report (threshold=0.5):")
    print(classification_report(all_y_true, y_pred, target_names=["Open", "Closed"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_y_true, y_pred))

    return all_y_true, all_y_prob, fold_aucs, fold_briers


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(y_true, y_prob, fold_aucs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC curve (aggregated)
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"LSTM (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Aggregated CV)")
    ax.legend()

    # 2. Per-fold AUC bar chart
    ax = axes[1]
    ax.bar(range(1, len(fold_aucs) + 1), fold_aucs)
    ax.axhline(np.mean(fold_aucs), color="red", linestyle="--", label=f"Mean={np.mean(fold_aucs):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Per-Fold ROC AUC")
    ax.set_ylim(0, 1)
    ax.legend()

    # 3. Predicted probability distribution
    ax = axes[2]
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Open", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Closed", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig("road_closure_lstm_results.png", dpi=150)
    print("\nSaved plots to road_closure_lstm_results.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, df_feat = load_and_prepare_data()
    labels = generate_synthetic_labels(df)

    df_feat_np = df_feat.values.astype(np.float32)
    num_features = df_feat_np.shape[1]

    y_true, y_prob, fold_aucs, fold_briers = run_cv(
        df_feat_np, labels, num_features,
        n_splits=5, epochs=30, batch_size=512,
        lr=1e-3, hidden_size=64, num_layers=2,
    )

    plot_results(y_true, y_prob, fold_aucs)

    overall_auc = roc_auc_score(y_true, y_prob)
    overall_brier = brier_score_loss(y_true, y_prob)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
LSTM 5-Fold CV Results:
  ROC AUC:     {overall_auc:.4f}
  Brier Score: {overall_brier:.4f}

Note: This dataset is tabular (static road features, no temporal
sequence). The LSTM treats each feature as a "time step" which is
a valid but unconventional approach. Compare with the XGBoost
baseline in road_closure_model.py — tree-based models typically
outperform LSTMs on tabular data.

For a more natural LSTM use case, you would need time-series data
(e.g., hourly weather + road status over multiple days per segment).
""")


if __name__ == "__main__":
    main()
