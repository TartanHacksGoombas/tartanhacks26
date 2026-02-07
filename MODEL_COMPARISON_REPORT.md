# Model Comparison Report: `road-data-demo` vs `lstm-model`

> Generated 2026-02-07. Compares ML approaches across both branches for the Snow Road Closure Prediction project.

---

## 1. Branch Overview

| | `road-data-demo` (current) | `lstm-model` |
|---|---|---|
| **Base commit** | `8d32487` (shared ancestor) | `8d32487` (shared ancestor) |
| **ML commits** | 2 commits (LightGBM pipeline + ranker improvements) | 5 commits (XGBoost + LSTM models, merged from `xgboost-model`) |
| **ML files** | 6 pipeline scripts (`build_labels`, `build_training_data`, `train_model`, `train_ranker`, `evaluate`, `predict`) | 4 standalone model scripts + saved model artifacts in `models/` |
| **Net diff** | — | +2,327 lines, 13 files added/modified vs shared ancestor |

---

## 2. Architecture Comparison

### 2.1 `road-data-demo`: Storm-Event LightGBM Pipeline

Two models trained on **storm-level data** (126 storms x 17,549 segments = ~2.2M rows):

| Model | Type | Objective | Trees | Features | Weather? |
|---|---|---|---|---|---|
| **LightGBM Regression** (fallback) | `regression` / RMSE | Pointwise risk score | 11 | 35 (static + interactions) | Excluded (applied at prediction time) |
| **LightGBM LambdaRank** (primary) | `lambdarank` / NDCG | Within-storm road ranking | 43 | 41 (static + weather + interactions) | Included as features |

**Key architectural decisions:**
- LambdaRank directly optimizes NDCG, aligning with the downstream task of "rank roads by risk within a storm"
- Strong regularization: `min_child_samples=200`, `lambda_l2=2.0`, `max_depth=7` — guards against noisy proxy labels
- Regression model excludes weather features to prevent them from dominating, using weather as a post-hoc calibration multiplier instead

### 2.2 `lstm-model`: XGBoost + LSTM Models

Four independent models operating on **static per-segment data** (~28K rows):

| Model | Type | Task | Architecture |
|---|---|---|---|
| **XGBoost Classifier** (`road_closure_model.py`) | Binary classification | Predict road closure (yes/no) | `xgb.XGBClassifier(n_estimators=200, max_depth=5)` |
| **LSTM Classifier** (`road_closure_lstm.py`) | Binary classification | Predict road closure (yes/no) | PyTorch LSTM (hidden=64, layers=2, dropout=0.3), input_size=1, treating 21 features as 21 "time steps" |
| **XGBoost Regressor** (`road_safety_model.py`) | Regression | Predict safety score (0-100) | `xgb.XGBRegressor(n_estimators=200, max_depth=5)` |
| **LSTM Regressor** (`road_safety_lstm.py`) | Regression | Predict safety score (0-100) | PyTorch LSTM with learned embedding (Linear 18->256, reshaped to 4x64), hidden=64, layers=2 |

**Key architectural decisions:**
- The closure LSTM treats each of 21 features as a "time step" with `input_size=1` — there is no actual temporal sequence
- The safety LSTM improves on this with a learned embedding layer (18 -> 256 -> 4x64 pseudo-sequence), which is more principled
- XGBoost models use 80/20 train/test splits; LSTMs use stratified 5-fold CV
- No shared training/evaluation pipeline — each script is fully self-contained

---

## 3. Feature Comparison

### 3.1 `road-data-demo` Features (41 total for ranker)

**Static features (31):**
- Road attributes: `speedlimit`, `num_lanes`, `roadwidth`, `domi_class_enc`, `paveclass_enc`, `highway_type_osm_enc`, `surface_osm_enc`, `oneway_enc`
- Topographic: `elevation_m`, `near_steep_slope`, `near_landslide`, `incline_pct`
- Structural: `is_bridge`, `bridge_age_years`
- Historical aggregates: `winter_crash_count` (+ log), `snow_complaint_count` (+ log), `plow_coverage_score` (+ log), `domi_closure_count`, `domi_full_closure_count`, `winter_crash_fatal`
- PennDOT: `penndot_aadt`, `penndot_pavement_idx`, `has_penndot_data`
- Spatial: `mid_lat`, `mid_lng`
- Flag: `is_snow_emergency_route`

**Weather features (6):**
- `total_snowfall_cm`, `max_daily_snowfall_cm`, `min_temp_c`, `max_wind_kmh`, `duration_days`, `precip_total_mm`

**Interaction features (5):**
- `snow_x_steep_slope`, `snow_x_incline`, `temp_x_bridge`, `wind_x_roadwidth`, `snow_x_elevation`

### 3.2 `lstm-model` Features

**Closure models (21 features):**
- Road: `domi_class`, `paveclass`, `speedlimit`, `num_lanes`, `roadwidth`, `oneway`, `highway_type_osm`, `surface_osm`
- Topographic: `elevation_m`, `near_steep_slope`, `near_landslide`
- Structural: `is_bridge`, `bridge_age_years`
- Historical: `winter_crash_count`, `snow_complaint_count`, `plow_coverage_score`, `domi_closure_count`, `domi_full_closure_count`, `winter_crash_fatal`
- PennDOT: `penndot_aadt`, `penndot_pavement_idx`

**Safety models (18 features):**
- Same as closure models but **excludes** `winter_crash_count`, `snow_complaint_count`, `plow_coverage_score` (to reduce leakage since these contribute to the target score)

### 3.3 Feature Engineering Differences

| Aspect | `road-data-demo` | `lstm-model` |
|---|---|---|
| **Encoding** | Ordinal encoding for categoricals, log1p transforms for counts, -1 sentinel for missing PennDOT | Label encoding for categoricals, StandardScaler for LSTM |
| **Weather features** | 6 storm-level weather columns | None — no weather conditioning |
| **Interaction features** | 5 weather x segment cross-features | None |
| **Incline** | Parsed from OSM strings ("10%", "steep", "up") to numeric | Not included |
| **Log transforms** | `_log` variants for crash/complaint/plow counts | Not used |
| **Spatial coords** | `mid_lat`, `mid_lng` as features | Not used as features |
| **Missing data** | -1 sentinel for PennDOT features | Implicit handling by XGBoost/imputation |

---

## 4. Label Construction

This is the **most critical difference** between the branches.

### 4.1 `road-data-demo`: Storm-Event Proxy Labels

- **126 storm events** detected from 11 winters of weather history (snowfall >= 1cm, consecutive days grouped, +2 day buffer)
- **4 proxy sources** joined spatially and temporally via KDTree `query_ball_point`:
  - 311 snow complaints (200m radius, weight 0.45)
  - Winter crashes (200m radius, monthly allocation by snowfall, weight 0.25)
  - Plow activity gaps (100m radius, binary, weight 0.15)
  - DOMI full closures (200m radius, weight 0.15)
- **Normalization**: `log1p(x) / log1p(max)` within each storm (zeros stay at 0)
- **Label variants**: continuous `risk_score`, binary `label_any_incident`, percentile flags `label_top_1pct`/`label_top_5pct`
- **Result**: ~2.2M labeled rows (segments x storms)

### 4.2 `lstm-model`: Synthetic / Composite Labels

**Closure labels** (`generate_synthetic_labels()`):
- Rule-based probability: base rate ~10%, increased by heuristics:
  - Steep slope: +12%, landslide proximity: +20%, bridge: +15%
  - High crash count: +5%, high complaints: +8%
  - Low speed limit & narrow: +5%, poor pavement: +3%
- Binary sampled from Bernoulli(p) — **no actual closure ground truth**

**Safety score** (`compute_safety_score()`):
- Weighted composite of the same features used for prediction:
  - 30% crash risk, 20% closure history, 15% community reports
  - 10% plow coverage, 10% terrain, 5% structural, 5% geometry, 5% pavement
- Scaled to 0-100

### 4.3 Label Quality Assessment

| Criterion | `road-data-demo` | `lstm-model` |
|---|---|---|
| **Data-driven?** | Yes — derived from real incident data | Partially — composite of real counts but via hand-crafted rules |
| **Temporal grounding** | Yes — labels tied to specific storms | No — static per-segment labels |
| **Weather conditioning** | Yes — risk varies by storm severity | No — same label regardless of weather |
| **Data leakage risk** | Low — labels come from temporal joins, features from static attributes | **High** — closure model uses crash/complaint counts as both features AND label signals; safety model mitigates by excluding these features |
| **Scale** | 2.2M rows (segments x storms) | ~28K rows (one per segment) |

---

## 5. Performance Comparison

### 5.1 `road-data-demo` Reported Metrics

**LambdaRank Ranker** (temporal holdout: 2023-25 seasons):

| Metric | Value |
|---|---|
| NDCG@200 | 0.301 |
| Recall@500 | 0.211 |
| Mean Spearman | 0.018 |

**LightGBM Regression** (temporal holdout):

| Metric | Value |
|---|---|
| RMSE | 0.099 |
| Spearman | 0.223 |
| PR-AUC | 0.358 |

**Top ranker features**: `snow_complaint_count` (22.6%), `winter_crash_count` (21.0%), `mid_lat` (6.7%), `min_temp_c` (6.5%)

### 5.2 `lstm-model` Reported Metrics

**XGBoost Classifier** (road closure):
- 5-fold CV ROC AUC + Platt/isotonic calibration (exact values in script output, not persisted to metadata)

**LSTM Classifier** (road closure):
- 5-fold stratified CV with ROC AUC and Brier score (evaluation only, no saved model)

**XGBoost Regressor** (safety score):
- 5-fold CV MAE, model artifacts saved to `models/`

**LSTM Regressor** (safety score):
- 5-fold CV with MAE, RMSE, R^2 per fold
- Model weights and scaler params saved for inference

### 5.3 Comparability Notes

Direct metric comparison is **not meaningful** because:
1. The branches predict **different targets** (storm-specific risk score vs. static closure probability / safety score)
2. The label definitions are fundamentally different (proxy-based temporal labels vs. synthetic/composite labels)
3. The evaluation protocols differ (temporal holdout vs. random/stratified CV)
4. The `lstm-model` metrics are not persisted in metadata files — they are printed during training but not saved

---

## 6. Prediction Pipeline

### 6.1 `road-data-demo`

- **Input**: Weather scenario (NWS forecast JSON or CLI args: `--snowfall_cm`, `--min_temp_c`, etc.)
- **Process**: Encode features -> broadcast weather -> compute interactions -> predict -> calibrate via sigmoid + weather-severity power transform
- **Output**: `predictions_latest.csv` and `predictions_latest.json` (GeoJSON) with per-segment `risk_score` and `risk_category` (very_low / low / moderate / high / very_high)
- **Calibration**: Weather severity controls score distribution shape — light snow compresses scores toward 0, blizzards spread them

### 6.2 `lstm-model`

- **Input**: Static road segment data from `dataset_prediction_ready.csv`
- **Process**: Encode features -> predict (no weather conditioning)
- **Output**: `road_closure_model.json`/`.ubj` (XGBoost), `models/road_safety_lstm.pt` + `roads_scored_lstm.csv` (LSTM)
- **No weather-aware prediction** — scores are fixed per road segment regardless of storm conditions
- **LSTM inference**: Full pipeline with saved scaler params and label encoder mappings in `road_safety_lstm_meta.json`

---

## 7. Strengths and Weaknesses

### `road-data-demo`

**Strengths:**
- Weather-conditioned predictions — risk changes with storm severity
- LambdaRank objective directly optimizes ranking (matching the routing use case)
- Large training set (2.2M rows) with storm-level variation
- Temporal train/test split prevents future data leakage
- Comprehensive calibration pipeline for interpretable output categories
- Interaction features capture weather x segment coupling (e.g., snow + steep slope)

**Weaknesses:**
- Labels are proxy-based (no ground-truth closures) — noisy supervision signal
- Ranker Spearman correlation is low (0.018) on temporal holdout
- Plow data only available for 3/126 storms
- Requires full pipeline execution (labels + training data + training) — not self-contained

### `lstm-model`

**Strengths:**
- Self-contained scripts — each model file is standalone and runnable
- LSTM safety regressor has a well-designed embedding layer (18 -> 256 -> 4x64)
- XGBoost models include probability calibration (Platt scaling + isotonic)
- Saved model artifacts enable inference without retraining
- Safety model separates target features from input features to reduce leakage

**Weaknesses:**
- **Synthetic labels** — closure labels are generated from hand-crafted rules, not from data
- **Data leakage** — closure model uses crash/complaint counts as both features and label drivers
- **No weather conditioning** — predictions are static regardless of storm severity
- **LSTM misapplication** — treating tabular features as a sequence is architecturally unsound; the closure LSTM uses input_size=1 with 21 "time steps" (one per feature)
- **No temporal evaluation** — random/stratified splits don't test generalization to future storms
- **Small training set** — ~28K rows (one per segment) vs. 2.2M storm-segment pairs
- **Missing dependencies** in `requirements.txt` — `torch`, `xgboost`, `scikit-learn` not listed

---

## 8. Summary

| Dimension | `road-data-demo` | `lstm-model` | Verdict |
|---|---|---|---|
| **Label quality** | Proxy-based from 4 real data sources, storm-temporal | Synthetic rules / composite of input features | `road-data-demo` |
| **Weather awareness** | Full: 6 weather features + 5 interactions + calibration | None | `road-data-demo` |
| **Model architecture** | LightGBM LambdaRank (ranking-native) + regression fallback | XGBoost + LSTM (tabular-as-sequence) | `road-data-demo` for ranking; LSTM approach is architecturally questionable for tabular data |
| **Training scale** | 2.2M rows (segments x storms) | ~28K rows (segments only) | `road-data-demo` |
| **Evaluation rigor** | Temporal holdout (2023-25), per-storm NDCG/Recall/Spearman | Random/stratified CV (no temporal split) | `road-data-demo` |
| **Prediction usability** | Weather-conditioned, GeoJSON output, calibrated categories | Static scores, model artifacts for inference | `road-data-demo` for end-user; `lstm-model` for quick prototyping |
| **Data leakage** | Low (temporal separation) | Moderate-to-high (closure model) | `road-data-demo` |
| **Code organization** | Pipeline with shared utilities (`encode_static_features`) | Standalone scripts with duplicated code | `road-data-demo` |

**Bottom line:** The `road-data-demo` branch has a more principled ML pipeline — storm-event labels, weather conditioning, ranking-native objective, temporal evaluation, and calibrated predictions. The `lstm-model` branch provides useful XGBoost baselines and an LSTM experiment, but the synthetic labels, lack of weather features, data leakage in the closure model, and misapplication of LSTMs to non-sequential data limit its practical value. The LSTM safety regressor's embedding architecture is the most promising element from that branch and could potentially be adapted for use with the storm-event data in `road-data-demo`.
