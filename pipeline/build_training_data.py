"""
Build the training matrix by joining static road features with storm weather
features and risk score labels.

Static features: from dataset_prediction_ready.csv (encoded + cleaned)
Weather features: from storm_events.csv (per-storm, broadcast to all segments)
Labels: from storm_labels.csv (risk_score per segment-storm)

Output: data/training_matrix.parquet (~2.8M rows x ~28 features)

Requires: pip install pandas numpy pyarrow
"""

import os
import sys

import numpy as np
import pandas as pd

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Input files
DATASET_CSV = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")
STORM_EVENTS_CSV = os.path.join(DATA_DIR, "storm_events.csv")
STORM_LABELS_CSV = os.path.join(DATA_DIR, "storm_labels.csv")
STORM_DAYS_CSV = os.path.join(DATA_DIR, "storm_days.csv")

# Output
OUTPUT_PARQUET = os.path.join(DATA_DIR, "training_matrix.parquet")

# ── Encoding maps ─────────────────────────────────────────────────────────────

DOMI_CLASS_ORDER = {
    "": 0, "Alley": 0, "Local": 1, "Collector": 2,
    "Minor Arterial": 3, "Major Arterial": 4, "Principal Arterial": 5,
}

PAVECLASS_ORDER = {
    "": 0, "UNPAVED": 0, "GRAVEL": 1, "BRICK": 2, "ASPHALT": 3, "CONCRETE": 4,
}

HIGHWAY_TYPE_OSM_ORDER = {
    "": 0, "service": 1, "residential": 2, "unclassified": 3,
    "tertiary": 4, "tertiary_link": 4,
    "secondary": 5, "secondary_link": 5,
    "primary": 6, "primary_link": 6,
    "trunk": 7, "trunk_link": 7,
    "motorway": 8, "motorway_link": 8,
}

SURFACE_OSM_MAP = {
    "": 0, "asphalt": 1, "concrete": 2, "paved": 3,
    "cobblestone": 4, "gravel": 5,
}

ONEWAY_MAP = {"": 0, "N": 0, "FT": 1, "TF": 2, "B": 0}


# ── Feature engineering ───────────────────────────────────────────────────────

def encode_static_features(df):
    """Encode and clean static road features from the prediction-ready dataset."""
    out = pd.DataFrame()
    out["objectid"] = df["objectid"]

    # ── Numeric features (median fill) ────────────────────────────────────────
    for col in ["speedlimit", "num_lanes", "roadwidth"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = 0

    if "elevation_m" in df.columns:
        out["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce")
        out["elevation_m"] = out["elevation_m"].fillna(out["elevation_m"].median())
    else:
        out["elevation_m"] = 0

    # ── Boolean features → int 0/1 ───────────────────────────────────────────
    for col in ["near_steep_slope", "near_landslide", "is_bridge", "is_snow_emergency_route"]:
        if col in df.columns:
            out[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        else:
            out[col] = 0

    # ── Historical risk features (numeric + log1p transforms) ─────────────────
    for col in ["winter_crash_count", "winter_crash_fatal", "snow_complaint_count",
                "plow_coverage_score"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            out[f"{col}_log"] = np.log1p(out[col])
        else:
            out[col] = 0
            out[f"{col}_log"] = 0

    # ── Closure counts ────────────────────────────────────────────────────────
    for col in ["domi_closure_count", "domi_full_closure_count"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            out[col] = 0

    # ── Ordinal encodings ─────────────────────────────────────────────────────
    if "domi_class" in df.columns:
        out["domi_class_enc"] = df["domi_class"].fillna("").map(
            lambda x: DOMI_CLASS_ORDER.get(str(x).strip(), 1)
        )
    else:
        out["domi_class_enc"] = 0

    if "paveclass" in df.columns:
        out["paveclass_enc"] = df["paveclass"].fillna("").map(
            lambda x: PAVECLASS_ORDER.get(str(x).strip(), 0)
        )
    else:
        out["paveclass_enc"] = 0

    if "highway_type_osm" in df.columns:
        out["highway_type_osm_enc"] = df["highway_type_osm"].fillna("").map(
            lambda x: HIGHWAY_TYPE_OSM_ORDER.get(str(x).strip(), 0)
        )
    else:
        out["highway_type_osm_enc"] = 0

    if "surface_osm" in df.columns:
        out["surface_osm_enc"] = df["surface_osm"].fillna("").map(
            lambda x: SURFACE_OSM_MAP.get(str(x).strip(), 0)
        )
    else:
        out["surface_osm_enc"] = 0

    if "oneway" in df.columns:
        out["oneway_enc"] = df["oneway"].fillna("").map(
            lambda x: ONEWAY_MAP.get(str(x).strip(), 0)
        )
    else:
        out["oneway_enc"] = 0

    # ── PennDOT features (sentinel for missing) ──────────────────────────────
    for col in ["penndot_aadt", "penndot_pavement_idx"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)
        else:
            out[col] = -1
    out["has_penndot_data"] = (out["penndot_aadt"] != -1).astype(int)

    # ── Incline (parse OSM incline tags like "10%", "up", "down") ────────────
    if "incline" in df.columns:
        def _parse_incline(val):
            s = str(val).strip().lower()
            if s in ("", "nan", "none"):
                return 0.0
            if s == "up":
                return 5.0  # moderate assumed
            if s == "down":
                return -5.0
            if s == "steep":
                return 15.0
            import re
            m = re.match(r"(-?\d+(?:\.\d+)?)\s*%?", s)
            if m:
                return float(m.group(1))
            return 0.0
        out["incline_pct"] = df["incline"].apply(_parse_incline)
    else:
        out["incline_pct"] = 0.0

    # ── Bridge age (numeric, 0 for non-bridge) ──────────────────────────────
    if "bridge_age_years" in df.columns:
        out["bridge_age_years"] = pd.to_numeric(df["bridge_age_years"], errors="coerce").fillna(0)
    else:
        out["bridge_age_years"] = 0

    # ── Coordinates ───────────────────────────────────────────────────────────
    out["mid_lat"] = pd.to_numeric(df["mid_lat"], errors="coerce")
    out["mid_lng"] = pd.to_numeric(df["mid_lng"], errors="coerce")

    # Drop rows without coords (~1.1%).
    # Preserve the original DataFrame index so callers can map back to the
    # source dataset row (e.g. to retrieve streetname, domi_class, etc.).
    before = len(out)
    out = out.dropna(subset=["mid_lat", "mid_lng"])
    dropped = before - len(out)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing coordinates")
    # Store the surviving original indices, then reset for clean 0-based use
    out.attrs["_original_index"] = out.index.tolist()
    out = out.reset_index(drop=True)

    return out


def prepare_weather_features(storms_df, storm_days_df=None):
    """Extract per-storm weather features, optionally merged with per-day temporal features.

    If storm_days_df is provided, the result has one row per (storm_id, day_offset)
    with temporal features; otherwise one row per storm_id (legacy behavior).
    """
    weather_cols = ["storm_id", "total_snowfall_cm", "max_daily_snowfall_cm",
                    "min_temp_c", "max_wind_kmh", "duration_days", "precip_total_mm",
                    "season"]

    # Only keep columns that exist
    available = [c for c in weather_cols if c in storms_df.columns]
    out = storms_df[available].copy()

    # Fill missing weather values
    for col in ["total_snowfall_cm", "max_daily_snowfall_cm", "min_temp_c",
                "max_wind_kmh", "duration_days", "precip_total_mm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    # Merge temporal features from storm_days if available
    if storm_days_df is not None and not storm_days_df.empty:
        temporal_cols = ["storm_id", "day_offset", "daily_snowfall_cm",
                         "cumulative_snowfall_cm", "days_since_peak_snowfall",
                         "is_post_storm", "cumulative_thaw_degree_hours"]
        avail_temporal = [c for c in temporal_cols if c in storm_days_df.columns]
        days = storm_days_df[avail_temporal].copy()
        for col in avail_temporal:
            if col not in ("storm_id", "day_offset"):
                days[col] = pd.to_numeric(days[col], errors="coerce").fillna(0)

        out = out.merge(days, on="storm_id", how="inner")
        print(f"  Merged temporal features: {len(out)} storm-day rows")

    return out


def downcast_floats(df):
    """Downcast float64 columns to float32 to halve memory."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    return df


def main():
    print("Building training matrix...")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\nLoading input data...")

    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found. Run build_dataset.py first.")
        sys.exit(1)
    dataset = pd.read_csv(DATASET_CSV)
    print(f"  Dataset: {len(dataset)} rows, {len(dataset.columns)} columns")

    if not os.path.exists(STORM_EVENTS_CSV):
        print(f"Error: {STORM_EVENTS_CSV} not found. Run build_labels.py first.")
        sys.exit(1)
    storms = pd.read_csv(STORM_EVENTS_CSV)
    print(f"  Storm events: {len(storms)} storms")

    if not os.path.exists(STORM_LABELS_CSV):
        print(f"Error: {STORM_LABELS_CSV} not found. Run build_labels.py first.")
        sys.exit(1)

    # Load storm_days if available (temporal features from Part B)
    storm_days = None
    has_day_level = False
    if os.path.exists(STORM_DAYS_CSV):
        storm_days = pd.read_csv(STORM_DAYS_CSV)
        has_day_level = True
        print(f"  Storm days: {len(storm_days)} storm-day rows")
    else:
        print(f"  Storm days: NOT FOUND (temporal features will be skipped)")

    # ── Encode static features ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Encoding static features...")
    print(f"{'='*60}")
    static = encode_static_features(dataset)
    static = downcast_floats(static)

    # Deduplicate by objectid — dataset may have duplicate objectids from
    # multi-segment roads; keep first occurrence per objectid
    before_dedup = len(static)
    static = static.drop_duplicates(subset=["objectid"], keep="first").reset_index(drop=True)
    if len(static) < before_dedup:
        print(f"  Deduplicated: {before_dedup} → {len(static)} (removed {before_dedup - len(static)} duplicate objectids)")

    print(f"  Static features: {len(static)} segments, {len(static.columns)} columns")
    print(f"  Columns: {', '.join(static.columns)}")

    # Free raw dataset
    del dataset

    # ── Prepare weather features ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Preparing weather features...")
    print(f"{'='*60}")
    weather = prepare_weather_features(storms, storm_days_df=storm_days)
    weather = downcast_floats(weather)
    print(f"  Weather features: {len(weather)} rows")
    print(f"  Columns: {', '.join(weather.columns)}")

    # Determine the merge key for labels ↔ weather
    weather_merge_keys = ["storm_id"]
    if has_day_level and "day_offset" in weather.columns:
        weather_merge_keys = ["storm_id", "day_offset"]

    # Temporal feature columns (Part B) — defined first so we can exclude from weather
    temporal_feature_cols = []
    if has_day_level:
        temporal_feature_cols = [
            "day_offset", "days_since_peak_snowfall", "is_post_storm",
            "cumulative_snowfall_cm", "cumulative_thaw_degree_hours", "daily_snowfall_cm",
        ]

    # Determine feature columns from a sample merge
    sample_static = static.drop(columns=["objectid"])
    sample_weather = weather.drop(columns=["storm_id"])
    # Exclude metadata cols and temporal cols (listed separately) from weather features
    non_feature_weather_cols = {"season", "day_offset"} | set(temporal_feature_cols)
    weather_feature_cols = [c for c in sample_weather.columns if c not in non_feature_weather_cols]
    static_feature_cols = list(sample_static.columns)

    # Weather × segment interaction features (computed during chunked join)
    interaction_cols = [
        "snow_x_steep_slope", "snow_x_incline", "temp_x_bridge",
        "wind_x_roadwidth", "snow_x_elevation",
    ]
    # Temporal × segment interaction features (Part B)
    temporal_interaction_cols = []
    if has_day_level:
        temporal_interaction_cols = [
            "thaw_x_steep_slope", "post_storm_x_elevation", "day_offset_x_snow",
        ]

    feature_cols = static_feature_cols + weather_feature_cols + temporal_feature_cols + interaction_cols + temporal_interaction_cols
    label_variant_cols = ["label_any_incident", "label_top_1pct", "label_top_5pct"]
    # day_offset is in temporal_feature_cols, so don't duplicate in id_cols
    all_cols = ["objectid", "storm_id", "risk_score"] + label_variant_cols + static_feature_cols + weather_feature_cols + temporal_feature_cols + interaction_cols + temporal_interaction_cols + ["season"]

    del sample_static, sample_weather

    # ── Process labels in chunks to avoid OOM ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Building training matrix (chunked processing)...")
    print(f"{'='*60}")

    CHUNK_SIZE = 500_000
    chunk_num = 0
    total_rows = 0
    first_chunk = True
    sample_rows = []

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Determine schema from first few rows
    labels_peek = pd.read_csv(STORM_LABELS_CSV, nrows=5)
    print(f"  Label columns: {', '.join(labels_peek.columns)}")

    writer = None

    for chunk in pd.read_csv(STORM_LABELS_CSV, chunksize=CHUNK_SIZE):
        chunk_num += 1
        print(f"  Processing chunk {chunk_num} ({len(chunk)} rows)...")

        # Keep only the columns we need from labels
        label_id_cols = ["objectid", "storm_id"]
        if has_day_level and "day_offset" in chunk.columns:
            label_id_cols.append("day_offset")
        label_cols_needed = label_id_cols + ["risk_score"] + [
            c for c in label_variant_cols if c in chunk.columns
        ]
        label_chunk = chunk[label_cols_needed].copy()

        # Merge static features
        label_chunk = label_chunk.merge(static, on="objectid", how="inner")

        # Merge weather features (includes season + temporal if available)
        label_chunk = label_chunk.merge(weather, on=weather_merge_keys, how="inner")

        # Compute weather × segment interaction features
        snow_col = "total_snowfall_cm" if "total_snowfall_cm" in label_chunk.columns else None
        temp_col = "min_temp_c" if "min_temp_c" in label_chunk.columns else None
        wind_col = "max_wind_kmh" if "max_wind_kmh" in label_chunk.columns else None

        if snow_col:
            label_chunk["snow_x_steep_slope"] = (
                label_chunk[snow_col] * label_chunk.get("near_steep_slope", 0)
            ).astype(np.float32)
            if "incline_pct" in label_chunk.columns:
                label_chunk["snow_x_incline"] = (
                    label_chunk[snow_col] * label_chunk["incline_pct"].abs()
                ).astype(np.float32)
            else:
                label_chunk["snow_x_incline"] = np.float32(0)
            label_chunk["snow_x_elevation"] = (
                label_chunk[snow_col] * label_chunk.get("elevation_m", 0) / 1000.0
            ).astype(np.float32)
        else:
            for c in ["snow_x_steep_slope", "snow_x_incline", "snow_x_elevation"]:
                label_chunk[c] = np.float32(0)

        if temp_col:
            # Lower temp on bridges is riskier (negative temp → positive interaction)
            label_chunk["temp_x_bridge"] = (
                (-label_chunk[temp_col]) * label_chunk.get("is_bridge", 0)
            ).astype(np.float32)
        else:
            label_chunk["temp_x_bridge"] = np.float32(0)

        if wind_col:
            # Narrower roads are more affected by wind
            rw = label_chunk.get("roadwidth", 0)
            rw = pd.to_numeric(rw, errors="coerce").fillna(0)
            label_chunk["wind_x_roadwidth"] = (
                label_chunk[wind_col] / (rw + 1)
            ).astype(np.float32)
        else:
            label_chunk["wind_x_roadwidth"] = np.float32(0)

        # Compute temporal × segment interaction features (Part B)
        if has_day_level:
            thaw = label_chunk.get("cumulative_thaw_degree_hours", 0)
            thaw = pd.to_numeric(thaw, errors="coerce").fillna(0)
            label_chunk["thaw_x_steep_slope"] = (
                thaw * label_chunk.get("near_steep_slope", 0)
            ).astype(np.float32)

            post_storm = label_chunk.get("is_post_storm", 0)
            post_storm = pd.to_numeric(post_storm, errors="coerce").fillna(0)
            label_chunk["post_storm_x_elevation"] = (
                post_storm * label_chunk.get("elevation_m", 0) / 1000.0
            ).astype(np.float32)

            day_off = label_chunk.get("day_offset", 0)
            day_off = pd.to_numeric(day_off, errors="coerce").fillna(0)
            snow_val = label_chunk.get("total_snowfall_cm", 0)
            snow_val = pd.to_numeric(snow_val, errors="coerce").fillna(0)
            label_chunk["day_offset_x_snow"] = (
                day_off * snow_val
            ).astype(np.float32)

        # Ensure all feature columns are numeric and filled
        for col in feature_cols:
            if col in label_chunk.columns:
                label_chunk[col] = pd.to_numeric(label_chunk[col], errors="coerce").fillna(0).astype(np.float32)

        # Ensure column order is consistent
        for col in all_cols:
            if col not in label_chunk.columns:
                label_chunk[col] = 0
        label_chunk = label_chunk[all_cols]

        # Save sample from first chunk
        if first_chunk:
            sample_rows.append(label_chunk.head(1000))
            first_chunk = False

        total_rows += len(label_chunk)

        # Write to parquet incrementally
        table = pa.Table.from_pandas(label_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema)
        writer.write_table(table)

        # Free chunk memory
        del label_chunk, table, chunk

    if writer is not None:
        writer.close()

    # Save CSV sample
    sample_csv = os.path.join(DATA_DIR, "training_matrix_sample.csv")
    if sample_rows:
        pd.concat(sample_rows).head(1000).to_csv(sample_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    # Read back a small slice for summary stats
    summary_df = pd.read_parquet(OUTPUT_PARQUET, columns=["risk_score", "season", "objectid", "storm_id"])

    print(f"\n{'='*60}")
    print("TRAINING MATRIX SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows: {total_rows}")
    print(f"  Total columns: {len(all_cols)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Feature columns: {', '.join(feature_cols)}")
    print(f"  Unique segments: {summary_df['objectid'].nunique()}")
    print(f"  Unique storms: {summary_df['storm_id'].nunique()}")
    print(f"  Seasons: {sorted(summary_df['season'].unique())}")
    print(f"  Risk score range: [{summary_df['risk_score'].min():.4f}, {summary_df['risk_score'].max():.4f}]")
    print(f"\n  Saved to {OUTPUT_PARQUET}")
    print(f"  Sample saved to {sample_csv}")


if __name__ == "__main__":
    main()
