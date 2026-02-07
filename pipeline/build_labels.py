"""
Construct storm events and per-segment-per-storm risk labels.

Step 1 — Storm event detection from weather_historical_pgh.csv:
    - Flag days with snowfall_cm >= 1.0
    - Group consecutive snow days into storm events
    - Extend each event window by +2 days (captures lagged complaints, ongoing plows)

Step 2 — Per-segment-per-storm labeling using 4 proxy sources:
    - 311 snow complaints (spatial + temporal)
    - Winter crashes (spatial + temporal, fractional monthly allocation)
    - Plow activity (spatial + temporal, gap detection)
    - DOMI full closures (spatial + temporal)

Step 3 — Composite risk score via percentile normalization + weighted sum.

Outputs: data/storm_events.csv, data/storm_labels.csv

Requires: pip install pandas numpy scipy
"""

import math
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Input files
WEATHER_CSV = os.path.join(DATA_DIR, "weather_historical_pgh.csv")
CENTERLINES_CSV = os.path.join(DATA_DIR, "centerlines_pgh.csv")
DATASET_CSV = os.path.join(DATA_DIR, "dataset_prediction_ready.csv")
SNOW_311_CSV = os.path.join(DATA_DIR, "311_snow_pgh.csv")
CRASHES_CSV = os.path.join(DATA_DIR, "crashes_winter_pgh.csv")
PLOW_CSV = os.path.join(DATA_DIR, "plow_activity_pgh.csv")
DOMI_CSV = os.path.join(DATA_DIR, "domi_closures_pgh.csv")

# Outputs
STORM_EVENTS_CSV = os.path.join(DATA_DIR, "storm_events.csv")
STORM_LABELS_CSV = os.path.join(DATA_DIR, "storm_labels.csv")
STORM_DAYS_CSV = os.path.join(DATA_DIR, "storm_days.csv")

# Spatial join thresholds (meters)
SPATIAL_311_M = 200
SPATIAL_CRASH_M = 200
SPATIAL_PLOW_M = 100
SPATIAL_DOMI_M = 200

# Snowfall threshold for a "snow day"
SNOWFALL_THRESHOLD_CM = 1.0

# Days to extend storm window after last snow day
STORM_WINDOW_EXTENSION_DAYS = 2

# Approximate meters per degree at Pittsburgh latitude
LAT_M_PER_DEG = 111320
LNG_M_PER_DEG = 111320 * math.cos(math.radians(40.44))

# Risk score weights
W_311 = 0.45
W_CRASH = 0.25
W_PLOW = 0.15
W_DOMI = 0.15


def load_csv(path, label):
    """Load a CSV if it exists, or return an empty DataFrame."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {label}: {len(df)} rows")
        return df
    print(f"  {label}: NOT FOUND ({path})")
    return pd.DataFrame()


def build_kdtree(df, lat_col="mid_lat", lng_col="mid_lng"):
    """Build a KDTree from lat/lng columns (in approximate meter coordinates)."""
    df = df.dropna(subset=[lat_col, lng_col])
    if df.empty:
        return None, df
    coords = np.column_stack([
        df[lat_col].values * LAT_M_PER_DEG,
        df[lng_col].values * LNG_M_PER_DEG,
    ])
    tree = KDTree(coords)
    return tree, df


def to_meter_coords(lats, lngs):
    """Convert lat/lng arrays to approximate meter coordinates."""
    return np.column_stack([
        lats * LAT_M_PER_DEG,
        lngs * LNG_M_PER_DEG,
    ])


def get_season(date):
    """Return winter season string, e.g. '2024-25' for dates Oct 2024 - Sep 2025."""
    if date.month >= 10:
        return f"{date.year}-{str(date.year + 1)[-2:]}"
    return f"{date.year - 1}-{str(date.year)[-2:]}"


# ── Step 1: Storm event detection ─────────────────────────────────────────────

def detect_storms(weather_df):
    """Detect storm events from daily weather data.

    Groups consecutive snow days (snowfall >= 1 cm) into events,
    then extends each window by +2 days.

    Returns a DataFrame with one row per storm.
    """
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df = weather_df.sort_values("date").reset_index(drop=True)

    # Flag snow days
    weather_df["is_snow_day"] = weather_df["snowfall_cm"] >= SNOWFALL_THRESHOLD_CM
    snow_days = weather_df[weather_df["is_snow_day"]].copy()

    if snow_days.empty:
        print("  No snow days found!")
        return pd.DataFrame()

    print(f"  Found {len(snow_days)} snow days across dataset")

    # Group consecutive snow days into storms
    snow_days = snow_days.sort_values("date").reset_index(drop=True)
    snow_days["gap"] = (snow_days["date"].diff() > timedelta(days=1)).astype(int)
    snow_days["storm_group"] = snow_days["gap"].cumsum()

    storms = []
    for group_id, group in snow_days.groupby("storm_group"):
        start_date = group["date"].min()
        end_date = group["date"].max()
        window_end = end_date + timedelta(days=STORM_WINDOW_EXTENSION_DAYS)

        # Weather stats from the weather dataframe within the storm window
        mask = (weather_df["date"] >= start_date) & (weather_df["date"] <= window_end)
        storm_weather = weather_df[mask]

        total_snowfall = group["snowfall_cm"].sum()
        max_daily_snowfall = group["snowfall_cm"].max()

        # Temperature and wind from full window
        min_temp = storm_weather["temp_min_c"].min() if "temp_min_c" in storm_weather.columns else np.nan
        max_wind = storm_weather["wind_max_kmh"].max() if "wind_max_kmh" in storm_weather.columns else np.nan
        precip_total = storm_weather["precip_mm"].sum() if "precip_mm" in storm_weather.columns else np.nan

        storms.append({
            "storm_id": len(storms),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "window_end": window_end.strftime("%Y-%m-%d"),
            "total_snowfall_cm": round(total_snowfall, 2),
            "max_daily_snowfall_cm": round(max_daily_snowfall, 2),
            "min_temp_c": round(min_temp, 1) if not np.isnan(min_temp) else np.nan,
            "max_wind_kmh": round(max_wind, 1) if not np.isnan(max_wind) else np.nan,
            "precip_total_mm": round(precip_total, 1) if not np.isnan(precip_total) else np.nan,
            "duration_days": (end_date - start_date).days + 1,
            "season": get_season(start_date),
        })

    storms_df = pd.DataFrame(storms)
    print(f"  Detected {len(storms_df)} storm events")
    return storms_df


def expand_storm_days(storms_df, weather_df):
    """Expand each storm into per-day rows with temporal features.

    For each (storm, day) pair, computes:
        - day_offset: 0-based from storm start
        - days_since_peak_snowfall: days since the peak snowfall day
        - is_post_storm: 1 if after the last snow day, 0 otherwise
        - cumulative_snowfall_cm: running total through this day
        - daily_snowfall_cm: this day's snowfall
        - cumulative_thaw_degree_hours: degree-hours above 0C since last snow

    Args:
        storms_df: DataFrame from detect_storms()
        weather_df: Daily weather DataFrame with date, snowfall_cm, temp_max_c

    Returns:
        DataFrame with one row per (storm_id, day_offset)
    """
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_lookup = weather_df.set_index("date")

    rows = []
    for _, storm in storms_df.iterrows():
        sid = storm["storm_id"]
        start = pd.Timestamp(storm["start_date"])
        end = pd.Timestamp(storm["end_date"])  # last snow day
        window_end = pd.Timestamp(storm["window_end"])

        # Generate all days from start to window_end
        all_days = pd.date_range(start, window_end, freq="D")

        cumulative_snow = 0.0
        peak_snow_day_offset = 0
        peak_snow_amount = 0.0
        last_snow_day_offset = 0
        cumulative_thaw = 0.0
        in_post_storm = False

        for i, day in enumerate(all_days):
            daily_snow = 0.0
            daily_temp_max = 0.0
            if day in weather_lookup.index:
                w = weather_lookup.loc[day]
                # Handle duplicate dates (take first row)
                if isinstance(w, pd.DataFrame):
                    w = w.iloc[0]
                daily_snow = float(w.get("snowfall_cm", 0) or 0)
                daily_temp_max = float(w.get("temp_max_c", 0) or 0)

            cumulative_snow += daily_snow

            # Track peak snowfall day
            if daily_snow > peak_snow_amount:
                peak_snow_amount = daily_snow
                peak_snow_day_offset = i

            # Track last snow day
            if daily_snow >= SNOWFALL_THRESHOLD_CM:
                last_snow_day_offset = i
                in_post_storm = False
                cumulative_thaw = 0.0  # reset thaw counter on snow days
            else:
                if i > last_snow_day_offset:
                    in_post_storm = True
                # Accumulate thaw: degree-hours above 0C (approx 12 hrs of temp_max)
                if daily_temp_max > 0:
                    cumulative_thaw += daily_temp_max * 12.0

            rows.append({
                "storm_id": sid,
                "day_offset": i,
                "date": day.strftime("%Y-%m-%d"),
                "daily_snowfall_cm": round(daily_snow, 2),
                "cumulative_snowfall_cm": round(cumulative_snow, 2),
                "days_since_peak_snowfall": i - peak_snow_day_offset,
                "is_post_storm": int(in_post_storm),
                "cumulative_thaw_degree_hours": round(cumulative_thaw, 1),
            })

    storm_days_df = pd.DataFrame(rows)
    print(f"  Expanded {len(storms_df)} storms into {len(storm_days_df)} storm-day rows")
    return storm_days_df


# ── Step 2: Per-segment-per-storm labeling ────────────────────────────────────

def label_311(segments_df, seg_tree, seg_coords, storms_df, snow_311_df):
    """Count 311 snow complaints per (segment, storm, day_offset)."""
    if snow_311_df.empty:
        print("  311: No data, skipping")
        return {}

    # Parse dates — handle both old/new column names
    date_col = None
    for col in ["create_date_et", "created_on", "CREATED_ON", "created_date"]:
        if col in snow_311_df.columns:
            date_col = col
            break
    if date_col is None:
        print("  311: No date column found, skipping")
        return {}

    snow_311_df = snow_311_df.copy()
    snow_311_df["_date"] = pd.to_datetime(snow_311_df[date_col], errors="coerce")
    snow_311_df = snow_311_df.dropna(subset=["_date"])

    # Parse coordinates
    lat_col = "latitude" if "latitude" in snow_311_df.columns else "LATITUDE"
    lng_col = "longitude" if "longitude" in snow_311_df.columns else "LONGITUDE"
    snow_311_df[lat_col] = pd.to_numeric(snow_311_df[lat_col], errors="coerce")
    snow_311_df[lng_col] = pd.to_numeric(snow_311_df[lng_col], errors="coerce")
    snow_311_df = snow_311_df.dropna(subset=[lat_col, lng_col])

    if snow_311_df.empty:
        print("  311: No valid rows after cleanup, skipping")
        return {}

    # Build 311 KDTree
    coords_311 = to_meter_coords(
        snow_311_df[lat_col].values,
        snow_311_df[lng_col].values,
    )
    tree_311 = KDTree(coords_311)

    # For each storm, find 311 complaints per day and count per segment
    results = {}
    for _, storm in storms_df.iterrows():
        sid = storm["storm_id"]
        start = pd.Timestamp(storm["start_date"])
        end = pd.Timestamp(storm["window_end"])

        # Temporal filter
        mask = (snow_311_df["_date"] >= start) & (snow_311_df["_date"] <= end)
        storm_311 = snow_311_df[mask]

        if storm_311.empty:
            continue

        # Assign day_offset to each complaint
        storm_311 = storm_311.copy()
        storm_311["_day_offset"] = (storm_311["_date"].dt.normalize() - start).dt.days

        # Spatial: for each 311 point, find ALL segments within radius
        pts_311 = to_meter_coords(
            storm_311[lat_col].values,
            storm_311[lng_col].values,
        )
        neighbor_lists = seg_tree.query_ball_point(pts_311, r=SPATIAL_311_M)

        # Count per (segment, day_offset)
        day_offsets = storm_311["_day_offset"].values
        for i, neighbors in enumerate(neighbor_lists):
            d = int(day_offsets[i])
            for idx in neighbors:
                oid = segments_df.iloc[idx]["objectid"]
                key = (oid, sid, d)
                results[key] = results.get(key, 0) + 1

    total = sum(results.values())
    print(f"  311: {total} associations across {len(results)} segment-storm-days")
    return results


def label_crashes(segments_df, seg_tree, seg_coords, storms_df, crashes_df):
    """Allocate monthly crash counts to storms weighted by snowfall.

    Distributes evenly across all days of the storm window since crash data
    only has year/month granularity.
    """
    if crashes_df.empty:
        print("  Crashes: No data, skipping")
        return {}

    crashes_df = crashes_df.copy()

    # Parse coordinates
    for col in ["latitude", "longitude"]:
        crashes_df[col] = pd.to_numeric(crashes_df[col], errors="coerce")
    crashes_df = crashes_df.dropna(subset=["latitude", "longitude"])

    if crashes_df.empty:
        print("  Crashes: No valid coords, skipping")
        return {}

    # Parse year/month
    year_col = "crash_year" if "crash_year" in crashes_df.columns else "CRASH_YEAR"
    month_col = "crash_month" if "crash_month" in crashes_df.columns else "CRASH_MONTH"
    if year_col not in crashes_df.columns or month_col not in crashes_df.columns:
        print("  Crashes: No year/month columns, skipping")
        return {}

    crashes_df["_year"] = pd.to_numeric(crashes_df[year_col], errors="coerce")
    crashes_df["_month"] = pd.to_numeric(crashes_df[month_col], errors="coerce")
    crashes_df = crashes_df.dropna(subset=["_year", "_month"])
    crashes_df["_year"] = crashes_df["_year"].astype(int)
    crashes_df["_month"] = crashes_df["_month"].astype(int)

    # Build crash KDTree
    crash_coords = to_meter_coords(
        crashes_df["latitude"].values,
        crashes_df["longitude"].values,
    )
    crash_tree = KDTree(crash_coords)

    # For each crash, find nearest segment
    dists, idxs = seg_tree.query(crash_coords)
    crashes_df["_seg_idx"] = idxs
    crashes_df["_seg_dist"] = dists
    crashes_df = crashes_df[crashes_df["_seg_dist"] <= SPATIAL_CRASH_M]

    # Map each crash to its nearest segment objectid
    crashes_df["_oid"] = crashes_df["_seg_idx"].map(
        lambda i: segments_df.iloc[i]["objectid"]
    )

    # Group crashes by (year, month, segment)
    crash_groups = crashes_df.groupby(["_year", "_month", "_oid"]).size().reset_index(name="count")

    # Parse storm dates
    storms_df = storms_df.copy()
    storms_df["_start"] = pd.to_datetime(storms_df["start_date"])
    storms_df["_end"] = pd.to_datetime(storms_df["window_end"])

    # For each (year, month), find storms and allocate proportionally by snowfall
    # Distribute across all days of the storm window
    results = {}
    for (year, month), grp in crash_groups.groupby(["_year", "_month"]):
        # Find storms overlapping this month
        month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
        if month == 12:
            month_end = pd.Timestamp(year=int(year) + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=int(year), month=int(month) + 1, day=1) - timedelta(days=1)

        overlapping = storms_df[
            (storms_df["_start"] <= month_end) & (storms_df["_end"] >= month_start)
        ]

        if overlapping.empty:
            continue

        # Weight by snowfall
        total_snow = overlapping["total_snowfall_cm"].sum()
        if total_snow <= 0:
            total_snow = 1.0

        for _, storm in overlapping.iterrows():
            weight = storm["total_snowfall_cm"] / total_snow
            sid = storm["storm_id"]
            # Number of days in the storm window
            n_days = (storm["_end"] - storm["_start"]).days + 1
            per_day_weight = weight / max(n_days, 1)

            for _, row in grp.iterrows():
                oid = row["_oid"]
                score_per_day = row["count"] * per_day_weight
                for d in range(n_days):
                    key = (oid, sid, d)
                    results[key] = results.get(key, 0) + score_per_day

    total = sum(results.values())
    print(f"  Crashes: {total:.1f} weighted associations across {len(results)} segment-storm-days")
    return results


def label_plow_gaps(segments_df, seg_tree, seg_coords, storms_df, plow_df):
    """Flag segments with zero plow points during storms that have plow data.

    Plow data is binary per-storm, so replicate the flag across all days
    of the storm window.
    """
    if plow_df.empty:
        print("  Plow: No data, skipping")
        return {}, set()

    plow_df = plow_df.copy()
    for col in ["latitude", "longitude"]:
        plow_df[col] = pd.to_numeric(plow_df[col], errors="coerce")
    plow_df = plow_df.dropna(subset=["latitude", "longitude"])

    if plow_df.empty:
        print("  Plow: No valid coords, skipping")
        return {}, set()

    # Parse event dates
    date_col = None
    for col in ["event_date", "EVENT_DATE", "timestamp"]:
        if col in plow_df.columns:
            date_col = col
            break
    if date_col is None:
        print("  Plow: No date column found, skipping")
        return {}, set()

    # event_date might be a range like "2018-11-15 - 2018-11-16" or a single date
    plow_df["_date_str"] = plow_df[date_col].astype(str)

    # Build plow KDTree
    plow_coords = to_meter_coords(
        plow_df["latitude"].values,
        plow_df["longitude"].values,
    )
    plow_tree = KDTree(plow_coords)

    # For each plow point, find nearest segment
    dists, idxs = seg_tree.query(plow_coords)
    plow_df["_seg_idx"] = idxs
    plow_df["_seg_dist"] = dists
    plow_df = plow_df[plow_df["_seg_dist"] <= SPATIAL_PLOW_M]
    plow_df["_oid"] = plow_df["_seg_idx"].map(
        lambda i: segments_df.iloc[i]["objectid"]
    )

    # Determine which storms have plow data (by date range overlap)
    storms_with_plow = set()
    results = {}

    all_oids = set(segments_df["objectid"].values)

    for _, storm in storms_df.iterrows():
        sid = storm["storm_id"]
        start = storm["start_date"]
        end = storm["window_end"]
        n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1

        # Check if any plow points fall in storm window
        # Plow event_date can be a range — check if any part overlaps
        mask = plow_df["_date_str"].str.contains(
            "|".join(pd.date_range(start, end).strftime("%Y-%m-%d")),
            na=False,
        )
        storm_plow = plow_df[mask]

        if len(storm_plow) < 10:
            # Not enough plow data for this storm
            continue

        storms_with_plow.add(sid)

        # Segments that DID get plowed
        plowed_oids = set(storm_plow["_oid"].values)

        # Gap = segments with zero plow points, replicated across all days
        for oid in all_oids:
            gap_val = 0 if oid in plowed_oids else 1
            for d in range(n_days):
                results[(oid, sid, d)] = gap_val

    print(f"  Plow: {len(storms_with_plow)} storms with plow data, "
          f"{sum(v == 1 for v in results.values())} gap flags")
    return results, storms_with_plow


def label_domi_closures(segments_df, seg_tree, seg_coords, storms_df, domi_df):
    """Flag segments with DOMI full closures per (segment, storm, day_offset)."""
    if domi_df.empty:
        print("  DOMI: No data, skipping")
        return {}

    domi_df = domi_df.copy()

    # Filter to full closures
    full_col = None
    for col in ["full_closure", "FULL_CLOSURE"]:
        if col in domi_df.columns:
            full_col = col
            break
    if full_col:
        domi_df = domi_df[domi_df[full_col].astype(str).str.lower().isin(["true", "1", "yes"])]

    if domi_df.empty:
        print("  DOMI: No full closures found, skipping")
        return {}

    # Parse dates
    from_col = "from_date" if "from_date" in domi_df.columns else "FROM_DATE"
    to_col = "to_date" if "to_date" in domi_df.columns else "TO_DATE"
    if from_col not in domi_df.columns:
        print("  DOMI: No from_date column, skipping")
        return {}

    domi_df["_from"] = pd.to_datetime(domi_df[from_col], errors="coerce")
    domi_df["_to"] = pd.to_datetime(domi_df[to_col], errors="coerce")
    domi_df = domi_df.dropna(subset=["_from"])
    # If no to_date, assume same day
    domi_df["_to"] = domi_df["_to"].fillna(domi_df["_from"])

    # Parse coordinates
    lat_col = "latitude" if "latitude" in domi_df.columns else "LATITUDE"
    lng_col = "longitude" if "longitude" in domi_df.columns else "LONGITUDE"
    domi_df[lat_col] = pd.to_numeric(domi_df[lat_col], errors="coerce")
    domi_df[lng_col] = pd.to_numeric(domi_df[lng_col], errors="coerce")
    domi_df = domi_df.dropna(subset=[lat_col, lng_col])

    if domi_df.empty:
        print("  DOMI: No valid rows after cleanup, skipping")
        return {}

    # Spatial: find ALL segments within radius for each DOMI closure
    domi_coords = to_meter_coords(
        domi_df[lat_col].values,
        domi_df[lng_col].values,
    )
    neighbor_lists = seg_tree.query_ball_point(domi_coords, r=SPATIAL_DOMI_M)
    # Build a mapping: each DOMI row index → list of segment objectids
    domi_df = domi_df.reset_index(drop=True)
    domi_seg_oids = []
    for i, neighbors in enumerate(neighbor_lists):
        oids = [segments_df.iloc[idx]["objectid"] for idx in neighbors]
        domi_seg_oids.append(oids)

    # Temporal: for each storm, find overlapping closures and assign to days
    results = {}
    for _, storm in storms_df.iterrows():
        sid = storm["storm_id"]
        start = pd.Timestamp(storm["start_date"])
        end = pd.Timestamp(storm["window_end"])

        mask = (domi_df["_from"] <= end) & (domi_df["_to"] >= start)
        storm_indices = domi_df.index[mask].tolist()

        if not storm_indices:
            continue

        # Flag segments per day within radius of any matching DOMI closure
        for i in storm_indices:
            closure_from = domi_df.at[i, "_from"]
            closure_to = domi_df.at[i, "_to"]
            # Determine which days of the storm window this closure overlaps
            overlap_start = max(start, closure_from)
            overlap_end = min(end, closure_to)
            for day in pd.date_range(overlap_start, overlap_end, freq="D"):
                d = (day - start).days
                for oid in domi_seg_oids[i]:
                    results[(oid, sid, d)] = 1

    print(f"  DOMI: {len(results)} segment-storm-day closure flags")
    return results


# ── Step 3: Composite risk score ──────────────────────────────────────────────

def compute_risk_scores(storms_df, segments_df, labels_311, labels_crash,
                        labels_plow, storms_with_plow, labels_domi):
    """Compute composite risk score per segment per storm per day.

    Uses log-scaled min-max normalization within each (storm, day) pair
    (zeros stay at 0), then weighted sum. Also outputs binary label variants.

    Labels are now keyed by (objectid, storm_id, day_offset).
    """
    all_oids = segments_df["objectid"].values
    n_segments = len(all_oids)

    rows = []
    n_storms = len(storms_df)

    for storm_idx, (_, storm) in enumerate(storms_df.iterrows()):
        sid = storm["storm_id"]
        start = pd.Timestamp(storm["start_date"])
        window_end = pd.Timestamp(storm["window_end"])
        n_days = (window_end - start).days + 1

        if (storm_idx + 1) % 20 == 0 or storm_idx == 0:
            print(f"  Computing risk scores: storm {storm_idx + 1}/{n_storms} ({n_days} days)...")

        has_plow = sid in storms_with_plow

        # Log-scaled min-max normalization (within this storm-day).
        # Zeros stay at 0; only segments with actual incidents get > 0.
        def log_minmax_norm(arr):
            """Log1p min-max normalization to [0, 1]. Zeros → 0, NaN preserved."""
            valid = ~np.isnan(arr)
            if valid.sum() == 0:
                result = np.zeros_like(arr)
                result[~valid] = np.nan
                return result
            log_arr = np.log1p(np.clip(arr[valid], 0, None))
            max_val = log_arr.max()
            result = np.full_like(arr, np.nan, dtype=float)
            if max_val > 0:
                result[valid] = log_arr / max_val
            else:
                result[valid] = 0.0
            return result

        storm_day_rows = []

        for d in range(n_days):
            # Gather raw values for this (storm, day)
            raw_311 = np.zeros(n_segments)
            raw_crash = np.zeros(n_segments)
            raw_plow = np.full(n_segments, np.nan)
            raw_domi = np.zeros(n_segments)

            has_any_data = False

            for i, oid in enumerate(all_oids):
                key = (oid, sid, d)
                if key in labels_311:
                    raw_311[i] = labels_311[key]
                    has_any_data = True
                if key in labels_crash:
                    raw_crash[i] = labels_crash[key]
                    has_any_data = True
                if key in labels_plow:
                    raw_plow[i] = labels_plow[key]
                    has_any_data = True
                if key in labels_domi:
                    raw_domi[i] = labels_domi[key]
                    has_any_data = True

            if not has_any_data:
                # Still emit rows for this day (all zeros) so training data has full coverage
                for i, oid in enumerate(all_oids):
                    storm_day_rows.append({
                        "objectid": oid,
                        "storm_id": sid,
                        "day_offset": d,
                        "storm_311_count": 0,
                        "storm_crash_score": 0,
                        "storm_plow_gap": raw_plow[i] if has_plow else np.nan,
                        "storm_domi_closure": 0,
                        "risk_score": 0.0,
                        "label_any_incident": 0,
                    })
                continue

            pctile_311 = log_minmax_norm(raw_311)
            pctile_crash = log_minmax_norm(raw_crash)
            pctile_domi = log_minmax_norm(raw_domi)
            pctile_plow = log_minmax_norm(raw_plow)

            for i, oid in enumerate(all_oids):
                if has_plow:
                    score = (W_311 * pctile_311[i] +
                             W_CRASH * pctile_crash[i] +
                             W_PLOW * pctile_plow[i] +
                             W_DOMI * pctile_domi[i])
                else:
                    w_total = W_311 + W_CRASH + W_DOMI
                    score = ((W_311 / w_total) * pctile_311[i] +
                             (W_CRASH / w_total) * pctile_crash[i] +
                             (W_DOMI / w_total) * pctile_domi[i])

                score = max(0.0, min(1.0, score))

                any_incident = int(
                    raw_311[i] > 0 or raw_crash[i] > 0
                    or raw_domi[i] > 0
                    or (has_plow and not np.isnan(raw_plow[i]) and raw_plow[i] > 0)
                )

                storm_day_rows.append({
                    "objectid": oid,
                    "storm_id": sid,
                    "day_offset": d,
                    "storm_311_count": raw_311[i],
                    "storm_crash_score": round(raw_crash[i], 4),
                    "storm_plow_gap": raw_plow[i] if has_plow else np.nan,
                    "storm_domi_closure": raw_domi[i],
                    "risk_score": round(score, 6),
                    "label_any_incident": any_incident,
                })

        # Compute top-percentile labels within this storm (across all days)
        scores = np.array([r["risk_score"] for r in storm_day_rows])
        p99 = np.percentile(scores, 99) if len(scores) > 0 else 0
        p95 = np.percentile(scores, 95) if len(scores) > 0 else 0
        for r in storm_day_rows:
            r["label_top_1pct"] = int(r["risk_score"] >= p99 and r["risk_score"] > 0)
            r["label_top_5pct"] = int(r["risk_score"] >= p95 and r["risk_score"] > 0)

        rows.extend(storm_day_rows)

    labels_df = pd.DataFrame(rows)
    return labels_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building storm labels...")
    print("=" * 60)

    # Load inputs
    print("\nLoading input data...")
    weather = load_csv(WEATHER_CSV, "Weather history")
    if weather.empty:
        print(f"\nError: {WEATHER_CSV} not found. Run collect_weather_historical.py first.")
        sys.exit(1)

    # Use the prediction-ready dataset for segment coords
    segments = load_csv(DATASET_CSV, "Dataset (segments)")
    if segments.empty:
        # Fall back to raw centerlines
        segments = load_csv(CENTERLINES_CSV, "Centerlines (fallback)")
    if segments.empty:
        print(f"\nError: No segment data found. Run build_dataset.py first.")
        sys.exit(1)

    # Ensure numeric coords
    for col in ["mid_lat", "mid_lng"]:
        segments[col] = pd.to_numeric(segments[col], errors="coerce")
    segments = segments.dropna(subset=["mid_lat", "mid_lng"]).reset_index(drop=True)

    # Deduplicate by objectid — dataset may have duplicate objectids from
    # multi-segment roads; keep first occurrence per objectid
    before_dedup = len(segments)
    segments = segments.drop_duplicates(subset=["objectid"], keep="first").reset_index(drop=True)
    if len(segments) < before_dedup:
        print(f"  Deduplicated: {before_dedup} → {len(segments)} unique objectids")

    print(f"  Segments with valid coords: {len(segments)}")

    snow_311 = load_csv(SNOW_311_CSV, "311 snow complaints")
    crashes = load_csv(CRASHES_CSV, "Winter crashes")
    plow = load_csv(PLOW_CSV, "Plow activity")
    domi = load_csv(DOMI_CSV, "DOMI closures")

    # ── Step 1: Detect storms ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 1: Detecting storm events...")
    print(f"{'='*60}")
    storms_df = detect_storms(weather)
    if storms_df.empty:
        print("No storms detected. Exiting.")
        sys.exit(1)

    storms_df.to_csv(STORM_EVENTS_CSV, index=False)
    print(f"  Saved {len(storms_df)} storms to {STORM_EVENTS_CSV}")

    # Expand storms into per-day rows with temporal features
    print(f"\n  Expanding storms into per-day rows...")
    storm_days_df = expand_storm_days(storms_df, weather)
    storm_days_df.to_csv(STORM_DAYS_CSV, index=False)
    print(f"  Saved {len(storm_days_df)} storm-day rows to {STORM_DAYS_CSV}")

    # ── Step 2: Build segment KDTree ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2: Building segment spatial index...")
    print(f"{'='*60}")
    seg_tree, seg_valid = build_kdtree(segments)
    seg_coords = to_meter_coords(
        segments["mid_lat"].values,
        segments["mid_lng"].values,
    )
    print(f"  Segment KDTree: {len(segments)} points")

    # ── Step 3: Label each source ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3: Labeling from proxy sources...")
    print(f"{'='*60}")

    print("\n  [311 Snow Complaints]")
    labels_311 = label_311(segments, seg_tree, seg_coords, storms_df, snow_311)

    print("\n  [Winter Crashes]")
    labels_crash = label_crashes(segments, seg_tree, seg_coords, storms_df, crashes)

    print("\n  [Plow Activity Gaps]")
    labels_plow, storms_with_plow = label_plow_gaps(
        segments, seg_tree, seg_coords, storms_df, plow
    )

    print("\n  [DOMI Closures]")
    labels_domi = label_domi_closures(
        segments, seg_tree, seg_coords, storms_df, domi
    )

    # ── Step 4: Compute composite risk scores ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4: Computing composite risk scores...")
    print(f"{'='*60}")

    labels_df = compute_risk_scores(
        storms_df, segments,
        labels_311, labels_crash,
        labels_plow, storms_with_plow,
        labels_domi,
    )

    if labels_df.empty:
        print("  No labels generated. Check input data.")
        sys.exit(1)

    # Merge storm metadata for convenience
    labels_df = labels_df.merge(
        storms_df[["storm_id", "season", "total_snowfall_cm"]],
        on="storm_id",
        how="left",
    )

    labels_df.to_csv(STORM_LABELS_CSV, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("LABEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Storm events: {len(storms_df)}")
    print(f"  Label rows: {len(labels_df)}")
    print(f"  Unique segments: {labels_df['objectid'].nunique()}")
    print(f"  Unique storms in labels: {labels_df['storm_id'].nunique()}")
    print(f"  Risk score range: [{labels_df['risk_score'].min():.4f}, {labels_df['risk_score'].max():.4f}]")
    print(f"  Risk score mean: {labels_df['risk_score'].mean():.4f}")
    print(f"  Risk score median: {labels_df['risk_score'].median():.4f}")
    print(f"\n  Saved to {STORM_LABELS_CSV}")


if __name__ == "__main__":
    main()
