"""
Join all collected CSVs into a prediction-ready dataset.

Base: centerlines_pgh.csv (~19,000 road segments)
Joins: OSM attributes, elevation, slopes, bridges, landslides, crashes, 311,
       plow activity, PennDOT snow routes — all via spatial proximity or objectid.

Weather is kept as a separate table (joined at prediction time, not per-road).

Output: dataset_prediction_ready.csv (~19,000 rows, ~30 columns)

Requires: pip install pandas numpy scipy
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# Input files
CENTERLINES_CSV = "centerlines_pgh.csv"
OSM_CSV = "roads_osm_pgh.csv"
ELEVATION_CSV = "elevation_pgh.csv"
SLOPES_CSV = "slopes_pgh.csv"
BRIDGES_CSV = "bridges_pgh.csv"
LANDSLIDES_CSV = "landslides_pgh.csv"
CRASHES_CSV = "crashes_winter_pgh.csv"
SNOW_311_CSV = "311_snow_pgh.csv"
PLOW_CSV = "plow_activity_pgh.csv"
PENNDOT_CSV = "penndot_snow_routes_pgh.csv"

OUTPUT_CSV = "dataset_prediction_ready.csv"

# Spatial join threshold in meters
SPATIAL_THRESHOLD_M = 50

# Approximate meters per degree at Pittsburgh latitude
LAT_M_PER_DEG = 111320
LNG_M_PER_DEG = 111320 * math.cos(math.radians(40.44))


def load_csv(path, label):
    """Load a CSV if it exists, or return an empty DataFrame."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {label}: {len(df)} rows")
        return df
    else:
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


def spatial_count(base_df, target_df, target_lat="latitude", target_lng="longitude",
                  threshold_m=SPATIAL_THRESHOLD_M):
    """Count how many target points fall within threshold_m of each base point."""
    if target_df.empty:
        return pd.Series(0, index=base_df.index)

    target_df = target_df.dropna(subset=[target_lat, target_lng])
    if target_df.empty:
        return pd.Series(0, index=base_df.index)

    # Coerce to numeric and drop non-numeric values
    target_df = target_df.copy()
    target_df[target_lat] = pd.to_numeric(target_df[target_lat], errors="coerce")
    target_df[target_lng] = pd.to_numeric(target_df[target_lng], errors="coerce")
    target_df = target_df.dropna(subset=[target_lat, target_lng])
    if target_df.empty:
        return pd.Series(0, index=base_df.index)

    # Convert target to meter coords
    target_coords = np.column_stack([
        target_df[target_lat].values * LAT_M_PER_DEG,
        target_df[target_lng].values * LNG_M_PER_DEG,
    ])
    target_tree = KDTree(target_coords)

    # Query from base points
    base_valid = base_df.dropna(subset=["mid_lat", "mid_lng"])
    base_coords = np.column_stack([
        base_valid["mid_lat"].values * LAT_M_PER_DEG,
        base_valid["mid_lng"].values * LNG_M_PER_DEG,
    ])

    counts = target_tree.query_ball_point(base_coords, r=threshold_m)
    count_series = pd.Series(0, index=base_df.index)
    count_series.loc[base_valid.index] = [len(c) for c in counts]
    return count_series


def spatial_nearest_flag(base_df, target_df, target_lat="latitude", target_lng="longitude",
                         threshold_m=SPATIAL_THRESHOLD_M):
    """Flag base points within threshold_m of any target point."""
    if target_df.empty:
        return pd.Series(False, index=base_df.index)

    target_df = target_df.copy()
    target_df[target_lat] = pd.to_numeric(target_df[target_lat], errors="coerce")
    target_df[target_lng] = pd.to_numeric(target_df[target_lng], errors="coerce")
    target_df = target_df.dropna(subset=[target_lat, target_lng])
    if target_df.empty:
        return pd.Series(False, index=base_df.index)

    target_coords = np.column_stack([
        target_df[target_lat].values * LAT_M_PER_DEG,
        target_df[target_lng].values * LNG_M_PER_DEG,
    ])
    target_tree = KDTree(target_coords)

    base_valid = base_df.dropna(subset=["mid_lat", "mid_lng"])
    base_coords = np.column_stack([
        base_valid["mid_lat"].values * LAT_M_PER_DEG,
        base_valid["mid_lng"].values * LNG_M_PER_DEG,
    ])

    dists, _ = target_tree.query(base_coords)
    flags = pd.Series(False, index=base_df.index)
    flags.loc[base_valid.index] = dists <= threshold_m
    return flags


def fuzzy_spatial_join_osm(base_df, osm_df, threshold_m=100):
    """Join OSM attributes to centerlines by fuzzy name + spatial proximity."""
    if osm_df.empty:
        return base_df

    print("  Joining OSM attributes (fuzzy name + spatial proximity)...")

    # Build name lookup for OSM
    osm_df = osm_df.copy()
    osm_df["_name_key"] = osm_df["name"].astype(str).str.strip().str.lower()

    base_df = base_df.copy()
    base_df["_name_key"] = base_df["streetname"].astype(str).str.strip().str.lower()

    # First pass: exact name match
    osm_by_name = osm_df.groupby("_name_key").first().reset_index()
    merged = base_df.merge(
        osm_by_name[["_name_key", "highway_type", "surface", "incline"]],
        on="_name_key",
        how="left",
        suffixes=("", "_osm"),
    )

    # Second pass: spatial nearest for unmatched rows
    unmatched = merged[merged["highway_type"].isna()]
    if not unmatched.empty and not osm_df.empty:
        osm_valid = osm_df.dropna(subset=["mid_lat", "mid_lng"])
        if not osm_valid.empty:
            osm_tree, osm_valid = build_kdtree(osm_valid)
            if osm_tree is not None:
                base_unmatched = unmatched.dropna(subset=["mid_lat", "mid_lng"])
                if not base_unmatched.empty:
                    base_coords = np.column_stack([
                        base_unmatched["mid_lat"].values * LAT_M_PER_DEG,
                        base_unmatched["mid_lng"].values * LNG_M_PER_DEG,
                    ])
                    dists, idxs = osm_tree.query(base_coords)
                    for i, (dist, idx) in enumerate(zip(dists, idxs)):
                        if dist <= threshold_m:
                            base_idx = base_unmatched.index[i]
                            osm_row = osm_valid.iloc[idx]
                            merged.loc[base_idx, "highway_type"] = osm_row.get("highway_type", "")
                            merged.loc[base_idx, "surface"] = osm_row.get("surface", "")
                            merged.loc[base_idx, "incline"] = osm_row.get("incline", "")

    merged = merged.rename(columns={
        "highway_type": "highway_type_osm",
        "surface": "surface_osm",
    })
    merged = merged.drop(columns=["_name_key"], errors="ignore")

    matched = merged["highway_type_osm"].notna().sum()
    print(f"    Matched {matched}/{len(merged)} segments to OSM data")

    return merged


def join_penndot(base_df, penndot_df, threshold_m=100):
    """Join PennDOT route data by spatial proximity."""
    if penndot_df.empty:
        base_df["penndot_route"] = ""
        base_df["penndot_aadt"] = ""
        base_df["penndot_pavement_idx"] = ""
        return base_df

    print("  Joining PennDOT route data (spatial proximity)...")
    penndot_valid = penndot_df.dropna(subset=["mid_lat", "mid_lng"]).copy()
    if penndot_valid.empty:
        base_df["penndot_route"] = ""
        base_df["penndot_aadt"] = ""
        base_df["penndot_pavement_idx"] = ""
        return base_df

    penndot_tree, penndot_valid = build_kdtree(penndot_valid)

    base_valid = base_df.dropna(subset=["mid_lat", "mid_lng"])
    base_coords = np.column_stack([
        base_valid["mid_lat"].values * LAT_M_PER_DEG,
        base_valid["mid_lng"].values * LNG_M_PER_DEG,
    ])

    dists, idxs = penndot_tree.query(base_coords)

    base_df["penndot_route"] = ""
    base_df["penndot_aadt"] = ""
    base_df["penndot_pavement_idx"] = ""

    for i, (dist, idx) in enumerate(zip(dists, idxs)):
        if dist <= threshold_m:
            base_idx = base_valid.index[i]
            row = penndot_valid.iloc[idx]
            base_df.loc[base_idx, "penndot_route"] = row.get("route_no", "")
            base_df.loc[base_idx, "penndot_aadt"] = row.get("aadt", "")
            base_df.loc[base_idx, "penndot_pavement_idx"] = row.get("pavement_condition", "")

    matched = (base_df["penndot_route"] != "").sum()
    print(f"    Matched {matched}/{len(base_df)} segments to PennDOT routes")
    return base_df


def main():
    print("Loading input CSVs...")
    print("=" * 60)

    base = load_csv(CENTERLINES_CSV, "Centerlines (base)")
    if base.empty:
        print(f"\nError: {CENTERLINES_CSV} not found. Run collect_centerlines.py first.")
        sys.exit(1)

    osm = load_csv(OSM_CSV, "OSM Pittsburgh")
    elevation = load_csv(ELEVATION_CSV, "Elevation")
    slopes = load_csv(SLOPES_CSV, "Slopes")
    bridges = load_csv(BRIDGES_CSV, "Bridges")
    landslides = load_csv(LANDSLIDES_CSV, "Landslides")
    crashes = load_csv(CRASHES_CSV, "Winter Crashes")
    snow_311 = load_csv(SNOW_311_CSV, "311 Snow")
    plow = load_csv(PLOW_CSV, "Plow Activity")
    penndot = load_csv(PENNDOT_CSV, "PennDOT Routes")

    # Ensure mid_lat/mid_lng are numeric
    for col in ["mid_lat", "mid_lng"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    print(f"\n{'='*60}")
    print("Building prediction dataset...")
    print(f"{'='*60}")

    df = base.copy()

    # --- Join OSM attributes ---
    df = fuzzy_spatial_join_osm(df, osm)

    # --- Join elevation by objectid ---
    if not elevation.empty and "objectid" in elevation.columns:
        print("  Joining elevation by objectid...")
        elev = elevation[["objectid", "elevation_m"]].copy()
        elev["objectid"] = pd.to_numeric(elev["objectid"], errors="coerce")
        df = df.merge(elev, on="objectid", how="left")
        filled = df["elevation_m"].notna().sum()
        print(f"    {filled}/{len(df)} segments with elevation")
    else:
        df["elevation_m"] = ""

    # --- Join slopes by objectid ---
    if not slopes.empty and "objectid" in slopes.columns:
        print("  Joining slopes by objectid...")
        sl = slopes[["objectid", "near_steep_slope"]].copy()
        sl["objectid"] = pd.to_numeric(sl["objectid"], errors="coerce")
        df = df.merge(sl, on="objectid", how="left")
        df["near_steep_slope"] = df["near_steep_slope"].fillna(False)
        flagged = df["near_steep_slope"].sum()
        print(f"    {flagged} segments near steep slopes")
    else:
        df["near_steep_slope"] = False

    # --- Spatial join: bridges ---
    print("  Flagging segments near bridges (spatial)...")
    df["is_bridge"] = spatial_nearest_flag(
        df, bridges, target_lat="latitude", target_lng="longitude",
        threshold_m=SPATIAL_THRESHOLD_M,
    )
    bridge_segs = df["is_bridge"].sum()
    print(f"    {bridge_segs} segments near bridges")

    # Bridge age
    if not bridges.empty and "year_built" in bridges.columns:
        bridges_valid = bridges.dropna(subset=["latitude", "longitude", "year_built"])
        if not bridges_valid.empty:
            bridge_tree, bridges_valid = build_kdtree(
                bridges_valid, lat_col="latitude", lng_col="longitude"
            )
            if bridge_tree is not None:
                base_valid = df.dropna(subset=["mid_lat", "mid_lng"])
                base_coords = np.column_stack([
                    base_valid["mid_lat"].values * LAT_M_PER_DEG,
                    base_valid["mid_lng"].values * LNG_M_PER_DEG,
                ])
                dists, idxs = bridge_tree.query(base_coords)
                df["bridge_age_years"] = ""
                current_year = 2026
                for i, (dist, idx) in enumerate(zip(dists, idxs)):
                    if dist <= SPATIAL_THRESHOLD_M:
                        yb = bridges_valid.iloc[idx].get("year_built")
                        try:
                            df.loc[base_valid.index[i], "bridge_age_years"] = current_year - int(float(yb))
                        except (ValueError, TypeError):
                            pass
    if "bridge_age_years" not in df.columns:
        df["bridge_age_years"] = ""

    # --- Spatial join: landslides ---
    print("  Flagging segments near landslides (spatial)...")
    df["near_landslide"] = spatial_nearest_flag(
        df, landslides, target_lat="latitude", target_lng="longitude",
        threshold_m=SPATIAL_THRESHOLD_M,
    )
    ls_segs = df["near_landslide"].sum()
    print(f"    {ls_segs} segments near landslides")

    # --- Spatial count: crashes ---
    print("  Counting winter crashes per segment (spatial, 200m radius)...")
    df["winter_crash_count"] = spatial_count(
        df, crashes, target_lat="latitude", target_lng="longitude",
        threshold_m=200,
    )
    # Fatal crashes
    if not crashes.empty and "fatal_count" in crashes.columns:
        fatal_crashes = crashes[pd.to_numeric(crashes["fatal_count"], errors="coerce") > 0]
        df["winter_crash_fatal"] = spatial_count(
            df, fatal_crashes, target_lat="latitude", target_lng="longitude",
            threshold_m=200,
        )
    else:
        df["winter_crash_fatal"] = 0
    print(f"    Total crash associations: {df['winter_crash_count'].sum()}")

    # --- Spatial count: 311 ---
    print("  Counting 311 snow complaints per segment (spatial, 200m radius)...")
    df["snow_complaint_count"] = spatial_count(
        df, snow_311, target_lat="latitude", target_lng="longitude",
        threshold_m=200,
    )
    print(f"    Total 311 associations: {df['snow_complaint_count'].sum()}")

    # --- Plow coverage score ---
    print("  Computing plow coverage score (spatial, 100m radius)...")
    df["plow_coverage_score"] = spatial_count(
        df, plow, target_lat="latitude", target_lng="longitude",
        threshold_m=100,
    )
    print(f"    Segments with plow activity: {(df['plow_coverage_score'] > 0).sum()}")

    # --- Join PennDOT routes ---
    df = join_penndot(df, penndot)

    # --- Select and order final columns ---
    final_columns = [
        # Road attributes
        "objectid", "streetname", "domi_class", "paveclass", "speedlimit",
        "num_lanes", "roadwidth", "oneway", "highway_type_osm", "surface_osm",
        "mid_lat", "mid_lng",
        # Topographic risk
        "elevation_m", "near_steep_slope", "near_landslide",
        # Structural risk
        "is_bridge", "bridge_age_years",
        # Priority/classification
        "penndot_route", "penndot_aadt", "penndot_pavement_idx",
        # Historical risk
        "winter_crash_count", "winter_crash_fatal", "snow_complaint_count",
        "plow_coverage_score",
    ]

    # Only include columns that exist
    final_columns = [c for c in final_columns if c in df.columns]

    df_out = df[final_columns]

    # Save
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Rows: {len(df_out)}")
    print(f"  Columns: {len(df_out.columns)}")
    print(f"  Columns: {', '.join(df_out.columns)}")
    print(f"\n  Non-null counts:")
    for col in df_out.columns:
        non_null = df_out[col].notna().sum()
        non_empty = (df_out[col].astype(str).str.strip() != "").sum()
        print(f"    {col:30s} : {non_empty}")

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
