"""Load and join OSM + Google Maps road CSVs into a single DataFrame."""

import pandas as pd

# Load both CSVs
df_osm = pd.read_csv("roads_osm_cmu.csv")
df_google = pd.read_csv("roads_around_cmu.csv")

print(f"OSM data:    {len(df_osm)} rows")
print(f"Google data: {len(df_google)} rows")

# Normalize Google data columns to align with OSM
df_google = df_google.rename(columns={
    "latitude": "mid_lat",
    "longitude": "mid_lng",
})

# Add missing columns so the concat works cleanly
for col in df_osm.columns:
    if col not in df_google.columns:
        df_google[col] = ""

# Keep only columns that exist in OSM, plus source
df_google["data_source"] = df_google["source"]
df_osm["data_source"] = "osm"

# Select shared columns + source
shared_cols = list(df_osm.columns) + ["data_source"]
# Make sure data_source is in osm too
shared_cols = list(dict.fromkeys(shared_cols))  # dedupe preserving order

df = pd.concat([df_osm, df_google[shared_cols]], ignore_index=True)

print(f"\nJoined data (before dedup): {len(df)} rows")

# --- Deduplicate rows that represent the same road ---

# Normalize name for matching: lowercase, strip whitespace
df["_name_key"] = df["name"].astype(str).str.strip().str.lower()

# Drop rows with no name — can't match them meaningfully
df_named = df[df["_name_key"].notna() & (df["_name_key"] != "") & (df["_name_key"] != "nan")]
df_unnamed = df[~df.index.isin(df_named.index)]

# Define how to merge each column when grouping duplicates
float_cols = ["mid_lat", "mid_lng"]
sparse_str_cols = [
    "highway_type", "surface", "lanes", "maxspeed", "oneway", "bridge",
    "tunnel", "lit", "sidewalk", "cycleway", "width", "access", "ref",
]

def first_non_empty(series):
    """Return the first non-empty/non-NaN value in a series."""
    for val in series:
        if pd.notna(val) and str(val).strip() not in ("", "nan"):
            return val
    return ""

def merge_sources(series):
    """Combine data_source values."""
    sources = set(str(s) for s in series if pd.notna(s) and str(s).strip())
    return ",".join(sorted(sources))

agg_dict = {
    "name": "first",
    "osm_id": "first",
    "mid_lat": "mean",
    "mid_lng": "mean",
    "node_count": "sum",
    "data_source": merge_sources,
}

# Ensure numeric columns are actually numeric (Google rows may have empty strings)
for col in ["mid_lat", "mid_lng", "node_count"]:
    if col in df_named.columns:
        df_named[col] = pd.to_numeric(df_named[col], errors="coerce")
for col in sparse_str_cols:
    if col in df_named.columns:
        agg_dict[col] = first_non_empty

df_deduped = df_named.groupby("_name_key", as_index=False).agg(agg_dict)

# Add back unnamed rows as-is
df_final = pd.concat([df_deduped, df_unnamed], ignore_index=True)
df_final = df_final.drop(columns=["_name_key"])

print(f"After dedup:               {len(df_final)} rows")
print(f"  (merged {len(df) - len(df_final)} duplicate rows)\n")

print(df_final.info())
print(f"\n{df_final.head(15)}")

print(f"\n--- By data source ---")
print(df_final["data_source"].value_counts())

print(f"\n--- Column value counts ---")
for col in ["highway_type", "surface", "lanes", "maxspeed", "oneway"]:
    print(f"\n{col}:")
    print(df_final[col].value_counts(dropna=False))

# Save
df_final.to_csv("roads_joined_cmu.csv", index=False)
print(f"\nSaved {len(df_final)} rows to roads_joined_cmu.csv")
