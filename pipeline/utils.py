"""
Shared utilities for the pipeline scripts.

Provides common constants (DATA_DIR, coordinate conversion factors), HTTP
fetching with retry, CSV writing, file-existence caching, and spatial helpers
used by both collection and ML scripts.
"""

import csv
import math
import os
import time

import requests

# ── Paths ────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── HTTP retry defaults ──────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF = 2

# ── Spatial constants (Pittsburgh latitude ≈ 40.44°) ────────────────────────

LAT_M_PER_DEG = 111320
LNG_M_PER_DEG = 111320 * math.cos(math.radians(40.44))


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def fetch_with_retry(url, *, params=None, json_body=None, headers=None,
                     method="get", timeout=60, stream=False,
                     max_retries=MAX_RETRIES, retry_backoff=RETRY_BACKOFF):
    """Fetch a URL with exponential backoff on 429/5xx and connection errors.

    Supports GET (default) and POST (set method="post"). Pass *json_body* for
    POST JSON payloads or *params* for GET query-strings. Extra *headers* are
    merged with a default User-Agent. Set *stream=True* for large downloads.
    """
    _headers = {"User-Agent": "Mozilla/5.0 (compatible; PittsburghSnowProject/1.0)"}
    if headers:
        _headers.update(headers)

    for attempt in range(max_retries):
        try:
            if method.lower() == "post":
                resp = requests.post(url, json=json_body, headers=_headers,
                                     params=params, timeout=timeout,
                                     stream=stream)
            else:
                resp = requests.get(url, params=params, headers=_headers,
                                    timeout=timeout, stream=stream)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = retry_backoff ** (attempt + 1)
                print(f"  HTTP {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError:
            wait = retry_backoff ** (attempt + 1)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries")


# ── CSV / file helpers ───────────────────────────────────────────────────────

def write_csv(filepath, rows, fieldnames):
    """Write a list of dicts to *filepath* as CSV with a header row."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def skip_if_exists(path):
    """Return True (and print a message) if *path* already exists on disk."""
    if os.path.exists(path):
        print(f"{path} already exists. Delete it to re-fetch.")
        return True
    return False


def load_csv(path, label):
    """Load a CSV into a pandas DataFrame, or return an empty one."""
    import pandas as pd
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {label}: {len(df)} rows")
        return df
    print(f"  {label}: NOT FOUND ({path})")
    return pd.DataFrame()


# ── Spatial helpers ──────────────────────────────────────────────────────────

def build_kdtree(df, lat_col="mid_lat", lng_col="mid_lng"):
    """Build a scipy KDTree from lat/lng columns (approximate meter coords)."""
    from scipy.spatial import KDTree
    import numpy as np

    df = df.dropna(subset=[lat_col, lng_col])
    if df.empty:
        return None, df

    coords = np.column_stack([
        df[lat_col].values * LAT_M_PER_DEG,
        df[lng_col].values * LNG_M_PER_DEG,
    ])
    tree = KDTree(coords)
    return tree, df
