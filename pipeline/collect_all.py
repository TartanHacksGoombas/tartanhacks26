"""
Run all data collection scripts in dependency order.

Phase A (run first):     collect_centerlines.py → collect_osm_pgh.py
Phase B (needs Phase A): collect_elevation.py, collect_slopes.py, collect_bridges.py, collect_landslides.py
Phase C (independent):   collect_weather_historical.py, collect_weather_realtime.py, collect_penndot_winter.py
Phase D (independent):   collect_crashes.py, collect_311_snow.py, collect_plow_activity.py, collect_penndot_snow_routes.py
Phase E (needs all):     build_dataset.py
Phase F (ML pipeline):   build_labels.py → build_training_data.py → train_model.py → predict.py

Usage: python collect_all.py
"""

import importlib
import os
import sys
import time

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure sibling modules are importable
sys.path.insert(0, _SCRIPT_DIR)


def run_script(module_name):
    """Import and run a script's main() function."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {module_name}.py")
    print(f"{'='*60}\n")

    start = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - start
        print(f"\n  Completed {module_name} in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ERROR in {module_name} after {elapsed:.1f}s: {e}")
        return False


def main():
    start_total = time.time()

    print("Snow Road Closure Prediction — Data Collection Pipeline")
    print("=" * 60)

    results = {}

    # Phase A — Road Network Foundation (sequential)
    print("\n>>> PHASE A: Road Network Foundation")
    results["collect_centerlines"] = run_script("collect_centerlines")
    results["collect_osm_pgh"] = run_script("collect_osm_pgh")

    # Phase B — Environmental Risk Factors (sequential, needs Phase A)
    print("\n>>> PHASE B: Environmental Risk Factors")
    for script in ["collect_elevation", "collect_slopes", "collect_bridges", "collect_landslides",
                    "collect_snow_emergency_routes"]:
        results[script] = run_script(script)

    # Phase C — Weather Data
    print("\n>>> PHASE C: Weather Data")
    for script in ["collect_weather_historical", "collect_weather_realtime", "collect_penndot_winter"]:
        results[script] = run_script(script)

    # Phase D — Historical Incident Data
    print("\n>>> PHASE D: Historical Incident Data")
    for script in ["collect_crashes", "collect_311_snow", "collect_plow_activity", "collect_penndot_snow_routes",
                    "collect_domi_closures",
                    "collect_511pa_events", "collect_penndot_press"]:
        results[script] = run_script(script)

    # Phase E — Master Join
    print("\n>>> PHASE E: Master Join")
    results["build_dataset"] = run_script("build_dataset")

    # Phase F — ML Pipeline (labels → features → train → predict)
    print("\n>>> PHASE F: ML Pipeline")
    results["build_labels"] = run_script("build_labels")
    results["build_training_data"] = run_script("build_training_data")
    results["train_model"] = run_script("train_model")
    results["predict"] = run_script("predict")

    # Summary
    elapsed_total = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed_total:.1f}s total")
    print(f"{'='*60}")

    print(f"\nResults:")
    for script, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {script:35s} : {status}")

    # Check output files
    print(f"\nOutput files:")
    output_files = [
        ("centerlines_pgh.csv", "~19,000 road segments"),
        ("roads_osm_pgh.csv", "~15,000-30,000 OSM roads"),
        ("elevation_pgh.csv", "~19,000 elevations"),
        ("slopes_pgh.csv", "~19,000 slope flags"),
        ("bridges_pgh.csv", "~153 bridges"),
        ("landslides_pgh.csv", "~7,279 landslides"),
        ("weather_historical_pgh.csv", "~1,510 weather days"),
        ("weather_forecast_pgh.json", "7-day forecast"),
        ("weather_alerts_pgh.json", "active alerts"),
        ("penndot_conditions_pgh.csv", "current conditions"),
        ("crashes_winter_pgh.csv", "~19,897 crashes"),
        ("311_snow_pgh.csv", "311 snow complaints"),
        ("plow_activity_pgh.csv", "plow GPS traces"),
        ("penndot_snow_routes_pgh.csv", "PennDOT routes"),
        ("domi_closures_pgh.csv", "~63,000 permitted closures"),
        ("snow_emergency_routes_pgh.csv", "~286 route segments"),
        ("511pa_events_pgh.csv", "511PA traffic events"),
        ("penndot_press_closures_pgh.csv", "press release closures"),
        ("dataset_prediction_ready.csv", "~19,000 rows, ~30 cols"),
        ("storm_events.csv", "~126 storm events"),
        ("storm_labels.csv", "~2-3M segment-storm labels"),
        ("training_matrix.parquet", "~2.8M rows, ~28 features"),
        ("model_lgbm.txt", "LightGBM model"),
        ("model_metadata.json", "model hyperparams + metrics"),
        ("feature_importance.csv", "feature importance rankings"),
        ("predictions_latest.csv", "~19,000 risk predictions"),
        ("predictions_latest.json", "GeoJSON predictions"),
    ]

    for filename, expected in output_files:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 1_000_000:
                size_str = f"{size/1_000_000:.1f} MB"
            else:
                size_str = f"{size/1_000:.1f} KB"
            print(f"  {filename:40s} : {size_str} (expected: {expected})")
        else:
            print(f"  {filename:40s} : MISSING")

    failed = sum(1 for s in results.values() if not s)
    if failed:
        print(f"\n{failed} script(s) failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
