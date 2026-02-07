# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Snow Road Closure Prediction App for Pittsburgh. Predicts which roads will close during snowstorms and plans routes around them. Built for TartanHacks 2026.

The project is early-stage. The `main` branch has the project vision; the `road-data-demo` branch has initial data collection scripts.

## Architecture

Three standalone Python scripts for collecting road data around Carnegie Mellon University:

- **roads_osm.py** — Pulls road data from OpenStreetMap via the Overpass API (free, no key needed). Outputs `roads_osm_cmu.csv` and `roads_osm_cmu.json`. Rich attributes: lanes, surface, speed limits, road type, etc.
- **roads_demo.py** — Pulls road data via Google Maps APIs (Geocoding, Roads, Directions, Places). Requires `GOOGLE_MAPS_API_KEY` env var. Outputs `roads_around_cmu.csv`.
- **roads_analysis.py** — Joins OSM and Google Maps CSVs into a deduplicated DataFrame using pandas. Outputs `roads_joined_cmu.csv`.

## Running the Scripts

```bash
# OSM data (no API key needed)
pip install requests
python roads_osm.py

# Google Maps data (needs API key)
pip install googlemaps
export GOOGLE_MAPS_API_KEY='your-key'
python roads_demo.py

# Join both datasets (run after the above two)
pip install pandas
python roads_analysis.py
```

## Dependencies

No requirements.txt yet. Current dependencies: `requests`, `googlemaps`, `pandas`.
