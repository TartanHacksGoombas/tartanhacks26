"""
Collect historical winter weather data for Pittsburgh using Open-Meteo Archive API.

Fetches daily weather data for 10 winter seasons (Nov-Mar, 2014-2015 through
2024-2025). Snowfall, temperature, wind, precipitation.

Source: https://archive-api.open-meteo.com/v1/archive
Output: weather_historical_pgh.csv (~1,510 rows)

Requires: pip install requests
"""

import csv
import os
import time
import requests

# Pittsburgh coordinates (downtown)
PGH_LAT = 40.4406
PGH_LNG = -79.9959

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_CSV = "weather_historical_pgh.csv"

FIELDNAMES = [
    "date", "snowfall_cm", "temp_max_c", "temp_min_c", "wind_max_kmh",
    "precip_mm", "rain_mm", "weather_code", "data_source",
]

# Winter seasons: November 1 through March 31
SEASONS = []
for year in range(2014, 2025):
    SEASONS.append({
        "label": f"{year}-{year+1}",
        "start": f"{year}-11-01",
        "end": f"{year+1}-03-31",
    })

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, params=None):
    """Fetch with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = RETRY_BACKOFF ** (attempt + 1)
                print(f"  HTTP {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF ** (attempt + 1)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after {MAX_RETRIES} retries")


def fetch_season(start_date, end_date):
    """Fetch daily weather data for a date range."""
    params = {
        "latitude": PGH_LAT,
        "longitude": PGH_LNG,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "snowfall_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_sum,rain_sum,weather_code",
        "timezone": "America/New_York",
    }
    resp = fetch_with_retry(ARCHIVE_URL, params=params)
    return resp.json()


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    all_rows = []

    for i, season in enumerate(SEASONS):
        print(f"Fetching season {season['label']} ({i+1}/{len(SEASONS)})...")
        try:
            data = fetch_season(season["start"], season["end"])
            daily = data.get("daily", {})

            dates = daily.get("time", [])
            snowfall = daily.get("snowfall_sum", [])
            temp_max = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])
            wind_max = daily.get("wind_speed_10m_max", [])
            precip = daily.get("precipitation_sum", [])
            rain = daily.get("rain_sum", [])
            weather_code = daily.get("weather_code", [])

            for j, date in enumerate(dates):
                all_rows.append({
                    "date": date,
                    "snowfall_cm": snowfall[j] if j < len(snowfall) else "",
                    "temp_max_c": temp_max[j] if j < len(temp_max) else "",
                    "temp_min_c": temp_min[j] if j < len(temp_min) else "",
                    "wind_max_kmh": wind_max[j] if j < len(wind_max) else "",
                    "precip_mm": precip[j] if j < len(precip) else "",
                    "rain_mm": rain[j] if j < len(rain) else "",
                    "weather_code": weather_code[j] if j < len(weather_code) else "",
                    "data_source": "open_meteo",
                })

            print(f"  Got {len(dates)} days")
        except Exception as e:
            print(f"  Error fetching {season['label']}: {e}")

        if i < len(SEASONS) - 1:
            time.sleep(1)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    snow_days = sum(1 for r in all_rows
                    if r["snowfall_cm"] and r["snowfall_cm"] not in ("", "0", "0.0", 0, 0.0))
    print(f"\nSaved {len(all_rows)} days to {OUTPUT_CSV}")
    print(f"  Days with snowfall > 0: {snow_days}")


if __name__ == "__main__":
    main()
