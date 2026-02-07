"""
Collect current weather forecast and winter alerts for Pittsburgh from NWS API.

Source: https://api.weather.gov
  - Forecast: grid PBZ/78,66
  - Alerts: zone PAC003 (Allegheny County)

Output: weather_forecast_pgh.json, weather_alerts_pgh.json

Requires: pip install requests
"""

import json
import os
import time
import requests

FORECAST_URL = "https://api.weather.gov/gridpoints/PBZ/78,66/forecast"
ALERTS_URL = "https://api.weather.gov/alerts/active"

FORECAST_OUTPUT = "weather_forecast_pgh.json"
ALERTS_OUTPUT = "weather_alerts_pgh.json"

HEADERS = {
    "User-Agent": "(SnowRoadClosurePrediction, tartanhacks2026@example.com)",
    "Accept": "application/geo+json",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def fetch_with_retry(url, params=None):
    """Fetch with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
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
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries")


def fetch_forecast():
    """Fetch 7-day forecast for Pittsburgh."""
    print("Fetching NWS forecast for Pittsburgh...")
    resp = fetch_with_retry(FORECAST_URL)
    data = resp.json()

    periods = data.get("properties", {}).get("periods", [])
    print(f"  Got {len(periods)} forecast periods")

    for p in periods[:4]:
        print(f"  {p['name']}: {p['shortForecast']}, {p['temperature']}°{p['temperatureUnit']}")

    return data


def fetch_alerts():
    """Fetch active winter weather alerts for Allegheny County."""
    print("\nFetching active weather alerts for Allegheny County...")
    resp = fetch_with_retry(ALERTS_URL, params={"zone": "PAC003"})
    data = resp.json()

    features = data.get("features", [])
    print(f"  Got {len(features)} active alerts")

    # Filter for winter-related alerts
    winter_keywords = ["snow", "ice", "winter", "freeze", "blizzard", "frost", "cold"]
    winter_alerts = []
    for f in features:
        props = f.get("properties", {})
        event = (props.get("event") or "").lower()
        headline = (props.get("headline") or "").lower()
        if any(kw in event or kw in headline for kw in winter_keywords):
            winter_alerts.append(f)
            print(f"  WINTER ALERT: {props.get('event')} - {props.get('headline')}")

    if not winter_alerts and features:
        print("  (no winter-specific alerts active)")
    elif not features:
        print("  (no active alerts)")

    return data


def main():
    # Forecast
    forecast = fetch_forecast()
    with open(FORECAST_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)
    print(f"  Saved to {FORECAST_OUTPUT}")

    # Alerts
    alerts = fetch_alerts()
    with open(ALERTS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)
    print(f"  Saved to {ALERTS_OUTPUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
