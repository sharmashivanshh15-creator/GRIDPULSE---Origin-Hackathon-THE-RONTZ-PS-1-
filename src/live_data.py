import os
from datetime import datetime, timezone

import requests
import streamlit as st


ENERGYMAP_LATEST_URL = (
    "https://api.energymap.in/developer/v1/grid/demand/latest"
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

DELHI_LAT = 28.6139
DELHI_LON = 77.2090


def get_energymap_key():
    """Read API key from Streamlit secrets first, then environment."""
    try:
        key = st.secrets.get("ENERGYMAP_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("ENERGYMAP_API_KEY")


@st.cache_data(ttl=300)
def fetch_live_delhi_demand():
    """
    Fetch the latest Delhi demand record.

    The API response identifies the underlying source as
    Delhi SLDC loadcurve.
    """

    api_key = get_energymap_key()

    if not api_key:
        return {
            "ok": False,
            "error": "ENERGYMAP_API_KEY is not configured."
        }

    try:
        response = requests.get(
            ENERGYMAP_LATEST_URL,
            headers={"X-API-Key": api_key},
            timeout=15,
        )

        response.raise_for_status()
        payload = response.json()

        data = payload.get("data", {})
        records = data.get("items", []) if isinstance(data, dict) else []

        delhi = next(
            (
                item
                for item in records
                if str(item.get("state", "")).strip().lower() == "delhi"
            ),
            None,
        )

        if delhi is None:
            return {
                "ok": False,
                "error": "Delhi record was not found."
            }

        timestamp = delhi.get("timestamp")

        parsed_timestamp = None
        age_minutes = None

        if timestamp:
            parsed_timestamp = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

            now = datetime.now(timezone.utc)

            age_minutes = (
                now - parsed_timestamp.astimezone(timezone.utc)
            ).total_seconds() / 60

        if age_minutes is None:
            freshness = "UNKNOWN"
        elif age_minutes <= 15:
            freshness = "LIVE"
        elif age_minutes <= 60:
            freshness = "DELAYED"
        else:
            freshness = "STALE"

        return {
            "ok": True,
            "demand_mw": float(delhi["demand_mw"]),
            "peak_mw": (
                float(delhi["peak_mw"])
                if delhi.get("peak_mw") is not None
                else None
            ),
            "frequency_hz": (
                float(delhi["frequency_hz"])
                if delhi.get("frequency_hz") is not None
                else None
            ),
            "timestamp": timestamp,
            "source": delhi.get("source"),
            "source_type": delhi.get("source_type"),
            "source_kind": delhi.get("source_kind"),
            "age_minutes": age_minutes,
            "freshness": freshness,
        }

    except requests.RequestException as exc:
        return {
            "ok": False,
            "error": f"EnergyMap request failed: {exc}"
        }

    except (KeyError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "error": f"Unexpected EnergyMap response: {exc}"
        }


@st.cache_data(ttl=300)
def fetch_weather():
    """Fetch current and seven-day Delhi weather from Open-Meteo."""

    params = {
        "latitude": DELHI_LAT,
        "longitude": DELHI_LON,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "shortwave_radiation"
        ),
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        return {
            "ok": True,
            "data": response.json(),
        }

    except requests.RequestException as exc:
        return {
            "ok": False,
            "error": f"Open-Meteo request failed: {exc}"
        }
