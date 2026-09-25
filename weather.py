"""Read-only weather lookups using Open-Meteo's public APIs."""

from __future__ import annotations

import json
import re
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from settings import SettingsError, load_settings


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherError(Exception):
    """A location, forecast, or network lookup failed."""


def _get_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=10) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise WeatherError(f"The weather service returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise WeatherError("The weather service is unavailable right now.") from exc
    except ValueError as exc:
        raise WeatherError("The weather service returned invalid data.") from exc
    if not isinstance(result, dict) or result.get("error"):
        raise WeatherError("The weather service returned an error.")
    return result


def _condition(code: object) -> str | None:
    if not isinstance(code, int):
        return None
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "partly cloudy" if code != 3 else "overcast"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55, 56, 57):
        return "drizzle"
    if code in (61, 63, 65, 66, 67):
        return "rain"
    if code in (71, 73, 75, 77):
        return "snow"
    if code in (80, 81, 82):
        return "rain showers"
    if code in (85, 86):
        return "snow showers"
    if code in (95, 96, 99):
        return "thunderstorms"
    return None


def get_weather(location: str = "", day: str = "today") -> dict:
    """Return current conditions and one daily forecast in the location's time zone."""
    if not isinstance(location, str) or not isinstance(day, str):
        raise WeatherError("Location and day must be text.")
    try:
        settings = load_settings()
    except SettingsError as exc:
        raise WeatherError(str(exc)) from exc
    location = location.strip() or settings["default_city"]
    if not location:
        raise WeatherError("No location was given. Ask the user for a city or set a default city.")
    if len(location) > 120:
        raise WeatherError("The location name is too long.")
    day = day.strip().lower()
    yearless_day = re.fullmatch(r"\d{2}-\d{2}", day) is not None
    if yearless_day:
        try:
            date.fromisoformat(f"2000-{day}")
        except ValueError as exc:
            raise WeatherError("Use a valid month and day in MM-DD format.") from exc
    elif day not in ("today", "tomorrow", "now"):
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise WeatherError("Use today, tomorrow, MM-DD, or a date like 2026-09-25.") from exc

    geocoding = _get_json(f"{GEOCODING_URL}?{urlencode({'name': location, 'count': 1, 'language': 'en'})}")
    places = geocoding.get("results")
    if not isinstance(places, list) or not places or not isinstance(places[0], dict):
        raise WeatherError(f"No location found for {location!r}. Ask for a more specific city.")
    place = places[0]
    try:
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
        place_name = ", ".join(
            part for part in (place.get("name"), place.get("admin1"), place.get("country"))
            if isinstance(part, str) and part
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError("The location service returned incomplete data.") from exc

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "forecast_days": 16,
        "temperature_unit": settings["temperature_unit"],
        "precipitation_unit": "inch",
        "current": "temperature_2m,apparent_temperature,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
    }
    forecast = _get_json(f"{FORECAST_URL}?{urlencode(parameters)}")
    daily = forecast.get("daily")
    if not isinstance(daily, dict) or not isinstance(daily.get("time"), list):
        raise WeatherError("The weather service returned incomplete forecast data.")
    dates = daily["time"]
    index = 0 if day in ("today", "now") else 1 if day == "tomorrow" else -1
    if index == -1 and yearless_day:
        index = next((number for number, value in enumerate(dates) if value[5:] == day), -1)
    if index == -1:
        try:
            index = dates.index(day)
        except ValueError as exc:
            raise WeatherError("That date is outside the available 16-day forecast.") from exc
    if index >= len(dates):
        raise WeatherError("That date is outside the available forecast.")

    def daily_value(key: str) -> object:
        values = daily.get(key)
        return values[index] if isinstance(values, list) and index < len(values) else None

    result = {
        "source": "Open-Meteo",
        "location": place_name,
        "timezone": forecast.get("timezone"),
        "date": dates[index],
        "temperature_unit": "C" if settings["temperature_unit"] == "celsius" else "F",
        "forecast": {
            "high": daily_value("temperature_2m_max"),
            "low": daily_value("temperature_2m_min"),
            "precipitation_probability_percent": daily_value("precipitation_probability_max"),
            "condition": _condition(daily_value("weather_code")),
        },
    }
    if index == 0 and isinstance(forecast.get("current"), dict):
        current = forecast["current"]
        result["current"] = {
            "observed_at": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "condition": _condition(current.get("weather_code")),
        }
    return result
