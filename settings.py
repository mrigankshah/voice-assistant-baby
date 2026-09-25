"""Persistent, local preferences for the voice assistant."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock


SETTINGS_PATH = Path(__file__).with_name("assistant_settings.json")
_write_lock = Lock()


class SettingsError(Exception):
    """A preference is invalid or the settings file cannot be used."""


def normalize_temperature_unit(value: str) -> str:
    if not isinstance(value, str):
        raise SettingsError("Temperature unit must be Celsius or Fahrenheit.")
    cleaned = value.strip().casefold().replace("°", "")
    if cleaned in ("c", "celsius", "celcius", "centigrade"):
        return "celsius"
    if cleaned in ("f", "fahrenheit"):
        return "fahrenheit"
    raise SettingsError("Temperature unit must be Celsius or Fahrenheit.")


def load_settings(*, path: Path = SETTINGS_PATH) -> dict[str, str]:
    try:
        with path.open(encoding="utf-8") as settings_file:
            saved = json.load(settings_file)
    except FileNotFoundError:
        return {
            "default_city": os.environ.get("WEATHER_DEFAULT_LOCATION", "").strip(),
            "temperature_unit": "fahrenheit",
        }
    except (OSError, ValueError) as exc:
        raise SettingsError("Cannot read assistant settings.") from exc
    if not isinstance(saved, dict):
        raise SettingsError("Assistant settings have an invalid format.")
    city = saved.get("default_city", "")
    unit = saved.get("temperature_unit", "fahrenheit")
    if not isinstance(city, str) or len(city) > 120:
        raise SettingsError("Saved default city is invalid.")
    try:
        unit = normalize_temperature_unit(unit)
    except SettingsError as exc:
        raise SettingsError("Saved temperature unit is invalid.") from exc
    return {"default_city": city, "temperature_unit": unit}


def update_settings(
    *, default_city: str | None = None,
    temperature_unit: str | None = None,
    path: Path = SETTINGS_PATH,
) -> dict[str, str]:
    if default_city is None and temperature_unit is None:
        raise SettingsError("Give a default city or temperature unit to change.")
    with _write_lock:
        settings = load_settings(path=path)
        if default_city is not None:
            if not isinstance(default_city, str):
                raise SettingsError("Default city must be text.")
            city = default_city.strip()
            if len(city) > 120:
                raise SettingsError("Default city is too long.")
            settings["default_city"] = city
        if temperature_unit is not None:
            settings["temperature_unit"] = normalize_temperature_unit(temperature_unit)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(settings, temporary_file, indent=2)
                temporary_file.write("\n")
            os.replace(temporary_path, path)
        except OSError as exc:
            raise SettingsError("Cannot save assistant settings.") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
    return settings
