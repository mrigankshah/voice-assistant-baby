"""Persistent weather preference tests using a temporary file."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from settings import SettingsError, load_settings, update_settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "assistant_settings.json"

    def test_changes_survive_reload_and_preserve_other_preference(self):
        with patch.dict("settings.os.environ", {}, clear=True):
            update_settings(default_city="Boston, Massachusetts", path=self.path)
            update_settings(temperature_unit="celcius", path=self.path)
            self.assertEqual(
                load_settings(path=self.path),
                {"default_city": "Boston, Massachusetts", "temperature_unit": "celsius"},
            )

    def test_saved_city_overrides_environment_fallback(self):
        with patch.dict("settings.os.environ", {"WEATHER_DEFAULT_LOCATION": "New York"}):
            self.assertEqual(load_settings(path=self.path)["default_city"], "New York")
            update_settings(default_city="Boston", path=self.path)
            self.assertEqual(load_settings(path=self.path)["default_city"], "Boston")
            update_settings(default_city="", path=self.path)
            self.assertEqual(load_settings(path=self.path)["default_city"], "")

    def test_bad_unit_does_not_overwrite_saved_settings(self):
        update_settings(default_city="Boston", path=self.path)
        with self.assertRaises(SettingsError):
            update_settings(temperature_unit="kelvin", path=self.path)
        self.assertEqual(load_settings(path=self.path)["default_city"], "Boston")
        self.assertEqual(load_settings(path=self.path)["temperature_unit"], "fahrenheit")


if __name__ == "__main__":
    unittest.main()
