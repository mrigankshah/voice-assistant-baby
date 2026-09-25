"""Weather tool tests without an internet connection."""

import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from weather import WeatherError, get_weather


class WeatherTests(unittest.TestCase):
    @staticmethod
    def fake_request(url):
        parsed = urlsplit(url)
        if "geocoding" in parsed.hostname:
            return {"results": [{"name": "Boston", "admin1": "Massachusetts", "country": "United States", "latitude": 42.36, "longitude": -71.06}]}
        return {
            "timezone": "America/New_York",
            "current": {"time": "2026-09-25T10:00", "temperature_2m": 65, "apparent_temperature": 64, "wind_speed_10m": 8, "weather_code": 3},
            "daily": {
                "time": ["2026-09-25", "2026-09-26"],
                "temperature_2m_max": [70, 68],
                "temperature_2m_min": [55, 53],
                "precipitation_probability_max": [10, 80],
                "weather_code": [3, 61],
            },
        }

    def test_forecast_uses_location_and_tomorrow(self):
        with patch("weather._get_json", side_effect=self.fake_request) as fetch:
            result = get_weather("Boston", "tomorrow")
        self.assertEqual(result["location"], "Boston, Massachusetts, United States")
        self.assertEqual(result["date"], "2026-09-26")
        self.assertEqual(result["forecast"]["condition"], "rain")
        self.assertNotIn("current", result)
        self.assertEqual(parse_qs(urlsplit(fetch.call_args_list[1].args[0]).query)["timezone"], ["auto"])

    def test_current_weather_and_default_location(self):
        with patch.dict("weather.os.environ", {"WEATHER_DEFAULT_LOCATION": "Boston"}):
            with patch("weather._get_json", side_effect=self.fake_request):
                result = get_weather()
        self.assertEqual(result["current"]["temperature"], 65)
        self.assertEqual(result["temperature_unit"], "F")

    def test_missing_location_gives_clear_error(self):
        with patch.dict("weather.os.environ", {}, clear=True):
            with self.assertRaisesRegex(WeatherError, "No location"):
                get_weather()

    def test_date_outside_forecast_gives_clear_error(self):
        with patch("weather._get_json", side_effect=self.fake_request):
            with self.assertRaisesRegex(WeatherError, "outside"):
                get_weather("Boston", "2026-10-20")


if __name__ == "__main__":
    unittest.main()
