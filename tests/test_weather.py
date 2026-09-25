"""Weather tool tests without an internet connection."""

import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from weather import WeatherError, get_weather


class WeatherTests(unittest.TestCase):
    def setUp(self):
        settings = patch("weather.load_settings", return_value={"default_city": "", "temperature_unit": "fahrenheit"})
        self.settings = settings.start()
        self.addCleanup(settings.stop)

    @staticmethod
    def fake_request(url):
        parsed = urlsplit(url)
        if "geocoding" in parsed.hostname:
            return {"results": [{"name": "Boston", "admin1": "Massachusetts", "country": "United States", "latitude": 42.36, "longitude": -71.06}]}
        return {
            "timezone": "America/New_York",
            "current": {"time": "2026-09-25T10:00", "temperature_2m": 65, "apparent_temperature": 64, "weather_code": 3},
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
        self.settings.return_value = {"default_city": "Boston", "temperature_unit": "fahrenheit"}
        with patch("weather._get_json", side_effect=self.fake_request):
            result = get_weather()
        self.assertEqual(result["current"]["temperature"], 65)
        self.assertEqual(result["temperature_unit"], "F")
        self.assertNotIn("wind_speed_mph", result["current"])

    def test_missing_location_gives_clear_error(self):
        with self.assertRaisesRegex(WeatherError, "No location"):
            get_weather()

    def test_celsius_preference_changes_forecast_request_and_result_unit(self):
        self.settings.return_value = {"default_city": "Boston", "temperature_unit": "celsius"}
        with patch("weather._get_json", side_effect=self.fake_request) as fetch:
            result = get_weather()
        self.assertEqual(result["temperature_unit"], "C")
        parameters = parse_qs(urlsplit(fetch.call_args_list[1].args[0]).query)
        self.assertEqual(parameters["temperature_unit"], ["celsius"])
        self.assertNotIn("wind_speed_unit", parameters)
        self.assertNotIn("wind_speed_10m", parameters["current"][0])

    def test_date_outside_forecast_gives_clear_error(self):
        with patch("weather._get_json", side_effect=self.fake_request):
            with self.assertRaisesRegex(WeatherError, "outside"):
                get_weather("Boston", "2026-10-20")

    def test_yearless_date_uses_year_in_forecast(self):
        def fake_request(url):
            result = self.fake_request(url)
            if "geocoding" not in url:
                result["daily"]["time"].extend(["2026-09-27", "2026-09-28"])
                result["daily"]["temperature_2m_max"].extend([67, 66])
                result["daily"]["temperature_2m_min"].extend([52, 51])
                result["daily"]["precipitation_probability_max"].extend([30, 70])
                result["daily"]["weather_code"].extend([2, 61])
            return result

        with patch("weather._get_json", side_effect=fake_request):
            result = get_weather("Boston", "09-28")
        self.assertEqual(result["date"], "2026-09-28")
        self.assertEqual(result["forecast"]["condition"], "rain")

    def test_yearless_date_can_resolve_into_next_year(self):
        def fake_request(url):
            result = self.fake_request(url)
            if "geocoding" not in url:
                result["daily"]["time"] = ["2026-12-31", "2027-01-01"]
            return result

        with patch("weather._get_json", side_effect=fake_request):
            result = get_weather("Boston", "01-01")
        self.assertEqual(result["date"], "2027-01-01")


if __name__ == "__main__":
    unittest.main()
