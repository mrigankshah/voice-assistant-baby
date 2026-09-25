"""Check that the clock tool returns consistent local date and time fields."""

import unittest
from datetime import datetime

from local_clock import get_current_datetime


class ClockTests(unittest.TestCase):
    def test_current_datetime_has_timezone_and_matching_fields(self):
        result = get_current_datetime()
        observed = datetime.fromisoformat(result["iso_datetime"])
        self.assertIsNotNone(observed.utcoffset())
        self.assertEqual(result["date"], observed.date().isoformat())
        self.assertEqual(result["time"], observed.strftime("%H:%M:%S"))
        self.assertEqual(result["weekday"], observed.strftime("%A"))


if __name__ == "__main__":
    unittest.main()
