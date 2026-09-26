import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_routing import main


class RoutingEvaluationTests(unittest.TestCase):
    def test_runner_records_all_cases_and_does_not_execute_workflows(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("evaluate_routing.__file__", str(Path(directory) / "evaluate_routing.py")), patch(
                "evaluate_routing.interpret_request", return_value={"intent": "chat"}), patch(
                "assistant.get_weather"
            ) as weather, patch("assistant.update_settings") as settings, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--model", "test"]), 0)
            output = next((Path(directory) / "eval-results").iterdir())
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 32)
            self.assertIn("13/32", (output / "report.md").read_text(encoding="utf-8"))
            followup = next(row for row in rows if row["id"] == "weather-followup-day")
            self.assertEqual(followup["active_task"]["location"], "London")
            weather.assert_not_called()
            settings.assert_not_called()
