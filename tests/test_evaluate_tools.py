"""Verify evaluation scoring and artifacts without a real model."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_tools import CASE_PATH, build_messages, grade_response, load_cases, main, summarize


def reply(name=None, arguments=None, content=""):
    message = {"content": content}
    if name:
        message["tool_calls"] = [{"function": {"name": name, "arguments": arguments or {}}}]
    return {"done": True, "message": message}


class ToolEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "weather", "prompt": "Weather tomorrow?", "expected": [
            {"name": "get_weather", "arguments": {"day": "tomorrow"}},
        ]}

    def test_default_city_and_omission_pass_but_invented_city_fails(self):
        for arguments in ({"day": "tomorrow"}, {"location": "Boston", "day": "tomorrow"}):
            self.assertEqual(grade_response(reply("get_weather", arguments), self.case)["status"], "pass")
        result = grade_response(reply("get_weather", {"location": "San Francisco", "day": "tomorrow"}), self.case)
        self.assertEqual(result["status"], "wrong_arguments")

    def test_explicit_city_is_not_replaced_by_default(self):
        self.case["expected"][0]["arguments"]["location"] = "London"
        self.assertEqual(grade_response(reply("get_weather", {"day": "tomorrow"}), self.case)["status"], "wrong_arguments")

    def test_distinguishes_missing_unnecessary_and_wrong_tools(self):
        self.assertEqual(grade_response(reply(content="Sunny tomorrow."), self.case)["status"], "missing_tool")
        self.assertEqual(grade_response(reply("get_current_datetime"), self.case)["status"], "wrong_tools")
        self.assertEqual(grade_response(reply("get_weather"), {"expected": []})["status"], "unnecessary_tool")
        self.assertEqual(grade_response(reply(content="Hello"), {"expected": []})["status"], "pass")

    def test_empty_output_does_not_pass_no_tool_case(self):
        self.assertEqual(grade_response(reply(), {"expected": []})["status"], "empty_response")

    def test_malformed_arguments_are_errors(self):
        with self.assertRaises(ValueError):
            grade_response(reply("get_weather", "[]"), self.case)

    def test_cases_have_fixed_context_and_followup_history(self):
        cases = load_cases(CASE_PATH)
        case = next(case for case in cases if case["id"] == "weather-followup-day")
        messages = build_messages(case)
        self.assertIn("2026-09-25", messages[0]["content"])
        self.assertIn("city='Boston'", messages[0]["content"])
        self.assertEqual(messages[1:-1], case["history"])

    def test_full_run_preserves_raw_results_and_reports_failures(self):
        requests = []

        def request(path, payload=None, **kwargs):
            requests.append((path, payload))
            if path == "/api/version":
                return {"version": "test"}
            if path == "/api/show":
                return {"parameters": "temperature 0.1", "details": {"quantization_level": "Q4"}}
            if payload["messages"][-1]["content"] == "What's the weather?":
                return reply("get_weather", {"location": "Boston"})
            return reply("get_weather", {"location": "San Francisco", "day": "tomorrow"})

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            with patch("evaluate_tools.list_models", return_value=["test:latest"]), patch(
                "evaluate_tools.request_json", side_effect=request
            ), patch("weather.get_weather", side_effect=AssertionError("Must not execute tools")), contextlib.redirect_stdout(io.StringIO()):
                result = main(["--model", "test:latest", "--limit", "2", "--output", str(output)])
            self.assertEqual(result, 0)
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["pass", "wrong_arguments"])
            self.assertIn("San Francisco", json.dumps(rows[1]["response"]))
            self.assertIn("1/2", (output / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(summarize(rows)["pass_rate"], .5)
            chat_request = next(payload for path, payload in requests if path == "/api/chat")
            self.assertNotIn("temperature", chat_request["options"])
            self.assertFalse(chat_request["stream"])
            self.assertEqual(len(chat_request["tools"]), 4)


if __name__ == "__main__":
    unittest.main()
