"""Contract tests for the local Ollama requests and model picker."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from assistant import ChatCancellation, ChatInterrupted, _relative_day_from_prompt, _simple_setting_command, _tool_result, _yearless_day_from_prompt, answer_with_history, chat, choose_model, list_models


class FakeOllamaHandler(BaseHTTPRequestHandler):
    requests = []
    slow_chunk_sent = threading.Event()
    continue_slow_reply = threading.Event()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"models": [{"name": "zeta:latest"}, {"name": "alpha:latest"}]}).encode()
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if body["messages"][-1].get("content") in (
            "Weather in Boston tomorrow", "Weather in Boston tomorrow with stale date"
        ):
            stale_date = body["messages"][-1].get("content") == "Weather in Boston tomorrow with stale date"
            events = (
                {"message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": {"location": "Boston", "day": "2025-11-16" if stale_date else "tomorrow"}}}]}, "done": False},
                {"message": {"content": ""}, "done": True},
            )
        elif body["messages"][-1].get("content") == "Please report the current clock reading":
            events = (
                {"message": {"tool_calls": [{"function": {"name": "get_current_datetime", "arguments": {}}}]}, "done": False},
                {"message": {"content": ""}, "done": True},
            )
        elif body["messages"][-1].get("content") == "Weather in Boston on September 28th":
            events = (
                {"message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": {"location": "Boston", "day": "2025-09-28"}}}]}, "done": False},
                {"message": {"content": ""}, "done": True},
            )
        elif body["messages"][-1].get("content") == "Remember New York for future weather":
            events = (
                {"message": {"tool_calls": [{"function": {"name": "set_weather_preferences", "arguments": {"default_city": "New York"}}}]}, "done": False},
                {"message": {"content": ""}, "done": True},
            )
        elif body["messages"][-1].get("role") == "tool":
            reply = "The time is 10:15." if body["messages"][-1]["tool_name"] == "get_current_datetime" else "Boston will be rainy tomorrow."
            events = (
                {"message": {"content": reply[:len(reply) // 2]}, "done": False},
                {"message": {"content": reply[len(reply) // 2:]}, "done": False},
                {"message": {"content": ""}, "done": True},
            )
        else:
            events = (
                {"message": {"content": "Hello "}, "done": False},
                {"message": {"content": "from Ollama"}, "done": False},
                {"message": {"content": ""}, "done": True},
            )
        try:
            if body["messages"][-1]["content"] == "Slow reply":
                self.wfile.write((json.dumps(events[0]) + "\n").encode())
                self.wfile.flush()
                self.slow_chunk_sent.set()
                self.continue_slow_reply.wait(timeout=3)
                events = events[1:]
            for event in events:
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        except ConnectionError:
            pass

    def log_message(self, format, *args):
        pass


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_lists_models_from_local_ollama(self):
        self.assertEqual(list_models(base_url=self.base_url), ["alpha:latest", "zeta:latest"])

    def test_sends_selected_model_and_conversation_to_ollama(self):
        messages = [{"role": "user", "content": "Hello"}]
        chunks = []
        self.assertEqual(
            chat("alpha:latest", messages, on_chunk=chunks.append, base_url=self.base_url),
            "Hello from Ollama",
        )
        self.assertEqual(chunks, ["Hello ", "from Ollama"])
        self.assertEqual(
            FakeOllamaHandler.requests[-1],
            ("/api/chat", {"model": "alpha:latest", "messages": messages, "stream": True}),
        )

    def test_picker_retries_invalid_choice(self):
        answers = iter(["not a number", "3", "2"])
        output = []
        selected = choose_model(["alpha:latest", "zeta:latest"], read=lambda _: next(answers), write=output.append)
        self.assertEqual(selected, "zeta:latest")
        self.assertEqual(output.count("Enter a number from 1 to 2."), 2)

    def test_stream_stops_after_interruption(self):
        cancellation = ChatCancellation()
        received = []

        def on_chunk(chunk):
            received.append(chunk)
            cancellation.cancel()

        with self.assertRaises(ChatInterrupted):
            chat(
                "alpha:latest",
                [{"role": "user", "content": "Long answer"}],
                on_chunk=on_chunk,
                cancellation=cancellation,
                base_url=self.base_url,
            )
        self.assertEqual(received, ["Hello "])

    def test_interruption_allows_a_new_request_without_waiting(self):
        FakeOllamaHandler.slow_chunk_sent.clear()
        FakeOllamaHandler.continue_slow_reply.clear()
        cancellation = ChatCancellation()
        result = []

        def run_chat():
            try:
                chat(
                    "alpha:latest",
                    [{"role": "user", "content": "Slow reply"}],
                    cancellation=cancellation,
                    base_url=self.base_url,
                )
            except ChatInterrupted:
                result.append("interrupted")

        worker = threading.Thread(target=run_chat, daemon=True)
        try:
            worker.start()
            self.assertTrue(FakeOllamaHandler.slow_chunk_sent.wait(timeout=1))
            cancellation.cancel()
            new_reply = chat(
                "alpha:latest",
                [{"role": "user", "content": "New question"}],
                base_url=self.base_url,
            )
            self.assertEqual(new_reply, "Hello from Ollama")
        finally:
            FakeOllamaHandler.continue_slow_reply.set()
            worker.join(timeout=3)
        self.assertEqual(result, ["interrupted"])

    def test_answer_remembers_previous_turn(self):
        _, history = answer_with_history("alpha:latest", "Hello", [], base_url=self.base_url)
        answer_with_history("alpha:latest", "Follow up", history, base_url=self.base_url)
        self.assertEqual(
            FakeOllamaHandler.requests[-1][1]["messages"][-3:],
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from Ollama"},
                {"role": "user", "content": "Follow up"},
            ],
        )

    def test_weather_tool_runs_and_final_answer_streams(self):
        chunks = []
        debug = []
        with patch("assistant.get_weather", return_value={"forecast": {"condition": "rain"}}) as weather:
            reply, history = answer_with_history(
                "alpha:latest", "Weather in Boston tomorrow", [],
                on_chunk=chunks.append, on_tool_debug=debug.append, base_url=self.base_url,
            )
        weather.assert_called_once_with("Boston", "tomorrow")
        self.assertEqual(reply, "Boston will be rainy tomorrow.")
        self.assertEqual("".join(chunks), reply)
        self.assertEqual(history[-1], {"role": "assistant", "content": reply})
        tool_result = FakeOllamaHandler.requests[-1][1]["messages"][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertEqual(json.loads(tool_result["content"]), {"forecast": {"condition": "rain"}})
        self.assertIn('request get_weather: {"location": "Boston", "day": "tomorrow"}', debug[0])
        self.assertIn('result get_weather', debug[1])
        self.assertIn('"condition": "rain"', debug[1])
        self.assertEqual(len(debug), 2)

    def test_debug_reports_when_model_did_not_request_a_tool(self):
        debug = []
        answer_with_history(
            "alpha:latest", "Hello", [], on_tool_debug=debug.append, base_url=self.base_url
        )
        self.assertEqual(debug, ["[tool] Ollama replied without requesting a tool"])

    def test_tomorrow_ignores_a_stale_date_from_the_model(self):
        debug = []
        with patch("assistant.get_weather", return_value={"forecast": {"condition": "rain"}}) as weather:
            answer_with_history(
                "alpha:latest", "Weather in Boston tomorrow with stale date", [],
                on_tool_debug=debug.append, base_url=self.base_url,
            )
        weather.assert_called_once_with("Boston", "tomorrow")
        self.assertIn("using day='tomorrow'", debug[1])

    def test_comparison_of_two_days_does_not_override_tool_arguments(self):
        self.assertIsNone(_relative_day_from_prompt("Compare today and tomorrow in Boston"))
        self.assertIsNone(_relative_day_from_prompt("Tomorrow in Boston and Saturday in New York"))

    def test_yearless_month_and_day_ignores_a_stale_model_year(self):
        debug = []
        with patch("assistant.get_weather", return_value={"date": "2026-09-28"}) as weather:
            answer_with_history(
                "alpha:latest", "Weather in Boston on September 28th", [],
                on_tool_debug=debug.append, base_url=self.base_url,
            )
        weather.assert_called_once_with("Boston", "09-28")
        self.assertIn("using day='09-28'", debug[1])
        self.assertIsNone(_yearless_day_from_prompt("September 28th, 2027"))
        self.assertIsNone(_yearless_day_from_prompt("September 28th and October 1st"))

    def test_clock_tool_returns_the_pi_clock_to_the_model(self):
        clock = {"date": "2026-09-25", "time": "10:15:00", "weekday": "Friday", "timezone": "EDT", "utc_offset": "-0400", "iso_datetime": "2026-09-25T10:15:00-04:00"}
        with patch("assistant.get_current_datetime", return_value=clock):
            reply, _ = answer_with_history(
                "alpha:latest", "Please report the current clock reading", [], base_url=self.base_url,
            )
        self.assertEqual(reply, "The time is 10:15.")
        messages = FakeOllamaHandler.requests[-1][1]["messages"]
        self.assertIn("2026-09-25", messages[0]["content"])
        self.assertEqual(messages[-1]["tool_name"], "get_current_datetime")
        self.assertEqual(json.loads(messages[-1]["content"]), clock)

    def test_direct_clock_answers_only_requested_parts(self):
        clock = {"date": "2026-09-25", "time": "10:15:00", "weekday": "Friday", "timezone": "EDT", "utc_offset": "-0400", "iso_datetime": "2026-09-25T10:15:00-04:00"}
        before = len(FakeOllamaHandler.requests)
        debug = []
        with patch("assistant.get_current_datetime", return_value=clock):
            time_reply, _ = answer_with_history(
                "alpha:latest", "What time is it?", [],
                on_tool_debug=debug.append, base_url=self.base_url,
            )
            date_reply, _ = answer_with_history(
                "alpha:latest", "What's today's date?", [], base_url=self.base_url,
            )
            both_reply, _ = answer_with_history(
                "alpha:latest", "What's the date and time?", [], base_url=self.base_url,
            )
        self.assertEqual(len(FakeOllamaHandler.requests), before)
        self.assertEqual(time_reply, "10:15 AM.")
        self.assertEqual(date_reply, "September 25, 2026.")
        self.assertEqual(both_reply, "10:15 AM on September 25, 2026.")
        self.assertIn('"time": "10:15:00"', debug[1])
        self.assertNotIn('"date":', debug[1])

    def test_clock_tool_shares_only_requested_part_for_other_phrasings(self):
        clock = {"date": "2026-09-25", "time": "10:15:00", "weekday": "Friday", "timezone": "EDT", "utc_offset": "-0400", "iso_datetime": "2026-09-25T10:15:00-04:00"}
        call = {"function": {"name": "get_current_datetime", "arguments": {}}}
        with patch("assistant.get_current_datetime", return_value=clock):
            _, time_result = _tool_result(call, "Could you tell me the time?")
            _, date_result = _tool_result(call, "Could you tell me the date?")
        self.assertEqual(json.loads(time_result), {"time": "10:15:00", "timezone": "EDT"})
        self.assertEqual(json.loads(date_result), {"date": "2026-09-25", "weekday": "Friday"})

    def test_spoken_setting_commands_save_and_confirm_without_ollama(self):
        before = len(FakeOllamaHandler.requests)
        chunks = []
        debug = []
        with patch("assistant.update_settings", side_effect=[
            {"default_city": "Boston", "temperature_unit": "fahrenheit"},
            {"default_city": "Boston", "temperature_unit": "celsius"},
        ]) as save:
            city_reply, history = answer_with_history(
                "alpha:latest", "Set my default city to Boston", [],
                on_chunk=chunks.append, on_tool_debug=debug.append, base_url=self.base_url,
            )
            unit_reply, history = answer_with_history(
                "alpha:latest", "Use Celsius by default", history,
                on_chunk=chunks.append, on_tool_debug=debug.append, base_url=self.base_url,
            )
        self.assertEqual(len(FakeOllamaHandler.requests), before)
        self.assertEqual(save.call_args_list[0].kwargs, {"default_city": "Boston"})
        self.assertEqual(save.call_args_list[1].kwargs, {"temperature_unit": "celsius"})
        self.assertEqual(city_reply, "Default city set to Boston.")
        self.assertEqual(unit_reply, "Default temperature unit set to Celsius.")
        self.assertEqual(chunks, [city_reply, unit_reply])
        self.assertIn("request set_weather_preferences", debug[0])
        self.assertEqual(history[-1]["content"], unit_reply)

    def test_natural_spoken_setting_phrases(self):
        self.assertEqual(
            _simple_setting_command("Can you set my default city to New York?"),
            {"default_city": "New York"},
        )
        self.assertEqual(
            _simple_setting_command("I want the temperature in Celsius instead"),
            {"temperature_unit": "celsius"},
        )
        self.assertIsNone(_simple_setting_command("What's the temperature in Celsius today?"))

    def test_model_can_change_weather_preference_with_tool(self):
        with patch("assistant.update_settings", return_value={
            "default_city": "New York", "temperature_unit": "fahrenheit"
        }) as save:
            answer_with_history(
                "alpha:latest", "Remember New York for future weather", [], base_url=self.base_url
            )
        save.assert_called_once_with(default_city="New York")
        tool_message = FakeOllamaHandler.requests[-1][1]["messages"][-1]
        self.assertEqual(tool_message["tool_name"], "set_weather_preferences")
        self.assertEqual(json.loads(tool_message["content"])["default_city"], "New York")

    def test_interrupt_after_weather_lookup_skips_final_model_request(self):
        cancellation = ChatCancellation()
        before = len(FakeOllamaHandler.requests)

        def interrupted_lookup(*_args):
            cancellation.cancel()
            return {"forecast": {"condition": "rain"}}

        with patch("assistant.get_weather", side_effect=interrupted_lookup):
            with self.assertRaises(ChatInterrupted):
                answer_with_history(
                    "alpha:latest", "Weather in Boston tomorrow", [],
                    cancellation=cancellation, base_url=self.base_url,
                )
        self.assertEqual(len(FakeOllamaHandler.requests) - before, 1)


if __name__ == "__main__":
    unittest.main()
