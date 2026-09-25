"""The production request interpreter and Python workflows."""

import json
import unittest
from unittest.mock import call, patch

from assistant import ChatCancellation, ChatInterrupted, OllamaError, answer_with_history
from structured_assistant import ConversationHistory, ROUTER_SCHEMA, interpret_request


def plan(intent, **fields):
    value = {
        "intent": intent, "location": None, "day": None, "clock_part": None,
        "setting_name": None, "setting_value": None, "question": None,
    }
    value.update(fields)
    return {"role": "assistant", "content": json.dumps(value)}


WEATHER = {
    "location": "London, England", "date": "2026-09-26", "temperature_unit": "C",
    "forecast": {"condition": "rain", "high": 18, "low": 11, "precipitation_probability_percent": 70},
}

CLOCK = {
    "date": "2026-09-25", "time": "10:15:00", "weekday": "Friday", "timezone": "EDT",
    "utc_offset": "-0400", "iso_datetime": "2026-09-25T10:15:00-04:00",
}


class StructuredAssistantTests(unittest.TestCase):
    def test_greeting_after_weather_cannot_trigger_weather(self):
        history = ConversationHistory(active_task={"intent": "weather", "location": "London", "day": "today"})
        with patch("assistant._stream_chat", return_value=plan("weather")) as router, patch(
            "assistant.chat", return_value="I'm here and ready to help."
        ), patch("assistant.get_weather") as weather:
            answer_with_history("local", "How are you?", history)
        router.assert_not_called()
        weather.assert_not_called()

    def test_time_question_cannot_be_routed_to_settings(self):
        with patch("assistant._stream_chat", return_value=plan("read_setting")) as router, patch(
            "assistant.load_settings"
        ) as settings:
            reply, _ = answer_with_history("local", "What time is it?", [])
        self.assertEqual(reply, "10:15 AM.")
        router.assert_not_called()
        settings.assert_not_called()

    def test_unrelated_weather_action_is_rejected(self):
        with patch("assistant._stream_chat", return_value=plan("weather")), patch("assistant.get_weather") as weather:
            reply, _ = answer_with_history("local", "Are penguins birds?", [])
        weather.assert_not_called()
        self.assertIn("rephrase", reply)

    def test_router_sees_schema_and_no_unrelated_active_task(self):
        with patch("assistant._stream_chat", return_value=plan("chat")) as request:
            interpret_request("local", "Explain gravity", {"intent": "weather", "location": "London"})
        messages = request.call_args.args[1]
        self.assertEqual(len(messages), 2)
        self.assertIn("Output schema:", messages[0]["content"])
        self.assertIn("This is a new request", messages[0]["content"])
        self.assertNotIn("Active task for this follow-up", messages[0]["content"])
        self.assertEqual(request.call_args.kwargs["options"]["temperature"], 0)

    def setUp(self):
        clock_patch = patch("assistant.get_current_datetime", return_value=CLOCK)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        settings_patch = patch("assistant.load_settings", return_value={"default_city": "Boston", "temperature_unit": "celsius"})
        self.settings = settings_patch.start()
        self.addCleanup(settings_patch.stop)

    def test_weather_uses_saved_default_and_formats_verified_result(self):
        chunks, debug = [], []
        with patch("assistant._stream_chat", return_value=plan("weather", day="tomorrow")) as interpret, patch(
            "assistant.get_weather", return_value=WEATHER
        ) as weather, patch("assistant.chat") as chat:
            reply, history = answer_with_history(
                "local", "Will it rain tomorrow?", [], on_chunk=chunks.append, on_tool_debug=debug.append,
            )
        weather.assert_called_once_with("Boston", "tomorrow")
        chat.assert_not_called()
        self.assertIn("70% chance of rain", reply)
        self.assertEqual(chunks, [reply])
        self.assertEqual(history.active_task, {"intent": "weather", "location": "Boston", "day": "tomorrow"})
        self.assertIn("saved default", " ".join(debug))
        self.assertEqual(interpret.call_args.kwargs["format_schema"], ROUTER_SCHEMA)
        self.assertNotIn("tools", interpret.call_args.kwargs)

    def test_explicit_city_and_followup_remember_place_and_day(self):
        with patch("assistant._stream_chat", side_effect=[
            plan("weather", location="London", day="today"),
            plan("weather", day="tomorrow"),
            plan("weather", location="Paris"),
        ]), patch("assistant.get_weather", return_value=WEATHER) as weather:
            _, history = answer_with_history("local", "Weather in London today?", [])
            _, history = answer_with_history("local", "And tomorrow?", history)
            _, history = answer_with_history("local", "What about Paris?", history)
        self.assertEqual(weather.call_args_list, [
            call("London", "today"), call("London", "tomorrow"), call("Paris", "tomorrow"),
        ])
        self.assertEqual(history.active_task["location"], "Paris")
        history.clear()
        self.assertIsNone(history.active_task)

    def test_home_overrides_previous_city_with_saved_default(self):
        history = ConversationHistory(active_task={"intent": "weather", "location": "London", "day": "tomorrow"})
        with patch("assistant._stream_chat", return_value=plan("weather")), patch(
            "assistant.get_weather", return_value=WEATHER
        ) as weather:
            answer_with_history("local", "And what's the weather at home?", history)
        weather.assert_called_once_with("Boston", "tomorrow")

    def test_normal_chat_uses_separate_prompt_and_streams(self):
        chunks = []
        def chat(model, messages, **kwargs):
            self.assertIn("tell jokes", messages[0]["content"])
            self.assertEqual(messages[-1]["content"], "Why does rain happen?")
            kwargs["on_chunk"]("Rain ")
            kwargs["on_chunk"]("forms in clouds.")
            return "Rain forms in clouds."
        with patch("assistant._stream_chat", return_value=plan("chat")), patch(
            "assistant.chat", side_effect=chat
        ), patch("assistant.get_weather") as weather:
            reply, history = answer_with_history("local", "Why does rain happen?", [], on_chunk=chunks.append)
        weather.assert_not_called()
        self.assertEqual(chunks, ["Rain ", "forms in clouds."])
        self.assertEqual(history[-1]["content"], reply)

    def test_negated_setting_is_not_saved_even_if_interpreter_requests_it(self):
        with patch("assistant._stream_chat", return_value=plan(
            "write_setting", setting_name="default_city", setting_value="London"
        )), patch("assistant.update_settings") as save:
            reply, _ = answer_with_history("local", "Don't change my default city to London.", [])
        save.assert_not_called()
        self.assertIn("didn't change", reply)

    def test_explicit_setting_change_is_saved(self):
        with patch("assistant._stream_chat", return_value=plan(
            "write_setting", setting_name="default_city", setting_value="London"
        )), patch("assistant.update_settings", return_value={"default_city": "London", "temperature_unit": "celsius"}) as save:
            reply, _ = answer_with_history("local", "Set my default city to London", [])
        save.assert_called_once_with(default_city="London")
        self.assertEqual(reply, "Default city set to London.")

    def test_clock_only_answers_requested_part(self):
        with patch("assistant._stream_chat", return_value=plan("clock", clock_part="both")):
            reply, _ = answer_with_history("local", "What time is it?", [])
        self.assertEqual(reply, "10:15 AM.")

    def test_missing_city_prompts_then_completes_pending_weather_request(self):
        self.settings.return_value = {"default_city": "", "temperature_unit": "celsius"}
        with patch("assistant._stream_chat", side_effect=[
            plan("weather", day="tomorrow"), plan("weather", location="Boston"),
        ]), patch("assistant.get_weather", return_value=WEATHER) as weather:
            question, history = answer_with_history("local", "Weather tomorrow?", [])
            _, history = answer_with_history("local", "In Boston", history)
        self.assertEqual(question, "Which city should I check?")
        weather.assert_called_once_with("Boston", "tomorrow")

    def test_date_is_resolved_from_user_words_instead_of_model_guess(self):
        with patch("assistant._stream_chat", return_value=plan("weather", location="London", day="2025-09-28")), patch(
            "assistant.get_weather", return_value=WEATHER
        ) as weather:
            answer_with_history("local", "What's the weather in London on September 28th?", [])
        weather.assert_called_once_with("London", "09-28")

    def test_explicit_year_is_retained(self):
        with patch("assistant._stream_chat", return_value=plan("weather", location="London")), patch(
            "assistant.get_weather", return_value=WEATHER
        ) as weather:
            answer_with_history("local", "What's the weather in London on September 28th, 2026?", [])
        weather.assert_called_once_with("London", "2026-09-28")

    def test_cancellation_after_lookup_discards_result(self):
        cancellation = ChatCancellation()
        chunks = []
        def lookup(*_args):
            cancellation.cancel()
            return WEATHER
        with patch("assistant._stream_chat", return_value=plan("weather")), patch(
            "assistant.get_weather", side_effect=lookup
        ):
            with self.assertRaises(ChatInterrupted):
                answer_with_history("local", "Weather in London?", [], cancellation=cancellation, on_chunk=chunks.append)
        self.assertEqual(chunks, [])

    def test_bad_interpretation_does_not_execute_tools(self):
        with patch("assistant._stream_chat", return_value={"role": "assistant", "content": "not json"}), patch(
            "assistant.get_weather"
        ) as weather:
            with self.assertRaises(OllamaError):
                answer_with_history("local", "What's the weather?", [])
        weather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
