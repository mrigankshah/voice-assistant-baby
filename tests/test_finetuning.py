import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from assistant import ChatCancellation, ChatInterrupted, answer_with_history
from finetuning.contract import tools_for, validate_call
from finetuning.data import prepare
from finetuning.evaluate import score
from finetuning.runtime import execute
from finetuning.seed_data import call, reply
from finetuning.train import render_example
from local_schedule import Schedule


class FineTuningTests(unittest.TestCase):
    def test_rejects_bad_arguments_and_disabled_tools(self):
        for name, args in [("create_timer", {"seconds": True}), ("create_timer", {"seconds": -1}),
                           ("create_alarm", {"hour": 25, "minute": 0}), ("cancel_alarm", {}),
                           ("set_weather_preferences", {}), ("cancel_timer", {"id": 1, "all": True}),
                           ("send_message", {"text": "hi"})]:
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                validate_call(call(name, **args)["tool_calls"][0])
        with self.assertRaises(ValueError):
            validate_call(call("list_timers")["tool_calls"][0], [])

    def test_baseline_scores_arguments_not_just_tool_names(self):
        expected = call("cancel_timer", id=1)
        self.assertFalse(score(expected, call("cancel_timer", id=2), tools_for())["passed"])
        self.assertTrue(score(expected, expected, tools_for())["passed"])
        self.assertTrue(score(reply("Hello"), call("list_timers"), tools_for())["unwanted_tool"])

    def test_weather_omission_preserves_default_lookup(self):
        with patch("assistant.get_weather", return_value={"location": "Boston"}) as weather:
            result = execute(call("get_weather", day="tomorrow")["tool_calls"][0])
        weather.assert_called_once_with(day="tomorrow")
        self.assertIn("Boston", result)

    def test_structured_arguments_execute_without_regex(self):
        with tempfile.TemporaryDirectory() as folder:
            schedule = Schedule(Path(folder) / "test.sqlite3")
            with patch("finetuning.runtime.Schedule", return_value=schedule):
                result = execute(call("create_timer", seconds=480, label="tea")["tool_calls"][0])
                self.assertIn("480", result)
                job = schedule.list_jobs("timer")[0]
                result = execute(call("cancel_timer", id=job["id"])["tool_calls"][0])
                self.assertIn("Cancelled", result)
                self.assertEqual(schedule.list_jobs("timer"), [])

    def test_direct_chat_uses_one_model_call_and_streams(self):
        def prediction(*args, **kwargs):
            kwargs["on_chunk"]("Hello")
            return reply("Hello")
        chunks = []
        with patch("finetuning.runtime.predict", side_effect=prediction) as predict:
            result, history = answer_with_history("model", "hi", [], backend="tools", on_chunk=chunks.append)
        self.assertEqual(result, "Hello")
        self.assertEqual(chunks, ["Hello"])
        predict.assert_called_once()

    def test_tool_result_and_id_are_in_followup_history(self):
        with patch("finetuning.runtime.predict", return_value=call("create_timer", seconds=60)), \
             patch("finetuning.runtime.execute", return_value="Timer #9 set for 60 seconds."):
            result, history = answer_with_history("m", "one minute", [], backend="tools")
        self.assertEqual(history[-2]["role"], "tool")
        self.assertIn("#9", history[-2]["content"])

    def test_multiple_calls_never_partially_execute(self):
        result = call("cancel_timer", id=1)
        result["tool_calls"] += call("cancel_timer", id=2)["tool_calls"]
        with patch("finetuning.runtime.predict", return_value=result), patch("finetuning.runtime.execute") as execute:
            answer_with_history("m", "cancel both", [], backend="tools")
        execute.assert_not_called()

    def test_malformed_call_is_not_saved_as_tool_history(self):
        with patch("finetuning.runtime.predict", return_value={"tool_calls": [{"function": "invalid"}]}), \
             patch("finetuning.runtime.execute") as execute:
            text, history = answer_with_history("m", "cancel", [], backend="tools")
        self.assertIn("couldn't", text)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        execute.assert_not_called()

    def test_cancelled_turn_cannot_execute(self):
        cancel = ChatCancellation()
        cancel.cancel()
        with patch("finetuning.runtime.predict", return_value=call("cancel_timer", id=1)), \
             patch("finetuning.runtime.execute") as execute, self.assertRaises(ChatInterrupted):
            answer_with_history("m", "cancel one", [], backend="tools", cancellation=cancel)
        execute.assert_not_called()

    def test_prepared_data_has_tools_and_separate_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = prepare(Path(folder))
            self.assertEqual(manifest["splits"]["test"]["count"], 35)
            rows = [json.loads(line) for line in (Path(folder) / "train.jsonl").read_text().splitlines()]
            self.assertTrue(any(not row["tools"] for row in rows))
            self.assertTrue(any(len(row["messages"]) > 2 for row in rows))

    def test_training_masks_prompt_and_rejects_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "PROMPT:" if kwargs["add_generation_prompt"] else "PROMPT:answer"
            def __call__(self, text, **kwargs):
                return {"input_ids": list(text.encode())}
        row = {"id": "one", "messages": [], "target": reply("answer"), "tools": []}
        value = render_example(Tokenizer(), row, 100)
        self.assertEqual(value["labels"][:7], [-100] * 7)
        self.assertEqual(value["labels"][7:], list(b"answer"))
        with self.assertRaises(ValueError):
            render_example(Tokenizer(), row, 5)


if __name__ == "__main__":
    unittest.main()
