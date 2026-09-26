import json
from pathlib import Path
import tempfile
import unittest

from intent_classifier import ROOT, decide, load_data, metrics, render_input
from capabilities import resolve_route


class IntentClassifierTests(unittest.TestCase):
    def test_dataset_is_complete_and_disjoint(self):
        data = load_data(ROOT / "intent_data.json")
        self.assertGreater(len(data["train"]), len(data["test"]))

    def test_leaked_training_example_rejected(self):
        data = json.loads((ROOT / "intent_data.json").read_text(encoding="utf-8"))
        data["test"]["chat"].append(data["train"]["chat"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_data(path)

    def test_uncertainty_does_not_silently_become_action(self):
        self.assertEqual(decide([0.4, 0.6], ["chat", "weather"], 0.8), "uncertain")
        self.assertEqual(decide([0.1, 0.9], ["chat", "weather"], 0.8), "weather")

    def test_disabled_capability_does_not_need_model_retraining(self):
        self.assertEqual(resolve_route("weather", {"weather": {"enabled": False}}), "unsupported_action")
        self.assertEqual(resolve_route("book_cab"), "uncertain")
        self.assertEqual(resolve_route("unsupported_action"), "unsupported_action")

    def test_context_changes_input_but_does_not_include_previous_answer(self):
        plain = render_input({"text": "And tomorrow?"})
        contextual = render_input({"text": "And tomorrow?", "active_task": {"intent": "weather", "reply": "secret"}})
        self.assertNotEqual(plain, contextual)
        self.assertNotIn("secret", contextual)

    def test_unknown_actions_and_rejections_reported_separately(self):
        rows = [{"expected": "unsupported_action", "predicted": "weather"},
                {"expected": "chat", "predicted": "unsupported_action"},
                {"expected": "weather", "predicted": "uncertain"}]
        result = metrics(rows)
        self.assertEqual(result["false_action_on_unsupported"], 1)
        self.assertEqual(result["false_action_on_chat"], 0)
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["accepted"], 2)

    def test_uncertain_never_becomes_a_training_label(self):
        data = load_data(ROOT / "intent_data.json")
        self.assertNotIn("uncertain", {r["label"] for r in data["train"]})
        self.assertTrue(any(r["group"] == "unseen_action" for r in data["test"]))
