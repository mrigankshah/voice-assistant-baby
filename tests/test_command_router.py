import unittest

from command_router import route_command


class CommandRouterTests(unittest.TestCase):
    def test_explicit_actions_and_non_actions(self):
        cases = {
            "What's the weather tomorrow?": "weather",
            "What time is it?": "clock",
            "Set a timer for ten minutes": "timer",
            "Set an alarm for tomorrow at 7 AM": "alarm",
            "Set my default city to Boston": "write_setting",
            "What is my default city?": "read_setting",
            "How do black holes form?": "chat",
            "What is a timer?": "chat",
            "What is weather?": "chat",
            "What's a weather forecast?": "chat",
            "What is a date?": "chat",
            "Why does rain happen?": "chat",
            "I like rainy weather. Tell me a story about a rainy day.": "chat",
            "Can you look up weather forecasts? Just tell me whether you can.": "chat",
            "Do I need a jacket?": "chat",
            "Don't set an alarm": "chat",
            "Please don't set a timer for ten minutes": "chat",
            "I don't want to change my default city": "chat",
            "Book a cab": "unsupported_action",
            "Can you book a cab?": "unsupported_action",
        }
        for utterance, expected in cases.items():
            with self.subTest(utterance=utterance):
                self.assertEqual(route_command(utterance)["intent"], expected)

    def test_topic_change_overrides_weather_followup(self):
        task = {"intent": "weather", "location": "London", "day": "today"}
        self.assertEqual(route_command("And tomorrow?", task)["intent"], "weather")
        self.assertEqual(route_command("What about Paris?", task)["intent"], "weather")
        self.assertEqual(route_command("How do black holes form?", task)["intent"], "chat")
        self.assertEqual(route_command("Set a timer for five minutes", task)["intent"], "timer")

    def test_multiple_actions_ask_for_one(self):
        result = route_command("Check the weather and set an alarm")
        self.assertEqual(result["intent"], "clarify")

    def test_pending_question_does_not_trap_unrelated_conversation(self):
        pending = {"intent": "timer", "operation": "create", "pending": True}
        self.assertEqual(route_command("ten minutes", pending)["intent"], "timer")
        self.assertEqual(route_command("How are you?", pending)["intent"], "chat")


if __name__ == "__main__":
    unittest.main()
