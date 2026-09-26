from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from assistant import answer_with_history
from local_schedule import Schedule, next_alarm
from schedule_commands import timer_seconds
from schedule_commands import handle_alarm


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = [datetime(2026, 9, 26, 10, 0, tzinfo=ZoneInfo("UTC")).timestamp()]
        self.path = Path(self.temp.name) / "schedule.sqlite3"
        self.schedule = Schedule(self.path, clock=lambda: self.now[0], zone=ZoneInfo("UTC"))

    def test_timer_fires_once_and_survives_reopen(self):
        timer_id = self.schedule.create_timer(90, "pasta")
        self.assertEqual(len(Schedule(self.path, clock=lambda: self.now[0], zone=ZoneInfo("UTC")).list_jobs("timer")), 1)
        self.now[0] += 89
        self.assertEqual(self.schedule.fire_due(), [])
        self.now[0] += 1
        self.assertIn(f"Pasta #{timer_id} finished.", self.schedule.fire_due())
        self.assertEqual(self.schedule.fire_due(), [])
        self.assertEqual(len(self.schedule.take_notifications()), 1)
        self.assertEqual(self.schedule.take_notifications(), [])

    def test_duration_words_do_not_truncate_compound_numbers(self):
        self.assertEqual(timer_seconds("set a timer for twenty five minutes"), 1500)
        self.assertEqual(timer_seconds("set a timer for two hours and thirty minutes"), 9000)

    def test_alarm_next_occurrence_and_weekday_rollover(self):
        saturday = datetime(2026, 9, 26, 10, 0, tzinfo=ZoneInfo("UTC"))
        self.assertEqual(next_alarm(7, 0, "weekdays", saturday).weekday(), 0)
        self.assertEqual(next_alarm(7, 0, "once", saturday, tomorrow=True).date().day, 27)
        alarm_id, due = self.schedule.create_alarm(7, 0, recurrence="daily")
        self.now[0] = due.timestamp()
        self.assertIn(f"Alarm #{alarm_id} is ringing.", self.schedule.fire_due())
        self.assertEqual(len(self.schedule.list_jobs("alarm")), 1)
        self.assertGreater(self.schedule.list_jobs("alarm")[0]["due_at"], self.now[0])

    def test_alarm_never_silently_ignores_a_named_day(self):
        reply, _ = handle_alarm({"operation": "create"}, "Set an alarm for Friday at 7 AM", self.schedule)
        self.assertIn("Please say one of those", reply)
        self.assertEqual(self.schedule.list_jobs("alarm"), [])

    def test_ambiguous_cancel_never_guesses(self):
        self.schedule.create_timer(60, "tea")
        self.schedule.create_timer(120, "pasta")
        count, job = self.schedule.cancel("timer")
        self.assertEqual(count, 2)
        self.assertIsNone(job)
        self.assertEqual(len(self.schedule.list_jobs("timer")), 2)
        _, cancelled = self.schedule.cancel("timer", "tea")
        self.assertEqual(cancelled["label"], "tea")
        self.assertEqual(self.schedule.cancel_all("timer"), 1)
        self.assertEqual(self.schedule.list_jobs("timer"), [])

    def test_voice_workflows_do_not_call_ollama(self):
        with patch("structured_assistant.Schedule", return_value=self.schedule), patch("assistant._stream_chat") as router, patch("assistant.chat") as chat:
            reply, history = answer_with_history("local", "Set a pasta timer for five minutes", [])
            self.assertIn("Pasta timer", reply)
            reply, history = answer_with_history("local", "List my timers", history)
            self.assertIn("300 seconds", reply)
            reply, history = answer_with_history("local", "Set an alarm for tomorrow at 7 AM", history)
            self.assertIn("Sunday, September 27 at 07:00 AM", reply)
            reply, history = answer_with_history("local", "List my alarms", history)
            self.assertIn("Sep 27", reply)
            router.assert_not_called()
            chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
