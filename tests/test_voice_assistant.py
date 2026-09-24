"""Turn-taking behavior for short pauses in microphone speech."""

import unittest

from voice_assistant import ConversationWindow, UtteranceBuffer


class ConversationWindowTests(unittest.TestCase):
    def test_background_speech_cannot_wake_it(self):
        conversation = ConversationWindow(15)
        for text in ("What's for dinner?", "They said hey baby yesterday", "Hey babysitter"):
            conversation.speech_started(1)
            self.assertIsNone(conversation.accept(text, 2))
            self.assertFalse(conversation.awake)

    def test_wake_phrase_and_question_in_one_utterance(self):
        conversation = ConversationWindow(15)
        self.assertEqual(conversation.accept("Hey, BABY! Explain black holes.", 0), "Explain black holes.")
        self.assertTrue(conversation.awake)
        self.assertFalse(conversation.expire(300))  # Still waiting for the model.

    def test_wake_phrase_alone_opens_a_window(self):
        conversation = ConversationWindow(15)
        self.assertIsNone(conversation.accept("Hey Baby.", 0))
        self.assertFalse(conversation.expire(14.9))
        self.assertTrue(conversation.expire(15))
        self.assertIsNone(conversation.accept("Explain black holes", 16))

    def test_followup_started_in_time_can_finish_after_deadline(self):
        conversation = ConversationWindow(15)
        conversation.accept("Hey Baby, hello", 0)
        conversation.wait_for_followup(30)
        self.assertFalse(conversation.speech_started(44))
        self.assertFalse(conversation.expire(60))
        self.assertEqual(conversation.accept("Tell me more", 61), "Tell me more")
        self.assertFalse(conversation.expire(90))
        conversation.wait_for_followup(100)
        self.assertFalse(conversation.expire(114))
        self.assertTrue(conversation.expire(115))

    def test_late_speech_requires_wake_phrase_again(self):
        conversation = ConversationWindow(15)
        conversation.accept("Hey Baby", 0)
        self.assertTrue(conversation.speech_started(16))
        self.assertIsNone(conversation.accept("Tell me more", 18))
        self.assertEqual(conversation.accept("Hey Baby, tell me more", 20), "tell me more")

    def test_interruption_and_empty_transcript_do_not_leave_it_awake_forever(self):
        conversation = ConversationWindow(15)
        conversation.accept("Hey Baby, tell me a story", 0)
        conversation.speech_started(100)  # Interrupt a long reply without a wake phrase.
        self.assertTrue(conversation.awake)
        conversation.wait_for_followup(101)  # No words recovered from the interruption.
        self.assertTrue(conversation.expire(116))

    def test_go_to_sleep_ends_active_conversation(self):
        conversation = ConversationWindow(15)
        conversation.accept("Hey Baby", 0)
        conversation.speech_started(1)
        self.assertIsNone(conversation.accept("Go to sleep!", 2))
        self.assertFalse(conversation.awake)
        self.assertIsNone(conversation.accept("Tell me more", 3))


class UtteranceBufferTests(unittest.TestCase):
    def test_resumed_speech_extends_the_wait_and_joins_segments(self):
        buffer = UtteranceBuffer(pause_seconds=1.5)
        buffer.started()
        buffer.finished("What is", at=1.0)
        self.assertFalse(buffer.ready(2.0))

        buffer.started()
        self.assertFalse(buffer.ready(3.0))
        buffer.finished("the weather?", at=3.0)
        self.assertFalse(buffer.ready(4.0))
        self.assertTrue(buffer.ready(4.5))
        self.assertEqual(buffer.take(), "What is the weather?")

    def test_wake_phrase_can_span_transcript_segments(self):
        buffer = UtteranceBuffer(pause_seconds=1.5)
        buffer.started()
        buffer.finished("Hey", at=1)
        buffer.started()
        buffer.finished("baby, what time is it?", at=2)
        self.assertTrue(buffer.ready(3.5))
        self.assertEqual(ConversationWindow(15).accept(buffer.take(), 3.5), "what time is it?")


if __name__ == "__main__":
    unittest.main()
