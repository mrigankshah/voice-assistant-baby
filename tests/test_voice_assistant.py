"""Turn-taking behavior for short pauses in microphone speech."""

import unittest

from voice_assistant import UtteranceBuffer


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


if __name__ == "__main__":
    unittest.main()
