"""Contract tests for the local Ollama requests and model picker."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from assistant import ChatCancellation, ChatInterrupted, answer_with_history, chat, choose_model, list_models


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
        from unittest.mock import patch

        with patch("assistant.chat", return_value="Hello from Ollama") as fake_chat:
            _, history = answer_with_history("alpha:latest", "Hello", [])
            answer_with_history("alpha:latest", "Follow up", history)

        self.assertEqual(
            fake_chat.call_args.args[1],
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from Ollama"},
                {"role": "user", "content": "Follow up"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
