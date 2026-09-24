"""Contract tests for the local Ollama requests and model picker."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from assistant import answer_with_history, chat, choose_model, list_models


class FakeOllamaHandler(BaseHTTPRequestHandler):
    requests = []

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
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"message": {"content": "Hello from Ollama"}}).encode())

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
        self.assertEqual(chat("alpha:latest", messages, base_url=self.base_url), "Hello from Ollama")
        self.assertEqual(
            FakeOllamaHandler.requests[-1],
            ("/api/chat", {"model": "alpha:latest", "messages": messages, "stream": False}),
        )

    def test_picker_retries_invalid_choice(self):
        answers = iter(["not a number", "3", "2"])
        output = []
        selected = choose_model(["alpha:latest", "zeta:latest"], read=lambda _: next(answers), write=output.append)
        self.assertEqual(selected, "zeta:latest")
        self.assertEqual(output.count("Enter a number from 1 to 2."), 2)

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
