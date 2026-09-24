"""A small text chat client for the Ollama server running on the Pi."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from http.client import HTTPConnection, HTTPException
from socket import SHUT_RDWR, socket
from threading import Event, Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


OLLAMA_URL = "http://127.0.0.1:11434"
MAX_HISTORY_MESSAGES = 12


class OllamaError(Exception):
    """An Ollama request failed or returned an unexpected response."""


class ChatInterrupted(Exception):
    """The user started speaking during a streamed reply."""


class ChatCancellation:
    """Stop a streamed reply by shutting down its local HTTP connection."""

    def __init__(self) -> None:
        self._cancelled = Event()
        self._lock = Lock()
        self._socket: socket | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            active_socket = self._socket
        if active_socket is not None:
            self._shutdown(active_socket)

    def check(self) -> None:
        if self._cancelled.is_set():
            raise ChatInterrupted()

    def attach(self, active_socket: socket | None) -> None:
        with self._lock:
            self._socket = active_socket
        if self._cancelled.is_set():
            if active_socket is not None:
                self._shutdown(active_socket)
            self.check()

    def detach(self, active_socket: socket | None) -> None:
        with self._lock:
            if self._socket is active_socket:
                self._socket = None

    @staticmethod
    def _shutdown(active_socket: socket) -> None:
        try:
            active_socket.shutdown(SHUT_RDWR)
        except OSError:
            pass


def request_json(path: str, payload: dict | None = None, *, base_url: str = OLLAMA_URL) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )

    try:
        with urlopen(request, timeout=300 if data is not None else 10) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            detail = json.load(exc).get("error", exc.reason)
        except (ValueError, AttributeError):
            detail = exc.reason
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise OllamaError(
            "Cannot reach Ollama on this computer. Check that Ollama is running "
            "with `ollama list`, then try again."
        ) from exc
    except ValueError as exc:
        raise OllamaError("Ollama returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise OllamaError("Ollama returned an unexpected response.")
    return result


def list_models(*, base_url: str = OLLAMA_URL) -> list[str]:
    result = request_json("/api/tags", base_url=base_url)
    models = result.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama did not return a model list.")
    return sorted(
        {model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)}
    )


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    on_chunk: Callable[[str], None] | None = None,
    cancellation: ChatCancellation | None = None,
    base_url: str = OLLAMA_URL,
) -> str:
    if cancellation is not None:
        cancellation.check()
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname is None:
        raise OllamaError("Ollama chat requires a local HTTP address.")
    connection = HTTPConnection(parsed_url.hostname, parsed_url.port, timeout=300)
    path = f"{parsed_url.path.rstrip('/')}/api/chat"
    body = json.dumps({"model": model, "messages": messages, "stream": True})
    parts: list[str] = []
    finished = False
    active_socket: socket | None = None
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        active_socket = connection.sock
        if cancellation is not None:
            cancellation.attach(active_socket)
        response = connection.getresponse()
        if response.status >= 400:
            try:
                detail = json.loads(response.read()).get("error", response.reason)
            except (ValueError, AttributeError):
                detail = response.reason
            raise OllamaError(f"Ollama returned HTTP {response.status}: {detail}")
        for line in response:
            if cancellation is not None:
                cancellation.check()
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise OllamaError("Ollama returned invalid streaming JSON.") from exc
            if not isinstance(event, dict):
                raise OllamaError("Ollama returned an unexpected streaming response.")
            if event.get("error"):
                raise OllamaError(f"Ollama error: {event['error']}")
            message = event.get("message")
            if message is not None and not isinstance(message, dict):
                raise OllamaError("Ollama returned an unexpected chat message.")
            content = message.get("content") if message else None
            if content is not None and not isinstance(content, str):
                raise OllamaError("Ollama returned unexpected chat content.")
            if content:
                parts.append(content)
                if on_chunk is not None:
                    on_chunk(content)
            if event.get("done") is True:
                finished = True
                break
    except (OSError, ValueError, HTTPException) as exc:
        if cancellation is not None:
            cancellation.check()
        raise OllamaError("Cannot complete the Ollama request. Check that Ollama is running.") from exc
    finally:
        if cancellation is not None:
            cancellation.detach(active_socket)
        connection.close()

    if cancellation is not None:
        cancellation.check()
    if not finished:
        raise OllamaError("Ollama stopped streaming before the reply finished.")
    reply = "".join(parts).strip()
    if not reply:
        raise OllamaError("Ollama returned no text. This model may not support chat.")
    return reply


def answer_with_history(
    model: str,
    prompt: str,
    history: list[dict[str, str]],
    *,
    on_chunk: Callable[[str], None] | None = None,
    cancellation: ChatCancellation | None = None,
) -> tuple[str, list[dict[str, str]]]:
    user_message = {"role": "user", "content": prompt}
    reply = chat(model, history + [user_message], on_chunk=on_chunk, cancellation=cancellation)
    updated_history = (history + [user_message, {"role": "assistant", "content": reply}])[
        -MAX_HISTORY_MESSAGES:
    ]
    return reply, updated_history


def choose_model(models: list[str], *, read=input, write=print) -> str:
    if not models:
        raise OllamaError("No models are installed in Ollama. Check `ollama list` on the Pi.")

    write("Available Ollama models:")
    for number, model in enumerate(models, start=1):
        write(f"  {number}. {model}")

    while True:
        choice = read("Choose a model number: ").strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        write(f"Enter a number from 1 to {len(models)}.")


def main() -> int:
    print("Looking for models on this Pi...")
    try:
        model = choose_model(list_models())
        print(f"\nChatting with {model}. Type /model to switch, /reset to clear context, or /quit to leave.")
        history: list[dict[str, str]] = []

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return 0

            if not prompt:
                continue
            if prompt.lower() == "/quit":
                print("Goodbye.")
                return 0
            if prompt.lower() == "/reset":
                history.clear()
                print("Conversation cleared.")
                continue
            if prompt.lower() == "/model":
                model = choose_model(list_models())
                history.clear()
                print(f"Now chatting with {model}. Conversation cleared.")
                continue

            try:
                print("\nAssistant: ", end="", flush=True)
                _, history = answer_with_history(
                    model, prompt, history, on_chunk=lambda chunk: print(chunk, end="", flush=True)
                )
            except OllamaError as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                continue
            print()
    except (OllamaError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, OllamaError):
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
