"""A small text chat client for the Ollama server running on the Pi."""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://127.0.0.1:11434"
MAX_HISTORY_MESSAGES = 12


class OllamaError(Exception):
    """An Ollama request failed or returned an unexpected response."""


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


def chat(model: str, messages: list[dict[str, str]], *, base_url: str = OLLAMA_URL) -> str:
    result = request_json(
        "/api/chat",
        {"model": model, "messages": messages, "stream": False},
        base_url=base_url,
    )
    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("Ollama returned no text. This model may not support chat.")
    return content.strip()


def answer_with_history(
    model: str, prompt: str, history: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    user_message = {"role": "user", "content": prompt}
    reply = chat(model, history + [user_message])
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
                reply, history = answer_with_history(model, prompt, history)
            except OllamaError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                continue

            print(f"\nAssistant: {reply}")
    except (OllamaError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, OllamaError):
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
