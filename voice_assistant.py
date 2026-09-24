"""Send completed Moonshine microphone transcripts to a local Ollama model."""

from __future__ import annotations

import sys
from queue import Empty, Queue

from assistant import OllamaError, answer_with_history, choose_model, list_models


def main() -> int:
    try:
        model = choose_model(list_models())
    except (OllamaError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, OllamaError):
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        from moonshine_voice import MicTranscriber, ModelArch
    except ImportError:
        print(
            "Moonshine is not available to this Python interpreter. Activate the same "
            "Python environment you use for `moonshine-voice mic`, then try again.",
            file=sys.stderr,
        )
        return 1

    completed_lines: Queue[str] = Queue()

    def on_line(line) -> None:
        text = line.text.strip()
        if text:
            completed_lines.put(text)

    mic = (
        MicTranscriber()
        .language("en")
        .model_arch(ModelArch.SMALL_STREAMING)
        .on_line(on_line)
    )
    history: list[dict[str, str]] = []
    print(f"Loading Moonshine. Chat model: {model}")
    with mic:
        mic.load()
        print("Listening. Speak a question, then pause for the reply. Press Ctrl+C to stop.")
        mic.start()
        try:
            while True:
                try:
                    prompt = completed_lines.get(timeout=0.1)
                except Empty:
                    continue

                print(f"\nYou: {prompt}")
                try:
                    print("\nAssistant: ", end="", flush=True)
                    _, history = answer_with_history(
                        model, prompt, history, on_chunk=lambda chunk: print(chunk, end="", flush=True)
                    )
                except OllamaError as exc:
                    print(f"\nError: {exc}", file=sys.stderr)
                    continue
                print()
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            mic.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
