"""Send completed Moonshine microphone transcripts to a local Ollama model."""

from __future__ import annotations

import argparse
import sys
from math import isfinite
from queue import Empty, Queue
from time import monotonic

from assistant import OllamaError, answer_with_history, choose_model, list_models


class UtteranceBuffer:
    """Join Moonshine lines until a complete quiet period has passed."""

    def __init__(self, pause_seconds: float) -> None:
        self.pause_seconds = pause_seconds
        self.parts: list[str] = []
        self.speaking = False
        self.deadline: float | None = None

    def started(self) -> None:
        self.speaking = True

    def finished(self, text: str, at: float) -> None:
        self.speaking = False
        if text:
            self.parts.append(text)
        if self.parts:
            self.deadline = at + self.pause_seconds

    def ready(self, now: float) -> bool:
        return bool(self.parts) and not self.speaking and self.deadline is not None and now >= self.deadline

    def take(self) -> str:
        prompt = " ".join(self.parts)
        self.parts.clear()
        self.deadline = None
        return prompt


def next_utterance(events: Queue[tuple[str, str, float]], pause_seconds: float) -> str:
    buffer = UtteranceBuffer(pause_seconds)
    while True:
        try:
            kind, text, at = events.get(timeout=0.1)
        except Empty:
            if buffer.ready(monotonic()):
                return buffer.take()
            continue

        if kind == "started":
            buffer.started()
        else:
            buffer.finished(text, at)


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to a local Ollama model through Moonshine.")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.5,
        help="Extra quiet time after a finished speech segment before replying (default: 1.5).",
    )
    args = parser.parse_args()
    if not isfinite(args.pause_seconds) or args.pause_seconds < 0:
        parser.error("--pause-seconds must be a finite number that is zero or greater")

    try:
        model = choose_model(list_models())
    except (OllamaError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, OllamaError):
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        from moonshine_voice import MicTranscriber, ModelArch, TranscriptEventListener
    except ImportError:
        print(
            "Moonshine is not available to this Python interpreter. Activate the same "
            "Python environment you use for `moonshine-voice mic`, then try again.",
            file=sys.stderr,
        )
        return 1

    speech_events: Queue[tuple[str, str, float]] = Queue()

    class SpeechActivity(TranscriptEventListener):
        def on_line_started(self, event) -> None:
            speech_events.put(("started", "", monotonic()))

    def on_line(line) -> None:
        text = line.text.strip()
        speech_events.put(("finished", text, monotonic()))

    mic = (
        MicTranscriber()
        .language("en")
        .model_arch(ModelArch.SMALL_STREAMING)
        .on_line(on_line)
    )
    mic.add_listener(SpeechActivity())
    history: list[dict[str, str]] = []
    print(f"Loading Moonshine. Chat model: {model}")
    with mic:
        mic.load()
        print(
            f"Listening. Pause for {args.pause_seconds:g} seconds when done speaking. "
            "Press Ctrl+C to stop."
        )
        mic.start()
        try:
            while True:
                prompt = next_utterance(speech_events, args.pause_seconds)
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
