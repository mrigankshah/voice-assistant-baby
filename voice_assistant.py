"""Send completed Moonshine microphone transcripts to a local Ollama model."""

from __future__ import annotations

import argparse
import sys
from math import isfinite
from queue import Empty, Queue
from threading import Thread
from time import monotonic

from assistant import (
    ChatCancellation,
    ChatInterrupted,
    OllamaError,
    answer_with_history,
    choose_model,
    list_models,
)


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
    utterance = UtteranceBuffer(args.pause_seconds)
    reply_events: Queue[tuple[str, int, object]] = Queue()
    turn_number = 0
    active_turn: int | None = None
    active_cancellation: ChatCancellation | None = None

    def run_reply(
        turn: int,
        prompt: str,
        prior_history: list[dict[str, str]],
        cancellation: ChatCancellation,
    ) -> None:
        def on_chunk(chunk: str) -> None:
            cancellation.check()
            reply_events.put(("chunk", turn, chunk))

        try:
            _, updated_history = answer_with_history(
                model, prompt, prior_history, on_chunk=on_chunk, cancellation=cancellation
            )
        except ChatInterrupted:
            return
        except OllamaError as exc:
            reply_events.put(("error", turn, str(exc)))
        else:
            reply_events.put(("done", turn, updated_history))

    print(f"Loading Moonshine. Chat model: {model}")
    with mic:
        mic.load()
        print(
            f"Listening. Pause for {args.pause_seconds:g} seconds when done speaking. "
            "Speak again to interrupt a reply. Press Ctrl+C to stop."
        )
        mic.start()
        try:
            while True:
                try:
                    kind, text, at = speech_events.get_nowait()
                except Empty:
                    pass
                else:
                    if kind == "started":
                        utterance.started()
                        if active_cancellation is not None:
                            active_cancellation.cancel()
                            active_cancellation = None
                            active_turn = None
                            print("\n[Interrupted. Listening for your new question.]")
                    else:
                        utterance.finished(text, at)
                    continue

                if active_turn is None and utterance.ready(monotonic()):
                    prompt = utterance.take()
                    print(f"\nYou: {prompt}")
                    print("\nAssistant: ", end="", flush=True)
                    turn_number += 1
                    active_turn = turn_number
                    active_cancellation = ChatCancellation()
                    Thread(
                        target=run_reply,
                        args=(active_turn, prompt, history, active_cancellation),
                        daemon=True,
                    ).start()
                    continue

                try:
                    kind, turn, payload = reply_events.get(timeout=0.05)
                except Empty:
                    continue
                if turn != active_turn:
                    continue
                if kind == "chunk":
                    print(payload, end="", flush=True)
                elif kind == "done":
                    history = payload
                    active_turn = None
                    active_cancellation = None
                    print()
                elif kind == "error":
                    active_turn = None
                    active_cancellation = None
                    print(f"\nError: {payload}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            if active_cancellation is not None:
                active_cancellation.cancel()
            mic.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
