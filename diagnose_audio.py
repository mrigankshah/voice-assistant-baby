"""Run isolated, measured Moonshine experiments on a Raspberry Pi.

The controller uses only the standard library. Workers use the same Python
interpreter and installed sounddevice/Moonshine as the caller. No speech is saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback


# Every optional experiment starts from no-audio; change one factor at a time.
SCENARIOS = {
    "baseline": {"return_audio_data": "true"},
    "no-audio": {"return_audio_data": "false"},
    "capture": {},
    "vad": {"return_audio_data": "false", "skip_transcription": "true"},
    "slower": {"return_audio_data": "false", "transcription_interval": "1.0"},
    "final-only": {"return_audio_data": "false", "decode_incomplete_lines": "false"},
    "host-block": {"return_audio_data": "false"},
    "larger-block": {"return_audio_data": "false"},
    "buffered": {"return_audio_data": "false"},
}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def process_sample(pid: int) -> dict:
    """Read current memory and cumulative CPU independently of the audio worker."""
    result = {}
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmSwap"):
                result[{"VmRSS": "rss_mb", "VmSwap": "swap_mb"}[key]] = int(value.split()[0]) / 1024
            elif key == "Threads":
                result["threads"] = int(value.strip())
        # Process names can contain spaces and parentheses; split after the last ).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        pass
    try:
        result["temperature_c"] = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        pass
    return result


def summarize(rows: list[dict]) -> dict:
    live = [row for row in rows if row.get("state") == "running"]
    result = {"samples": len(live)}
    if not live:
        return result
    result["observed_seconds"] = live[-1]["elapsed_s"] - live[0]["elapsed_s"]
    for key in ("rss_mb", "cpu_percent", "backlog_s", "heartbeat_age_s", "temperature_c"):
        values = [r[key] for r in live if isinstance(r.get(key), (int, float))]
        if values:
            result[key + "_max"] = max(values)
            result[key + "_median"] = statistics.median(values)
            # Skip initial model warmup when comparing resident memory.
            steady = [r[key] for r in live if r["elapsed_s"] >= 60 and isinstance(r.get(key), (int, float))]
            if steady:
                window = min(60, max(1, len(steady) // 4))
                result[key + "_late_minus_early"] = statistics.median(steady[-window:]) - statistics.median(steady[:window])
    for key in ("overflows", "processing_calls", "completed_lines", "parse_max_ms", "process_max_ms", "callback_gap_max_ms"):
        values = [r[key] for r in live if isinstance(r.get(key), (int, float))]
        if values:
            result[key] = max(values)
    result["last_backlog_s"] = live[-1].get("backlog_s")
    return result


def worker(args: argparse.Namespace) -> int:
    import sounddevice as sd

    folder = Path(args.run_dir)
    state_path = folder / "state.json"
    stats = {
        "state": "loading", "overflows": 0, "captured_s": 0.0,
        "processed_s": 0.0, "processing_calls": 0, "completed_lines": 0,
        "parse_max_ms": 0.0, "process_max_ms": 0.0, "callback_gap_max_ms": 0.0,
    }
    versions = {}
    for package in ("moonshine-voice", "sounddevice", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    device = args.device
    if device is not None and device.isdecimal():
        device = int(device)
    info = sd.query_devices(device, "input")
    rate = args.sample_rate or int(info["default_samplerate"])
    block = {"host-block": 0, "larger-block": 2048}.get(args.scenario, 1024)
    if args.scenario == "buffered":
        sd.default.latency = args.latency
    metadata = {
        "scenario": args.scenario, "python": sys.executable, "platform": platform.platform(),
        "versions": versions, "portaudio": sd.get_portaudio_version(),
        "device": dict(info), "requested_sample_rate": rate, "blocksize": block,
        "options": SCENARIOS[args.scenario], "model": args.model,
        "input_latency_setting": sd.default.latency[0],
    }
    atomic_json(folder / "environment.json", metadata)
    stop = threading.Event()
    snapshot_lock = threading.Lock()

    def snapshot() -> None:
        data = dict(stats)
        data["heartbeat_at"] = time.time()
        # Includes audio currently being processed, not just audio waiting in a queue.
        data["backlog_s"] = max(0.0, data["captured_s"] - data["processed_s"])
        with snapshot_lock:
            atomic_json(state_path, data)

    def heartbeat() -> None:
        while not stop.wait(2):
            snapshot()

    snapshot()
    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    previous_callback = None

    def count_capture(frames: int, sample_rate: float, status) -> None:
        nonlocal previous_callback
        now = time.monotonic()
        if previous_callback is not None:
            stats["callback_gap_max_ms"] = max(stats["callback_gap_max_ms"], (now - previous_callback) * 1000)
        previous_callback = now
        stats["overflows"] += int(status.input_overflow)
        stats["captured_s"] += frames / sample_rate

    try:
        if args.scenario == "capture":
            def capture_callback(data, frames, timing, status):
                count_capture(frames, rate, status)
                stats["processed_s"] += frames / rate

            with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32",
                                blocksize=block, callback=capture_callback) as stream:
                stats.update(state="running", actual_rate=stream.samplerate, actual_latency_s=stream.latency)
                snapshot()
                while True:
                    time.sleep(1)
        else:
            from moonshine_voice import MicTranscriber, ModelArch

            # Refuse to produce misleading capture numbers if the installed API differs.
            if not hasattr(MicTranscriber, "_open_input_stream"):
                raise RuntimeError("Installed MicTranscriber lacks _open_input_stream; send environment.json and worker.log so this probe can be adapted.")
            source = inspect.getsource(MicTranscriber)
            metadata["mic_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            metadata["mic_has_worker_queue"] = "_audio_queue" in source
            atomic_json(folder / "environment.json", metadata)

            class MeasuredMic(MicTranscriber):
                def _open_input_stream(self, samplerate, callback):
                    def measured(data, frames, timing, status):
                        if self._should_listen and not self._muted:
                            count_capture(frames, samplerate, status)
                        callback(data, frames, timing, status)
                    return super()._open_input_stream(samplerate, measured)

            def line_finished(line):
                stats["completed_lines"] += 1
                stats["last_line_duration_s"] = line.duration
                stats["last_decode_ms"] = line.last_transcription_latency_ms

            arch = ModelArch.TINY_STREAMING if args.model == "tiny" else ModelArch.SMALL_STREAMING
            mic = (MeasuredMic().language("en").model_arch(arch).device(device)
                   .samplerate(rate).blocksize(block).options(SCENARIOS[args.scenario])
                   .update_interval(1.0 if args.scenario == "slower" else 0.5).on_line(line_finished))
            with mic:
                mic.load()
                original_parse = mic.transcriber._parse_transcript

                def measured_parse(*values, **kwargs):
                    started = time.monotonic()
                    transcript = original_parse(*values, **kwargs)
                    elapsed = (time.monotonic() - started) * 1000
                    stats["parse_last_ms"] = elapsed
                    stats["parse_max_ms"] = max(stats["parse_max_ms"], elapsed)
                    stats["returned_lines"] = len(transcript.lines)
                    stats["returned_audio_s"] = sum(len(line.audio_data) if line.audio_data is not None else 0 for line in transcript.lines) / 16000
                    return transcript

                mic.transcriber._parse_transcript = measured_parse
                original_add = mic.mic_stream.add_audio

                def measured_add(audio, sample_rate=16000):
                    started = time.monotonic()
                    result = original_add(audio, sample_rate)
                    elapsed = (time.monotonic() - started) * 1000
                    stats["process_last_ms"] = elapsed
                    stats["process_max_ms"] = max(stats["process_max_ms"], elapsed)
                    stats["processing_calls"] += 1
                    stats["processed_s"] += len(audio) / sample_rate
                    return result

                mic.mic_stream.add_audio = measured_add
                mic.start()
                stats.update(state="running", actual_rate=mic._sd_stream.samplerate,
                             actual_latency_s=mic._sd_stream.latency)
                snapshot()
                while True:
                    time.sleep(1)
    except KeyboardInterrupt:
        stats["state"] = "stopped"
        return 0
    finally:
        stop.set()
        reporter.join(timeout=3)
        snapshot()


FIELDS = ["elapsed_s", "state", "rss_mb", "swap_mb", "cpu_percent", "threads", "temperature_c",
          "heartbeat_age_s", "overflows", "captured_s", "processed_s", "backlog_s", "processing_calls",
          "parse_last_ms", "parse_max_ms", "process_last_ms", "process_max_ms", "returned_lines",
          "returned_audio_s", "completed_lines", "last_line_duration_s", "last_decode_ms",
          "callback_gap_max_ms", "actual_rate", "actual_latency_s"]


def stop_worker(process: subprocess.Popen) -> bool:
    """Return True if graceful shutdown failed and forced termination was needed."""
    if process.poll() is not None:
        return False
    if os.name != "nt":
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
            return False
        except subprocess.TimeoutExpired:
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    return True


def run_phase(args: argparse.Namespace, name: str, folder: Path, seconds: float) -> dict:
    folder.mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--scenario", name,
               "--run-dir", str(folder), "--model", args.model,
               "--sample-rate", str(args.sample_rate), "--latency", str(args.latency)]
    if args.device is not None:
        command.extend(["--device", args.device])
    rows = []
    outcome = "completed"
    forced = False
    with (folder / "worker.log").open("w", encoding="utf-8") as log, (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != "nt"))
        launched = time.monotonic()
        running_since = None
        previous_cpu = None
        previous_time = None
        last_print = -30.0
        try:
            while True:
                now = time.monotonic()
                state = read_json(folder / "state.json")
                if state.get("state") == "running" and running_since is None:
                    running_since = now
                    print(f"  Listening for {seconds / 60:g} minutes. Use a similar mix of speech and quiet in each phase.", flush=True)
                elapsed = now - running_since if running_since is not None else 0.0
                resource = process_sample(process.pid)
                cpu = resource.pop("cpu_seconds", None)
                if cpu is not None and previous_cpu is not None:
                    resource["cpu_percent"] = max(0.0, 100 * (cpu - previous_cpu) / (now - previous_time))
                previous_cpu, previous_time = cpu, now
                row = {**state, **resource, "elapsed_s": round(elapsed, 2)}
                if "heartbeat_at" in state:
                    row["heartbeat_age_s"] = max(0.0, time.time() - state["heartbeat_at"])
                rows.append(row)
                writer.writerow(row)
                output.flush()
                if elapsed - last_print >= 30 and running_since is not None:
                    def show(key):
                        value = row.get(key)
                        return f"{value:.1f}" if isinstance(value, (int, float)) else "?"
                    print(f"  {elapsed / 60:4.1f} min | CPU {show('cpu_percent')}% | RAM {show('rss_mb')} MB | overflows {row.get('overflows', '?')} | backlog {show('backlog_s')} s | parse {show('parse_last_ms')} ms", flush=True)
                    last_print = elapsed
                if process.poll() is not None:
                    outcome = f"worker exited early (code {process.returncode}); see worker.log"
                    break
                if running_since is not None and elapsed >= seconds:
                    break
                if running_since is None and now - launched > 300:
                    outcome = "startup exceeded 5 minutes; see worker.log"
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            outcome = "interrupted"
        finally:
            forced = stop_worker(process)
    result = {"scenario": name, "outcome": outcome, "forced_shutdown": forced,
              "requested_seconds": seconds, "exit_code": process.returncode, **summarize(rows)}
    atomic_json(folder / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["compare", *SCENARIOS], default="compare")
    parser.add_argument("--minutes", type=float, default=30, help="Total listening time; compare splits it equally (default: 30).")
    parser.add_argument("--model", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--device", help="sounddevice input index or name; defaults to your usual microphone")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Requested capture rate; 0 uses the device's default rate.")
    parser.add_argument("--latency", type=float, default=0.2, help="Requested seconds for the buffered experiment only.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("--minutes must be finite and positive")
    if args.sample_rate < 0 or not math.isfinite(args.latency) or args.latency <= 0:
        parser.error("sample rate must be nonnegative and latency finite and positive")
    if args.worker:
        if args.scenario == "compare" or not args.run_dir:
            parser.error("worker requires a single scenario and a run directory")
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 1
    folder = Path("diagnostics") / (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    folder = folder.resolve()
    folder.mkdir(parents=True)
    phases = ["baseline", "no-audio"] if args.scenario == "compare" else [args.scenario]
    print(f"Reports: {folder}\nStop other microphone/STT programs before running. No audio or transcript text is recorded.", flush=True)
    print("The timer starts after model loading. Ctrl+C saves partial results.", flush=True)
    results = []
    for name in phases:
        print(f"\nStarting {name} ({args.model})...", flush=True)
        result = run_phase(args, name, folder / name, args.minutes * 60 / len(phases))
        results.append(result)
        atomic_json(folder / "summary.json", {"phases": results})
        if result["outcome"] != "completed":
            break
    print(f"\nFinished. Send back {folder / 'summary.json'} and the phase metrics.csv files.", flush=True)
    print("A comparison is inconclusive if the baseline did not reproduce the slowdown. Longer individual runs are supported.")
    return 0 if all(r["outcome"] == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
