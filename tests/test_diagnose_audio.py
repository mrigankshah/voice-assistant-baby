"""Validate diagnostic accounting and the controller without Pi hardware."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import diagnose_audio as diagnostic


class MeasurementTests(unittest.TestCase):
    def test_proc_cpu_and_memory_with_spaces_and_parentheses_in_name(self):
        fields = ["0"] * 30
        fields[11], fields[12] = "250", "50"
        def read(path, *args, **kwargs):
            if str(path).endswith("status"):
                return "VmRSS:\t2048 kB\nVmSwap:\t512 kB\nThreads:\t4\n"
            if str(path).endswith("stat"):
                return "123 (worker (audio)) " + " ".join(fields)
            return "57000"
        with patch.object(Path, "read_text", read), patch.object(os, "sysconf", return_value=100, create=True):
            result = diagnostic.process_sample(123)
        self.assertEqual(result["cpu_seconds"], 3)
        self.assertEqual(result["rss_mb"], 2)
        self.assertEqual(result["swap_mb"], 0.5)
        self.assertEqual(result["threads"], 4)
        self.assertEqual(result["temperature_c"], 57)

    def test_summary_excludes_loading_and_missing_measurements(self):
        rows = [{"state": "loading", "rss_mb": 9999, "overflows": 100}]
        rows += [{"state": "running", "elapsed_s": t, "rss_mb": 100 + t,
                  "overflows": t // 30, "backlog_s": t / 100} for t in (0, 60, 90, 120)]
        result = diagnostic.summarize(rows)
        self.assertEqual(result["rss_mb_max"], 220)
        self.assertEqual(result["rss_mb_late_minus_early"], 60)
        self.assertEqual(result["overflows"], 4)
        self.assertEqual(result["observed_seconds"], 120)
        self.assertNotIn("cpu_percent_max", result)

    def test_early_successful_exit_is_still_failed_experiment(self):
        args = argparse.Namespace(model="tiny", sample_rate=16000, latency=0.2, device=None)
        class Exited:
            pid = 123
            returncode = 0
            def poll(self):
                return 0
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(subprocess, "Popen", return_value=Exited()), patch.object(diagnostic, "process_sample", return_value={}):
                result = diagnostic.run_phase(args, "capture", Path(directory) / "capture", 30)
        self.assertIn("exited early", result["outcome"])
        self.assertEqual(result["samples"], 0)


FAKE_SOUNDDEVICE = '''
import threading
import time
from types import SimpleNamespace
default = SimpleNamespace(latency=(0.1, 0.1))
def query_devices(device, kind):
    return {"name": "Simulated input", "default_samplerate": 16000}
def get_portaudio_version():
    return (1, "simulated")
class InputStream:
    def __init__(self, samplerate, callback, **kwargs):
        self.samplerate = samplerate
        self.latency = 0.1
        self.callback = callback
        self.stop = threading.Event()
    def __enter__(self):
        def pump():
            while not self.stop.wait(0.05):
                self.callback([0.0] * 800, 800, None, SimpleNamespace(input_overflow=True))
        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
'''


FAKE_MOONSHINE = '''
from types import SimpleNamespace
import sounddevice as sd
class ModelArch:
    TINY_STREAMING = 2
    SMALL_STREAMING = 4
class MicTranscriber:
    def __init__(self):
        self._should_listen = False
        self._muted = False
        self._sd_stream = None
        self.config = {}
    def language(self, value): return self
    def model_arch(self, value): return self
    def device(self, value): return self
    def samplerate(self, value): self.rate = value; return self
    def blocksize(self, value): return self
    def update_interval(self, value): return self
    def options(self, value): self.config = value; return self
    def on_line(self, value): self.listener = value; return self
    def load(self):
        self.transcriber = SimpleNamespace(_parse_transcript=lambda transcript: transcript)
        def add(audio, rate):
            line = SimpleNamespace(duration=0.05, last_transcription_latency_ms=2,
                audio_data=audio if self.config.get("return_audio_data") == "true" else None)
            self.transcriber._parse_transcript(SimpleNamespace(lines=[line]))
            self.listener(line)
        self.mic_stream = SimpleNamespace(add_audio=add)
    def _open_input_stream(self, samplerate, callback):
        return sd.InputStream(samplerate=samplerate, callback=callback)
    def start(self):
        def callback(data, frames, timing, status):
            self.mic_stream.add_audio(data, self.rate)
        self._should_listen = True
        self._sd_stream = self._open_input_stream(self.rate, callback)
        self._sd_stream.__enter__()
    def __enter__(self): return self
    def __exit__(self, *args):
        if self._sd_stream is not None:
            self._sd_stream.__exit__(*args)
'''


class ControllerIntegrationTests(unittest.TestCase):
    def test_short_capture_run_reports_overflows_and_saves_results(self):
        script = Path(diagnostic.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "sounddevice.py").write_text(FAKE_SOUNDDEVICE, encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(folder))
            completed = subprocess.run(
                [sys.executable, str(script), "--scenario", "capture", "--minutes", "0.05"],
                cwd=folder, env=env, capture_output=True, text=True, timeout=25,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            run = next((folder / "diagnostics").iterdir())
            summary = json.loads((run / "summary.json").read_text())["phases"][0]
            self.assertEqual(summary["outcome"], "completed")
            self.assertGreater(summary["overflows"], 0)
            self.assertEqual(summary["last_backlog_s"], 0)
            self.assertGreaterEqual(summary["observed_seconds"], 3)
            self.assertTrue((run / "capture" / "metrics.csv").is_file())
            environment = json.loads((run / "capture" / "environment.json").read_text())
            self.assertEqual(environment["device"]["name"], "Simulated input")

    def test_moonshine_probe_records_parsing_without_saving_speech(self):
        script = Path(diagnostic.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "sounddevice.py").write_text(FAKE_SOUNDDEVICE, encoding="utf-8")
            (folder / "moonshine_voice.py").write_text(FAKE_MOONSHINE, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), "--scenario", "no-audio", "--minutes", "0.05"],
                cwd=folder, env=dict(os.environ, PYTHONPATH=str(folder)),
                capture_output=True, text=True, timeout=25,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            run = next((folder / "diagnostics").iterdir())
            summary = json.loads((run / "summary.json").read_text())["phases"][0]
            self.assertGreater(summary["processing_calls"], 0)
            self.assertGreater(summary["completed_lines"], 0)
            self.assertGreaterEqual(summary["parse_max_ms"], 0)
            state = json.loads((run / "no-audio" / "state.json").read_text())
            self.assertEqual(state["returned_audio_s"], 0)
            self.assertEqual(state["returned_lines"], 1)
            self.assertEqual(state["last_decode_ms"], 2)


if __name__ == "__main__":
    unittest.main()
