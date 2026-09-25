# Voice Assistant Baby

The Pi can chat by typed text or listen through Moonshine and print a reply from Ollama. Both modes list the models installed in Ollama, let you choose one, keep a short conversation history, and show Ollama's reply as it is generated. The assistant can also look up live weather. Spoken replies will come later.

## Run on the Pi

1. Confirm Ollama is installed, running, and has at least one model:

   ```bash
   ollama list
   ```

2. After committing and pushing these files from Windows, pull this repository on the Pi and run from its directory:

   ```bash
   git pull --ff-only origin main
   python3 assistant.py
   ```

3. Choose a model number and type a message. Use `/model` to switch models, `/reset` to clear the conversation, or `/quit` to exit.

The script uses Ollama's local API at `http://127.0.0.1:11434`. That HTTP connection stays on the Pi; it does not send your speech or prompts to the internet. You do not need to leave an interactive `ollama run <model-name>` session open. If `ollama list` cannot connect, start the Ollama service using your Pi's existing setup before running the script.

## Ask about the weather

Choose your installed LFM2.5-1.2B-Instruct model and try a typed question first:

```text
What's the weather in Boston today?
Will it rain in Boston tomorrow?
```

Then try the same question by voice with "Hey Baby". The Pi asks Ollama whether to call `get_weather`; when it does, the Pi resolves the city and gets current conditions or a forecast from [Open-Meteo](https://open-meteo.com/en/docs). Location and forecast requests go to Open-Meteo over the internet. Ollama and Moonshine remain local. No weather API key or extra Python package is needed for personal use.

You can set a default location on the Pi before starting either program, so "What's the weather?" works without naming a city:

```bash
export WEATHER_DEFAULT_LOCATION="Boston, Massachusetts"
./.venv/bin/python voice_assistant.py
```

The weather tool covers current conditions and daily forecasts up to 16 days ahead, in Fahrenheit and mph. The first matching city is named in the answer; give a state or country if the name is ambiguous. If the internet or weather service is unavailable, the assistant should report that instead of guessing. Weather answers require the selected Ollama model to support tool calling. [Liquid AI lists LFM2.5-1.2B-Instruct for tool calling](https://ollama.com/LiquidAI/lfm2.5-1.2b-instruct), but the final behavior still needs to be checked with the exact model installed on your Pi.

To see what Ollama asks the Pi to do, add `--debug-tools` to either command:

```bash
./.venv/bin/python assistant.py --debug-tools
./.venv/bin/python voice_assistant.py --debug-tools
```

Debug lines show the requested tool and arguments, the returned data or error, and lookup time. They appear separately from the assistant's answer. If Ollama answers without requesting a tool, the terminal says so. Leave the flag off for normal use.

## Try the microphone

Your existing Moonshine command is `moonshine-voice mic --language en --model-arch 4`. Confirm it transcribes speech, then stop it with Ctrl+C. Since Moonshine is installed in the repo's `.venv`, run:

```bash
./.venv/bin/python voice_assistant.py
```

Choose an Ollama model and start with **"Hey Baby"**. You can say "Hey Baby, explain black holes" in one utterance, or say the wake phrase and then ask your question. The phrase must be at the beginning of the transcribed utterance; case and punctuation do not matter.

After each reply, you have **15 seconds** to start a follow-up without repeating the wake phrase. Speaking within that window keeps the conversation active through your question and the next reply. The timer does not run while you are speaking, pausing within your question, or waiting for Ollama. After it expires, the screen shows `[Waiting for "Hey Baby".]`. Say **"go to sleep"** during an active conversation to return to waiting immediately. Completed conversation history is retained until the program exits.

Moonshine still listens and transcribes locally while waiting, but ordinary conversation is ignored and is not sent to Ollama. The assistant disables returned audio data from Moonshine's transcripts because it only needs the recognized words; this avoids the slowdown measured in the Pi diagnostics. During an active conversation, nearby speech is treated as directed at the assistant; this does not identify individual speakers.

The assistant waits for 1.5 seconds of extra quiet after Moonshine finishes a speech segment before responding. If you resume speaking during that wait, it combines the segments into one question. The program then prints the transcript and shows Ollama's reply as it arrives. If the reply is going in the wrong direction, start speaking again: the current answer stops, and your new question replaces it. The interrupted answer is not kept in conversation history. Press Ctrl+C to stop. Moonshine architecture 4 is Small Streaming; this mode uses the same architecture. Replies are currently printed rather than spoken. The same response chunks can later feed speech output without waiting for the whole reply.

To change the follow-up window, run `./.venv/bin/python voice_assistant.py --conversation-timeout 5` (seconds). This is separate from the short pause within a question.

To wait longer after a pause, run `./.venv/bin/python voice_assistant.py --pause-seconds 2.5` from the repo on the Pi. Use a smaller value if it feels too slow.

If you set up automatic activation, `python voice_assistant.py` works too.

## Diagnose microphone slowdowns on the Pi

Stop the assistant and any other microphone programs, then run in the same Python
environment you use for Moonshine:

```bash
python3 diagnose_audio.py
```

If Moonshine is in the repository's virtual environment, use
`./.venv/bin/python diagnose_audio.py` instead. No additional Python packages are
needed beyond your existing Moonshine installation.

The default is **30 minutes of listening**, plus model loading and shutdown:
15 minutes of Tiny Streaming with returned audio enabled, followed by 15 minutes
in a fresh process with returned audio disabled. Both phases use the same
instrumentation and suppress transcript printing. This isolates the cost of
returning audio; it is not an exact reproduction of the Moonshine CLI's terminal
rendering. Use a similar mix of short sentences, longer speech, and quiet in each
phase. An identical audio clip played from another device gives a more repeatable
comparison. Do not run the two phases concurrently.

The controller prints status every 30 seconds and samples the worker's current
RAM, swap allocation, CPU and thread count every second on Linux. A separate
worker heartbeat reports audio overflow notifications, estimated audio backlog,
transcript parsing time, processing time, and returned audio duration. CPU 100%
means approximately one occupied core. Backlog includes audio currently being
processed, excludes already lost microphone samples, and is approximate while
counters are updating. Parsing measures Python conversion; processing includes
audio ingestion, any triggered decoding, parsing, and listeners. The decode time
reported on completed lines is not the full pipeline cost.

Results go into a timestamped `diagnostics/` directory, excluded from Git:

- `summary.json`: outcomes and aggregate measurements for each phase.
- Each phase's `metrics.csv`: measurements over time.
- Each phase's `environment.json`: versions, audio device, settings, and a source
  fingerprint to identify the installed microphone implementation.
- Each phase's `worker.log`: startup messages and errors; no transcript callbacks
  print text, and the runner does not save recordings.

Send back the summary and CSV files; include environment and worker logs if a
phase fails. Ctrl+C saves partial results and stops the current worker. The
controller enforces the time limit even if the worker is stuck, and records when
forced termination was necessary. A startup timeout of five minutes permits
model loading/downloads without using listening time.

**If baseline stays healthy, the comparison is inconclusive.** Run longer than
the usual time to failure. Model warmup, different speech, other programs and
microphone noise can affect results. A growing heartbeat age means worker
measurements may be stale; independently collected CPU/memory still help.

Other experiments are selectable individually, each 30 minutes by default:

| Command suffix | Experiment |
| --- | --- |
| `--scenario baseline` | Tiny, returned audio enabled |
| `--scenario no-audio` | Tiny, returned audio disabled |
| `--scenario capture` | Microphone capture and discard, no Moonshine |
| `--scenario vad` | Moonshine VAD/segmentation, STT decoding disabled |
| `--scenario slower` | Returned audio disabled; transcription interval 1 second |
| `--scenario final-only` | Returned audio disabled; decode completed lines only |
| `--scenario host-block` | Returned audio disabled; host chooses capture block size |
| `--scenario larger-block` | Returned audio disabled; 2048 frames instead of 1024 |
| `--scenario buffered` | Returned audio disabled; request 0.2 seconds capture latency |

For example:

```bash
python3 diagnose_audio.py --scenario capture --minutes 30
python3 diagnose_audio.py --scenario no-audio --model small --minutes 30
python3 diagnose_audio.py --scenario no-audio --sample-rate 0 --minutes 30
```

`--sample-rate 0` selects the input device's native default rate; the normal
request is 16000 Hz mono. Moonshine can fall back to a supported native rate,
which is recorded in the CSV. The capture-only probe deliberately does not
silently change rates: if 16000 is unsupported, specify the same actual rate
shown in the Moonshine run. These tests don't add a second resampler.

Use `python3 -m sounddevice` to list devices and `--device 2` (with your actual
input index) to select one explicitly. `--latency 0.3` adjusts the buffered
experiment only. The default device latency may already be higher than 0.2;
compare the actual values in the CSV before interpreting that experiment.

The runner instruments private Moonshine capture/parsing methods. It reports an
error if the expected API is absent, rather than silently running an unmeasured
test. It does not change the installed package. Buffer clearing, stream rotation,
and a wake detector are deliberately not part of this comparison: restarting
sessions would obscure the accumulation we are trying to measure.

## Develop on Windows

Edit code in the Windows copy of this repository, then commit and push it to GitHub. Pull on the Pi to test with its installed models. Model files, recordings, and local settings stay outside Git.

The text chat uses Python 3.10 or newer and only the standard library. Voice mode also needs your existing `moonshine-voice` installation on the Pi. Install Python on Windows if you want to run the tests yourself, then use:

```bash
python -m unittest discover -s tests
```

The tests use a small local fake Ollama server and do not require model files.
