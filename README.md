# Voice Assistant Baby

The Pi can chat by typed text or listen through Moonshine and print a reply from Ollama. Both modes list the models installed in Ollama, let you choose one, and remember a short conversation history. Ollama first interprets each request as a small structured action. Python resolves saved settings, dates, and follow-ups, then runs the right workflow. Ordinary conversation gets a separate prompt and still streams as it is generated. Spoken replies will come later.

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

Then try the same question by voice with "Hey Baby". Ollama identifies the request, and Python gets current conditions or a forecast from [Open-Meteo](https://open-meteo.com/en/docs). Location and forecast requests go to Open-Meteo over the internet. Ollama and Moonshine remain local. No weather API key or extra Python package is needed for personal use.

You can set a default location on the Pi before starting either program, so "What's the weather?" works without naming a city:

```bash
export WEATHER_DEFAULT_LOCATION="Boston, Massachusetts"
./.venv/bin/python voice_assistant.py
```

You can also change weather preferences by voice or in typed chat:

```text
Hey Baby, set my default city to Boston, Massachusetts.
Hey Baby, use Celsius by default.
Hey Baby, what's the weather tomorrow?
```

Say "use Fahrenheit by default" to switch back, or "clear my default city" to require a city in each weather question. The assistant confirms each change. Preferences are saved on the Pi in `assistant_settings.json` and survive restarts; this file is excluded from Git. A saved city takes priority over `WEATHER_DEFAULT_LOCATION`.

For each weather question, a city you name takes priority. A follow-up such as "and tomorrow?" keeps the last weather location. "At home" uses your saved default city, and a fresh weather question without a city uses that default too. If no city is available, the assistant asks for one. Python checks locations and dates against your words before it runs the lookup. The assistant keeps this active task in memory alongside the conversation history, and `/reset` clears it.

The weather workflow covers current conditions and daily forecasts up to 16 days ahead. Temperature defaults to Fahrenheit but can be changed to Celsius. The first matching city is named in the answer; give a state or country if the name is ambiguous. If the internet or weather service is unavailable, the assistant reports that instead of guessing. The selected Ollama model must support structured JSON output. Its interpretation quality still needs testing with the exact model installed on your Pi.

"What time is it?" gives only the time, "What's today's date?" gives only the date, and "What's the date and time?" gives both. These questions read the Pi's clock directly after interpretation. Check `timedatectl status` on the Pi if the reported date, time, or timezone is wrong. A weather request for just "today" or "tomorrow" uses that word directly, even if the model suggests a stale calendar date.

For a named month and day without a year, such as "September 28th", the weather workflow finds the matching date in the returned forecast. This can resolve to the current or next calendar year. If the date is beyond the available 16-day forecast, it reports that limit.

To see how a request was interpreted and executed, add `--debug-tools` to either command:

```bash
./.venv/bin/python assistant.py --debug-tools
./.venv/bin/python voice_assistant.py --debug-tools
```

Debug lines show the structured request (`[route]`), the chosen location and date, and the tool result or error. They appear separately from the assistant's answer. Leave the flag off for normal use.

## Try the microphone

Your existing Moonshine command is `moonshine-voice mic --language en --model-arch 4`. Confirm it transcribes speech, then stop it with Ctrl+C. Since Moonshine is installed in the repo's `.venv`, run:

```bash
./.venv/bin/python voice_assistant.py
```

Choose an Ollama model and start with **"Hey Baby"**. You can say "Hey Baby, explain black holes" in one utterance, or say the wake phrase and then ask your question. The phrase must be at the beginning of the transcribed utterance; case and punctuation do not matter.

After each reply, you have **15 seconds** to start a follow-up without repeating the wake phrase. Speaking within that window keeps the conversation active through your question and the next reply. The timer does not run while you are speaking, pausing within your question, or waiting for Ollama. After it expires, the screen shows `[Waiting for "Hey Baby".]`. Say **"go to sleep"** during an active conversation to return to waiting immediately. Completed conversation history is retained until the program exits.

Moonshine still listens and transcribes locally while waiting, but ordinary conversation is ignored and is not sent to Ollama. The assistant disables returned audio data from Moonshine's transcripts because it only needs the recognized words; this avoids the slowdown measured in the Pi diagnostics. During an active conversation, nearby speech is treated as directed at the assistant; this does not identify individual speakers.

The assistant waits for 1.5 seconds of extra quiet after Moonshine finishes a speech segment before responding. If you resume speaking during that wait, it combines the segments into one question. The program then prints the transcript, interprets the request, and streams ordinary chat replies. Routine weather, clock, and settings replies are formatted directly after their work completes. If the reply is going in the wrong direction, start speaking again: the current answer stops, and your new question replaces it. The interrupted answer is not kept in conversation history. Press Ctrl+C to stop. Moonshine architecture 4 is Small Streaming; this mode uses the same architecture. Replies are currently printed rather than spoken.

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

## Evaluate the current router

The assistant handles standard greetings, clock questions, and clear setting
commands directly. Other requests go to a compact JSON interpreter with explicit
schema instructions and examples. Only a follow-up receives the active task;
unrelated past replies are excluded from classification. The interpreter uses
temperature 0 and a 256-token output limit. Obvious unrelated action selections
are rejected with a clarification question. These checks do not guarantee correct
classification for every phrasing.

To test this actual routing path on the Pi without executing workflows, run:

```bash
python3 evaluate_routing.py --repeat 3
```

Choose the installed model. The command saves `report.md`, `results.jsonl`, and
`metadata.json` in a new `eval-results/routing-*` directory. It grades intent only;
argument accuracy and final answer quality still require checking. The debug log
records whether each decision came from a direct rule or the model, including
rejected model decisions. Stop the voice assistant first for meaningful timings.

## Evaluate real model tool decisions (previous architecture)

Run this on the Pi with Ollama running. Stop the voice assistant first so its
audio and model work do not distort the timing. This needs only standard Python;
it uses the real installed LLM but never executes tools or changes preferences.

Start with a five-case check and select your LFM model from the numbered list:

```bash
python3 evaluate_tools.py --limit 5
```

Then run all 32 starter cases three times to measure inconsistent decisions:

```bash
python3 evaluate_tools.py --repeat 3
```

Each case is an independent conversation; follow-up cases include a fixed
conversation history. The fixture date is September 25, 2026, the default city
is Boston (Chicago in one case), and the temperature unit is Celsius. Your own
saved settings are not used. Cases cover weather, clock requests, preferences,
follow-ups, negations, hypothetical requests, and ordinary chat.

The terminal prints the output folder. Read its `report.md` for failures and
timings, `results.jsonl` for full model replies, and `metadata.json` for model
parameters, templates, test prompts, and Ollama version. Results under
`eval-results/` are excluded from Git. Ctrl+C preserves completed cases and a
partial report. Run duration depends on model speed; this is not a fixed-duration
test. Each request has a 300-second timeout.

To compare models you have already installed, use their exact names from
`ollama list`, repeating `--model` for each model:

```bash
python3 evaluate_tools.py --model "FIRST-MODEL-NAME" --model "SECOND-MODEL-NAME" --repeat 3
```

Start with installed sampling defaults. Later, make a separate controlled run
with `--temperature 0.1`. Thinking is also left at the model default; use
`--think off` only if the model supports it. The default generation cap is 512
tokens per case. `token_limit` indicates a truncated response, not a reliable
tool decision; increase `--num-predict` or use a supported non-thinking mode.

This is a **baseline of the former direct tool-calling design**. The application
now uses a structured interpreter in `structured_assistant.py`; this evaluator
still sends the former tool list and prompt so the old results remain comparable.
It does not measure the new interpreter or execute a complete tool loop, grade answer
text, or test speech recognition. A no-tool pass only means the model didn't call
a tool, so inspect the reply for invented facts. A weather decision may omit the
city or explicitly provide the configured default; an unrelated invented city
fails. Timings include loading and thinking and measure the complete first
response, not time to the first streamed token. The raw results include Ollama's
separate timing fields. Compare accuracy as well as speed.

Add your real failure transcripts to `tool_eval_cases.json` (or a separate file
passed with `--cases`). Each case has an `id`, `prompt`, optional `history`, and
`expected` list of calls with `name` and `arguments`. An empty list means no tool
should be called. Expected argument matching is strict except for whitespace,
capitalization, and default weather city/day values; inspect failures for valid
alternative phrasing before treating the score as definitive. Once the baseline
is collected, use these same cases to evaluate a structured request interpreter
before enabling it in the voice assistant.

## Develop on Windows

Edit code in the Windows copy of this repository, then commit and push it to GitHub. Pull on the Pi to test with its installed models. Model files, recordings, and local settings stay outside Git.

The text chat uses Python 3.10 or newer and only the standard library. Voice mode also needs your existing `moonshine-voice` installation on the Pi. Install Python on Windows if you want to run the tests yourself, then use:

```bash
python -m unittest discover -s tests
```

The tests use a small local fake Ollama server and do not require model files.
