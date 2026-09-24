# Voice Assistant Baby

The Pi can chat by typed text or listen through Moonshine and print a reply from Ollama. Both modes list the models installed in Ollama, let you choose one, keep a short conversation history, and show Ollama's reply as it is generated. Spoken replies will come later.

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

The script uses Ollama's local API at `http://127.0.0.1:11434`. You do not need to leave an interactive `ollama run <model-name>` session open. If `ollama list` cannot connect, start the Ollama service using your Pi's existing setup before running the script.

## Try the microphone

Your existing Moonshine command is `moonshine-voice mic --language en --model-arch 4`. Confirm it transcribes speech, then stop it with Ctrl+C. In the same Python environment, run:

```bash
python3 voice_assistant.py
```

Choose an Ollama model, speak a question, and pause. The program prints the finished Moonshine transcript, then shows Ollama's reply as it arrives. Press Ctrl+C to stop. Moonshine architecture 4 is Small Streaming; this mode uses the same architecture. It sends only finished transcript lines to Ollama and currently prints replies rather than speaking them. The same response chunks can later feed speech output without waiting for the whole reply.

If Moonshine is installed in a `.venv` inside this repository, run `./.venv/bin/python voice_assistant.py` instead; this uses the right environment without activating it. If you set up automatic activation, `python voice_assistant.py` works too.

## Develop on Windows

Edit code in the Windows copy of this repository, then commit and push it to GitHub. Pull on the Pi to test with its installed models. Model files, recordings, and local settings stay outside Git.

The text chat uses Python 3.10 or newer and only the standard library. Voice mode also needs your existing `moonshine-voice` installation on the Pi. Install Python on Windows if you want to run the tests yourself, then use:

```bash
python -m unittest discover -s tests
```

The tests use a small local fake Ollama server and do not require model files.
