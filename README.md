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

The script uses Ollama's local API at `http://127.0.0.1:11434`. That HTTP connection stays on the Pi; it does not send your speech or prompts to the internet. You do not need to leave an interactive `ollama run <model-name>` session open. If `ollama list` cannot connect, start the Ollama service using your Pi's existing setup before running the script.

## Try the microphone

Your existing Moonshine command is `moonshine-voice mic --language en --model-arch 4`. Confirm it transcribes speech, then stop it with Ctrl+C. Since Moonshine is installed in the repo's `.venv`, run:

```bash
./.venv/bin/python voice_assistant.py
```

Choose an Ollama model and speak a question. The assistant waits for 1.5 seconds of extra quiet after Moonshine finishes a speech segment before responding. If you resume speaking during that wait, it combines the segments into one question. The program then prints the transcript and shows Ollama's reply as it arrives. If the reply is going in the wrong direction, start speaking again: the current answer stops, and your new question replaces it. The interrupted answer is not kept in conversation history. Press Ctrl+C to stop. Moonshine architecture 4 is Small Streaming; this mode uses the same architecture. Replies are currently printed rather than spoken. The same response chunks can later feed speech output without waiting for the whole reply.

To wait longer after a pause, run `./.venv/bin/python voice_assistant.py --pause-seconds 2.5` from the repo on the Pi. Use a smaller value if it feels too slow.

If you set up automatic activation, `python voice_assistant.py` works too.

## Develop on Windows

Edit code in the Windows copy of this repository, then commit and push it to GitHub. Pull on the Pi to test with its installed models. Model files, recordings, and local settings stay outside Git.

The text chat uses Python 3.10 or newer and only the standard library. Voice mode also needs your existing `moonshine-voice` installation on the Pi. Install Python on Windows if you want to run the tests yourself, then use:

```bash
python -m unittest discover -s tests
```

The tests use a small local fake Ollama server and do not require model files.
