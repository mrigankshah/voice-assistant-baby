# Voice Assistant Baby

This first milestone is a text chat with an Ollama model installed on the Raspberry Pi. It lists the models found on the Pi, lets you choose one, and keeps a short conversation history. Microphone input, Moonshine, and speech output will come later.

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

## Develop on Windows

Edit code in the Windows copy of this repository, then commit and push it to GitHub. Pull on the Pi to test with its installed models. Model files, recordings, and local settings stay outside Git.

The script uses Python 3.10 or newer and only the standard library. Install Python on Windows if you want to run the tests yourself, then use:

```bash
python -m unittest discover -s tests
```

The tests use a small local fake Ollama server and do not require model files.
