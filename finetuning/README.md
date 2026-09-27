# LFM 1.2B fine-tuning experiment

Branch: `codex/lfm-finetuning`. Original fixes are saved in `fe3c836`.
Default application behavior still uses command rules. The new `--backend tools`
mode uses one LFM for ordinary streamed conversation and native Ollama tool calls.
Python validates arguments, executes actions, and formats simple results without
a second generation. Sleep/wake and microphone interruption remain in the voice loop.

## Status

This is a **pilot**, not a trained model or an accuracy claim. There are 51 training,
19 validation, and 35 held-out test scenarios. Examples are hand-authored, with a
few scripted follow-ups and disabled-capability cases. Expand and independently
review these before a full experiment (target roughly 1,000–3,000 training and
200–300 held-out scenarios). Do not generate training paraphrases from the test set.
Family-level separation and broader real speech coverage are still needed for a
strong evaluation. The test set stays outside the Colab upload bundle.

No GPU run, Pi benchmark, or GGUF export has been verified yet. The first run
records installed library versions; freeze that known-working environment after
the pilot succeeds. Do not interpret validation loss as tool-call accuracy.

## 1. Run the untuned baseline on the Pi

After this branch is pushed, retrieve it on the Pi (save any Pi changes first):

```bash
git fetch origin
git switch codex/lfm-finetuning
python -m finetuning.evaluate
```

Use your usual Python interpreter. Choose your existing LFM 1.2B model at the
prompt. This evaluator **never executes a tool**: no settings changes, timer
creation/cancellation, or weather network requests. It calls only Ollama.
Results go under `eval-results/lfm-.../`. It saves each result immediately.
The first request can include cold model loading; subsequent timings are warm.
Text cases only automatically check nonempty text/no tool: manually review them
for correct chat, clarification, and unsupported behavior. Follow-up histories
are scripted; this pilot does not measure full autonomous conversation success.

For interactive tests (these DO execute requested tools):

```bash
python assistant.py --backend tools --debug-tools
python voice_assistant.py --backend tools --debug-tools
```

Use normal commands without `--backend tools` to return to the existing rules.
Native Ollama tool parsing must work for the selected model/template; baseline
failures or raw tool tokens in text must be resolved before training.

## 2. Prepare the free Colab pilot

On Windows, from the repository:

```powershell
python -m finetuning.bundle
```

This creates `models/finetuning-upload.zip` with only training code and curated
train/validation data. It contains no recordings, API keys, or local settings.

1. Open https://colab.research.google.com/ and select **Upload notebook**.
2. Upload `finetuning/train_colab.ipynb` from this repo.
3. Select **Runtime → Change runtime type → T4 GPU** (or an available GPU).
4. Run the cells in order. Sign in to Google Drive when requested.
5. Upload `models/finetuning-upload.zip` when the upload picker appears.
6. The pilot runs 20 optimizer steps and saves checkpoints every five steps.
   All outputs go to `MyDrive/voice-assistant-baby/lfm-pilot-01`.

Colab's free GPU is subject to availability; no paid API is used. If interrupted
after a checkpoint, reconnect, rerun setup/upload, and set `RESUME = True`.
Use a new RUN directory for a changed dataset. Export may require additional
time and memory; the adapter is saved first so an export failure does not lose it.

Sources: https://unsloth.ai/docs/models/tutorials/lfm2.5 and
https://research.google.com/colaboratory/faq.html .

## 3. Import the exported candidate

Download the Q4_K_M GGUF from Drive and transfer it to the Pi, outside Git.
Create a `Modelfile` with the actual local path:

```text
FROM /absolute/path/to/exported-model.gguf
PARAMETER temperature 0
PARAMETER num_ctx 4096
```

```bash
ollama create baby-lfm-tools-pilot -f Modelfile
python -m finetuning.evaluate --model baby-lfm-tools-pilot
```

Check Ollama's imported chat template and tool parser against the original model
before comparing scores. Do not merge until the quantized candidate passes the
expanded evaluation and a Pi microphone/latency test. Keep baseline and candidate
results, data hashes, environment.txt, run.json, adapter, and tokenizer together.
The training model is `LiquidAI/LFM2.5-1.2B-Instruct`, not the Ollama GGUF.

## Next training iteration

Add reviewed examples to `seed_data.py`, run `python -m finetuning.data`, then
build a new bundle. Add new tools to `contract.py`, executor tests, and datasets
together. Mix new examples with old capabilities and normal conversation.
Run validation to select checkpoints; reserve test evaluation for chosen candidates.
