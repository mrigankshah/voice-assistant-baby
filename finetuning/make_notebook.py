"""Build the checked-in Colab notebook from readable cells."""

import json
from pathlib import Path


def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell("markdown", """# Baby: LFM2.5 1.2B LoRA pilot
Choose **Runtime → Change runtime type → T4 GPU** (or an available GPU).
This is a 20-step plumbing pilot, not a deployment-ready training run.
It uploads only the curated examples, not recordings or personal settings.
Free GPU access is subject to Colab availability. No paid API key is required.
Run cells in order. Training checkpoints persist in your Google Drive.
The first GPU run must validate this environment; installed versions are saved with the run.
Source: https://unsloth.ai/docs/models/tutorials/lfm2.5
"""),
    cell("code", "%pip install -q unsloth datasets\n"),
    cell("code", """from google.colab import drive, files
from pathlib import Path
import zipfile
drive.mount('/content/drive')
print('Upload models/finetuning-upload.zip created on your laptop.')
uploaded = files.upload()
archive_name = next(name for name in uploaded if name.endswith('.zip'))
with zipfile.ZipFile(archive_name) as archive:
    destination = Path('/content/baby').resolve()
    for name in archive.namelist():
        if not (destination / name).resolve().is_relative_to(destination):
            raise ValueError('Unsafe archive path')
    archive.extractall(destination)
%cd /content/baby
"""),
    cell("code", """import subprocess, sys
RUN = '/content/drive/MyDrive/voice-assistant-baby/lfm-pilot-01'
RESUME = False  # True only to resume the same interrupted run and dataset.
EXPORT = True  # Saves adapter before attempting the larger GGUF export.
command = [sys.executable, '-m', 'finetuning.train', '--output', RUN, '--steps', '20']
if RESUME:
    command.append('--resume')
if EXPORT:
    command.append('--export')
subprocess.run(command, check=True)
"""),
    cell("markdown", """## After training
Find the adapter, environment.txt, run.json, metrics.json, and gguf folder in the Drive run directory.
Download the GGUF to the Pi. Follow finetuning/README.md to import it and run the same held-out evaluator.
Keep the original model. Do not replace your everyday assistant based on training loss alone.
If export fails, the adapter is already saved; keep the complete error output for troubleshooting.
"""),
]
notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
if __name__ == "__main__":
    Path("finetuning/train_colab.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
