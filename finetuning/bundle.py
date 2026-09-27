"""Create a small Colab upload zip containing only experiment code and pilot data."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from finetuning.data import prepare


def main():
    prepare(Path("finetuning/data"))
    output = Path("models/finetuning-upload.zip")
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in ("__init__.py", "train.py", "contract.py"):
            path = Path("finetuning") / name
            archive.write(path, path.as_posix())
        # Test set stays on the laptop/Pi, never enters the training upload.
        for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
            path = Path("finetuning/data") / name
            archive.write(path, path.as_posix())
    print(output.resolve())


if __name__ == "__main__":
    main()
