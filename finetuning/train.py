"""GPU-only LoRA pilot. Run inside the supplied Colab notebook."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def render_example(tokenizer, row, max_length):
    """Mask the entire prompt, including previous turns and tool descriptions."""
    prompt = tokenizer.apply_chat_template(row["messages"], tools=row["tools"],
                                           tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["messages"] + [row["target"]], tools=row["tools"],
                                         tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError(f"Chat template prefix mismatch for {row['id']}")
    for call in row["target"].get("tool_calls", []):
        if call["function"]["name"] not in full[len(prompt):]:
            raise ValueError("Tokenizer template silently dropped a tool call")
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError("Token boundary mismatch or empty training target")
    if len(ids) > max_length:
        raise ValueError(f"{row['id']} exceeds {max_length} tokens; do not silently truncate")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + ids[len(prefix):]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("finetuning/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", action="store_true", help="Also merge and export GGUF; can take extra time/RAM")
    args = parser.parse_args()
    if args.steps < 5:
        parser.error("Use at least five steps for the pilot")
    # Unsloth must patch libraries before transformers is imported.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running training.")
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split in ("train", "validation"):
        actual = hashlib.sha256((args.data / f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != manifest["splits"][split]["sha256"]:
            raise ValueError("Dataset changed after preparation; regenerate the manifest")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output directory already contains a run. Choose a new name or use --resume.")
    args.output.mkdir(parents=True, exist_ok=True)
    previous = args.output / "dataset-manifest.json"
    if args.resume and (not previous.exists() or json.loads(previous.read_text()) != manifest):
        raise ValueError("Resume requires the exact original dataset manifest")
    previous.write_text(json.dumps(manifest, indent=2))
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "environment.txt").write_text(frozen)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="LiquidAI/LFM2.5-1.2B-Instruct", max_seq_length=4096,
        dtype=torch.float16, load_in_4bit=False,
    )
    (args.output / "run.json").write_text(json.dumps({
        "base_model": "LiquidAI/LFM2.5-1.2B-Instruct", "revision": getattr(model.config, "_commit_hash", None),
        "steps": args.steps, "seed": 3407, "gpu": torch.cuda.get_device_name(),
        "precision": "float16", "max_length": 4096, "lora_rank": 16,
    }, indent=2))
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    datasets = {}
    for split in ("train", "validation"):
        rows = [json.loads(line) for line in (args.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        datasets[split] = Dataset.from_list([render_example(tokenizer, row, 4096) for row in rows])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    trainer = Trainer(
        model=model, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"), max_steps=args.steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=5e-5, warmup_ratio=0.1,
            fp16=True, bf16=False, optim="adamw_8bit", logging_steps=1,
            eval_strategy="steps", eval_steps=5, save_steps=5, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, prediction_loss_only=True, report_to="none", seed=3407,
        ),
    )
    checkpoint = get_last_checkpoint(str(args.output / "checkpoints")) if args.resume else None
    if args.resume and checkpoint is None:
        raise ValueError("No resumable checkpoint exists")
    trainer.train(resume_from_checkpoint=checkpoint)
    model.save_pretrained(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))
    (args.output / "metrics.json").write_text(json.dumps(trainer.evaluate(), indent=2))
    if args.export:
        model.save_pretrained_merged(str(args.output / "merged"), tokenizer, save_method="merged_16bit")
        model.save_pretrained_gguf(str(args.output / "gguf"), tokenizer, quantization_method="q4_k_m")
    print(f"Saved pilot to {args.output}. Validation loss is not tool-call accuracy; run the Pi evaluator.")


if __name__ == "__main__":
    main()
