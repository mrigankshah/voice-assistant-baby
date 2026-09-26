"""Offline intent training and benchmark. No tool execution or live routing changes."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import random
from time import perf_counter

from capabilities import CAPABILITIES, resolve_route

ROOT = Path(__file__).resolve().parent
# Historical six-class experiment. Live command routing includes timers and alarms;
# any future classifier retraining needs new labeled data for those classes.
LABELS = {"chat", "weather", "clock", "read_setting", "write_setting", "unsupported_action"}
SPLITS = ("train", "validation", "test", "challenge")
INPUT_VERSION = 2


def render_input(example):
    """Compact application-owned context, never previous answers or preferences."""
    context = example.get("active_task") or {}
    intent = context.get("intent", "none")
    if intent != "none" and intent not in CAPABILITIES:
        raise ValueError(f"Unknown active task: {intent}")
    return f"Previous task: {intent}\nUser: {example['text']}"


def load_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seen, rows = set(), {}
    for split in SPLITS:
        groups = data.get(split, {})
        allowed = LABELS | ({"uncertain"} if split == "challenge" else set())
        if not set(groups) <= allowed or (split != "challenge" and set(groups) != LABELS):
            raise ValueError(f"Invalid labels in {split}: expected {sorted(allowed)}")
        rows[split] = []
        for label, examples in groups.items():
            if split != "challenge" and not examples:
                raise ValueError(f"Empty class: {split}/{label}")
            for example in examples:
                row = {"text": example} if isinstance(example, str) else dict(example)
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError("Examples must have nonempty text")
                row.update(label=label, group=row.get("group", "everyday"))
                key = " ".join(render_input(row).lower().split()).rstrip(".!?")
                if key in seen:
                    raise ValueError(f"Duplicate example across dataset: {row['text']}")
                seen.add(key)
                rows[split].append(row)
    return rows


def decide(scores, labels, threshold):
    best = max(range(len(scores)), key=lambda i: scores[i])
    return str(labels[best]) if scores[best] >= threshold else "uncertain"


def metrics(rows):
    accepted = [r for r in rows if r["predicted"] != "uncertain"]
    correct = sum(r["predicted"] == r["expected"] for r in rows)
    return {
        "cases": len(rows), "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "accepted": len(accepted), "coverage": len(accepted) / len(rows) if rows else None,
        "accepted_accuracy": sum(r["predicted"] == r["expected"] for r in accepted) / len(accepted) if accepted else None,
        "false_action_on_chat": sum(r["expected"] == "chat" and r["predicted"] in CAPABILITIES for r in rows),
        "false_action_on_unsupported": sum(r["expected"] == "unsupported_action" and r["predicted"] in CAPABILITIES for r in rows),
        "false_setting_write": sum(r["expected"] != "write_setting" and r["predicted"] == "write_setting" for r in rows),
        "confusion": dict(Counter(f"{r['expected']} -> {r['predicted']}" for r in rows)),
    }


def package_versions():
    result = {"python": platform.python_version()}
    for package in ("sentence-transformers", "scikit-learn", "torch", "setfit", "numpy", "transformers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def train(args, data):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    if args.artifact.exists():
        raise ValueError("Artifact already exists; choose a new versioned --artifact directory.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = perf_counter()
    encoder = SentenceTransformer(args.model, device="cpu", trust_remote_code=False,
                                  local_files_only=args.local_files_only)
    texts = [render_input(row) for row in data["train"]]
    labels = [row["label"] for row in data["train"]]
    if args.method == "setfit":
        from setfit import SetFitModel, Trainer, TrainingArguments
        # Same head fitting/export as the baseline; SetFit is unnecessary on Pi.
        model = SetFitModel(model_body=encoder, normalize_embeddings=True)
        settings = TrainingArguments(
            output_dir=str(args.artifact / "training"), batch_size=16,
            max_steps=args.steps, seed=args.seed, report_to=[],
            save_strategy="no", logging_steps=25, show_progress_bar=False,
        )
        trainer = Trainer(model=model, args=settings)
        trainer.train_embeddings([args.embedding_prefix + text for text in texts],
                                 [sorted(LABELS).index(label) for label in labels])
        encoder = model.model_body
    vectors = encoder.encode(texts, prompt=args.embedding_prefix,
                             normalize_embeddings=True, show_progress_bar=False)
    head = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    head.fit(vectors, labels)
    args.artifact.mkdir(parents=True, exist_ok=True)
    encoder.save(str(args.artifact / "encoder"))
    metadata = {
        "input_version": INPUT_VERSION, "method": args.method, "model": args.model,
        "labels": head.classes_.tolist(), "weights": head.coef_.tolist(), "bias": head.intercept_.tolist(),
        "training_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "training_count": len(texts), "embedding_prefix": args.embedding_prefix,
        "normalize_embeddings": True, "seed": args.seed, "steps": args.steps if args.method == "setfit" else 0,
        "packages": package_versions(), "training_seconds": perf_counter() - started,
    }
    (args.artifact / "head.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.method} artifact: {args.artifact}")
    print(f"Training: {metadata['training_seconds']:.1f}s; {len(texts)} examples")


class IntentClassifier:
    """Reusable offline inference; returns decisions, never executes workflows."""

    def __init__(self, artifact, threshold=0.0):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.np, self.threshold = np, threshold
        self.metadata = json.loads((Path(artifact) / "head.json").read_text(encoding="utf-8"))
        if self.metadata.get("input_version") != INPUT_VERSION:
            raise ValueError("Old artifact format: train a version 2 model in a new folder.")
        self.encoder = SentenceTransformer(str(Path(artifact) / "encoder"), device="cpu", local_files_only=True)
        self.weights, self.bias = np.array(self.metadata["weights"]), np.array(self.metadata["bias"])

    def predict(self, text, active_task=None):
        vector = self.encoder.encode(render_input({"text": text, "active_task": active_task}),
                                     prompt=self.metadata["embedding_prefix"], normalize_embeddings=True)
        logits = self.weights @ vector + self.bias
        scores = self.np.exp(logits - logits.max())
        scores /= scores.sum()
        labels = self.metadata["labels"]
        intent = decide(scores, labels, self.threshold)
        return {"intent": resolve_route(intent), "scores": dict(zip(labels, scores.tolist()))}


def evaluate(args, data):
    import numpy as np
    started = perf_counter()
    classifier = IntentClassifier(args.artifact, args.threshold)
    load_seconds = perf_counter() - started
    classifier.predict("Warm up the classifier")
    rows = []
    for repeat in range(args.repeat):
        for example in data[args.split]:
            started = perf_counter()
            prediction = classifier.predict(example["text"], example.get("active_task"))
            rows.append({**example, "expected": example["label"], "predicted": prediction["intent"],
                         "scores": prediction["scores"], "seconds": perf_counter() - started,
                         "repeat": repeat + 1})
    if not rows:
        raise ValueError("No evaluation examples")
    times = [r["seconds"] for r in rows]
    summary = {
        **metrics(rows), "model": classifier.metadata["model"], "method": classifier.metadata["method"],
        "platform": platform.platform(), "threads": args.threads, "split": args.split,
        "repeat": args.repeat, "threshold": args.threshold, "load_seconds": load_seconds,
        "median_seconds": float(np.median(times)), "p95_seconds": float(np.percentile(times, 95)),
        "by_group": {g: metrics([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})},
        "by_intent": {label: metrics([r for r in rows if r["expected"] == label]) for label in sorted({r["expected"] for r in rows})},
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256((args.artifact / "head.json").read_bytes()).hexdigest(),
        "packages": package_versions(), "artifact": classifier.metadata,
    }
    if args.split == "validation":
        summary["threshold_sweep"] = {}
        for threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            adjusted = [{**r, "predicted": resolve_route(decide(list(r["scores"].values()), list(r["scores"]), threshold))} for r in rows]
            summary["threshold_sweep"][str(threshold)] = metrics(adjusted)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "eval-results" / f"intent-{stamp}"
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    report = [f"# {summary['method']} / {args.split}", "",
              f"Correct: {summary['correct']}/{len(rows)}; accepted: {summary['accepted']}/{len(rows)}.",
              f"Median: {summary['median_seconds'] * 1000:.1f} ms; P95: {summary['p95_seconds'] * 1000:.1f} ms.",
              "", "Counts include repeats. Synthetic scenarios, not a user-traffic accuracy estimate.",
              "Scores are not calibrated probabilities. No workflows executed.", "", "## Failures (first repeat)", ""]
    for row in rows:
        if row["repeat"] == 1 and row["expected"] != row["predicted"]:
            report.append(f"- {row['text']} ({row.get('active_task') or 'no context'}): {row['expected']} -> {row['predicted']}")
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:7]))
    print(f"Results: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "train", "evaluate"])
    parser.add_argument("--data", type=Path, default=ROOT / "intent_data.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "models" / "intent-minilm-v2")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--method", choices=["frozen", "setfit"], default="frozen")
    parser.add_argument("--embedding-prefix", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--split", choices=SPLITS[1:], default="validation")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.repeat, args.threads, args.steps) < 1 or not 0 <= args.threshold <= 1:
        parser.error("repeat/threads/steps must be positive and threshold must be in [0, 1]")
    try:
        data = load_data(args.data)
        if args.command == "validate":
            for split, rows in data.items():
                print(split, dict(Counter(r["label"] for r in rows)))
            return 0
        import torch
        torch.set_num_threads(args.threads)
        if args.command == "train":
            train(args, data)
        else:
            evaluate(args, data)
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"Cannot {args.command}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
