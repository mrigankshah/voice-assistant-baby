"""Prepare portable pilot JSONL and validate schemas without ML dependencies."""

import argparse
import hashlib
import json
from pathlib import Path

from finetuning.contract import SYSTEM, VERSION, tools_for, validate_call
from finetuning.seed_data import rows


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"contract": VERSION, "status": "hand-authored pilot; expand before deployment", "splits": {}}
    seen = {}
    for split in ("train", "validation", "test"):
        records = rows(split)
        for row in records:
            tools = tools_for(row.get("tools"))
            # Include capabilities and whole histories when checking duplicate scenarios.
            key = json.dumps([row["messages"], tools], sort_keys=True).casefold()
            if key in seen:
                raise ValueError(f"Duplicate scenario: {row['id']} and {seen[key]}")
            seen[key] = row["id"]
            for call in row["target"].get("tool_calls", []):
                validate_call(call, tools)
            row["tools"] = tools
            row["messages"] = [{"role": "system", "content": SYSTEM}, *row["messages"]]
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        (output / f"{split}.jsonl").write_text(payload, encoding="utf-8")
        manifest["splits"][split] = {"count": len(records), "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("finetuning/data"))
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
