"""Dry-run tool selection evaluation: never imports or invokes the executor."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import assistant as core
from finetuning.contract import validate_call


def score(target, prediction, tools):
    expected, actual = target.get("tool_calls", []), prediction.get("tool_calls", [])
    try:
        normalized = [validate_call(call, tools) for call in actual]
    except ValueError:
        return {"passed": False, "valid": False, "unwanted_tool": not expected and bool(actual)}
    if expected:
        wanted = [validate_call(call, tools) for call in expected]
        # Only documented default arguments are equivalent to omission.
        def defaults(call):
            name, args = call
            args = dict(args)
            for key, value in {"get_weather": {"day": "today"}, "create_timer": {"label": "timer"},
                               "create_alarm": {"recurrence": "once", "tomorrow": False}}.get(name, {}).items():
                args.setdefault(key, value)
            return name, args
        passed = [defaults(c) for c in normalized] == [defaults(c) for c in wanted]
    else:
        passed = not actual and bool(prediction.get("content", "").strip())
    return {"passed": passed, "valid": True, "unwanted_tool": not expected and bool(actual),
            "needs_text_review": not expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Ollama model name; otherwise show picker")
    parser.add_argument("--base-url", default=core.OLLAMA_URL)
    parser.add_argument("--cases", type=Path, default=Path("finetuning/data/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval-results") / ("lfm-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    model = args.model or core.choose_model(core.list_models(base_url=args.base_url))
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as output:
        for row in cases:
            started = perf_counter()
            chunks = []
            def chunk(text):
                chunks.append(perf_counter() - started)
            try:
                prediction = core._stream_chat(model, row["messages"], tools=row["tools"],
                    base_url=args.base_url, on_chunk=chunk,
                    options={"temperature": 0, "num_predict": 384, "num_ctx": 4096})
                result = score(row["target"], prediction, row["tools"])
                result["prediction"] = prediction
            except core.OllamaError as exc:
                result = {"passed": False, "valid": False, "error": str(exc)}
            result.update(id=row["id"], seconds=perf_counter() - started,
                          first_text_seconds=chunks[0] if chunks else None, target=row["target"])
            results.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(f"{row['id']}: {'PASS' if result['passed'] else 'FAIL'} ({result['seconds']:.2f}s)")
    summary = {"model": model, "cases": str(args.cases), "total": len(results),
               "passed": sum(r["passed"] for r in results),
               "unwanted_tools": sum(r.get("unwanted_tool", False) for r in results),
               "note": "Text passes check only no-tool/nonempty; manually review answers and clarification quality. Histories are scripted, not end-to-end conversations.",
               "warmup": "None: first request may include model loading."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
