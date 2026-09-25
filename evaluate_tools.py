"""Measure real Ollama tool decisions without executing any tools.

This benchmarks the first model response, before the application's direct
handlers and argument corrections. It does not grade answer text or test STT.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from time import monotonic
import uuid

from assistant import (
    CLOCK_TOOL, GET_WEATHER_PREFERENCES_TOOL, OLLAMA_URL, OllamaError,
    SET_WEATHER_PREFERENCES_TOOL, WEATHER_INSTRUCTIONS, WEATHER_TOOL,
    choose_model, list_models, request_json,
)

CASE_PATH = Path(__file__).with_name("tool_eval_cases.json")
TOOLS = [WEATHER_TOOL, CLOCK_TOOL, SET_WEATHER_PREFERENCES_TOOL, GET_WEATHER_PREFERENCES_TOOL]


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case file must contain a nonempty JSON list.")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Each case needs a text id.")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case id: {case['id']}")
        seen.add(case["id"])
        if not isinstance(case.get("prompt"), str) or not isinstance(case.get("expected"), list):
            raise ValueError(f"Case {case['id']} needs a prompt and expected call list.")
        for call in case["expected"]:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments", {}), dict):
                raise ValueError(f"Invalid expected call in {case['id']}.")
    return cases


def build_messages(case: dict) -> list[dict]:
    # Fixed fixtures make models and repeated runs comparable. Never read the
    # user's settings file or real clock while building evaluation requests.
    city = case.get("default_city", "Boston")
    unit = case.get("temperature_unit", "celsius")
    context = (
        f"Pi local date: 2026-09-25. Saved weather defaults: "
        f"city={city or 'not set'!r}, temperature_unit={unit}. "
    )
    return [
        {"role": "system", "content": context + WEATHER_INSTRUCTIONS},
        *case.get("history", []),
        {"role": "user", "content": case["prompt"]},
    ]


def normalized_arguments(name: str, arguments: dict, case: dict) -> dict:
    result = dict(arguments)
    if name == "get_weather":
        # Omitting the city and supplying the known default are both valid
        # decisions. An invented third city must still fail this comparison.
        result["location"] = result.get("location", "") or case.get("default_city", "Boston")
        result["day"] = result.get("day", "today")
    return {key: value.strip().casefold() if isinstance(value, str) else value for key, value in result.items()}


def grade_response(response: dict, case: dict) -> dict:
    message = response.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama returned no message object.")
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError("Ollama returned an invalid tool call list.")
    actual = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Ollama returned an invalid tool call.")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        actual.append({"name": function["name"], "arguments": arguments})
    expected = case["expected"]
    if not actual and not message.get("content", "").strip():
        status = "empty_response"
    elif actual and not expected:
        status = "unnecessary_tool"
    elif expected and not actual:
        status = "missing_tool"
    elif [call["name"] for call in actual] != [call["name"] for call in expected]:
        status = "wrong_tools"
    elif any(
        normalized_arguments(got["name"], got["arguments"], case)
        != normalized_arguments(want["name"], want.get("arguments", {}), case)
        for got, want in zip(actual, expected)
    ):
        status = "wrong_arguments"
    else:
        status = "pass"
    return {"status": status, "actual": actual, "expected": expected}


def summarize(rows: list[dict]) -> dict:
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    # Network failures do not become misleading fast decision timings.
    times = sorted(row["wall_seconds"] for row in rows if row["status"] != "error")
    return {
        "completed": len(rows), "counts": counts,
        "pass_rate": counts.get("pass", 0) / len(rows) if rows else 0,
        "median_seconds": statistics.median(times) if times else None,
        "p95_seconds": times[math.ceil(len(times) * .95) - 1] if times else None,
    }


def write_report(path: Path, rows: list[dict], models: list[str], interrupted: bool) -> None:
    lines = [
        "# Tool decision evaluation", "",
        "This grades the first model decision, before app corrections. No tools were executed.",
        "A pass for a no-tool case only means no tool was called; review the answer text separately.",
        "Timings are complete first responses, including any model loading and thinking. They are not time to first spoken word.",
        "Results use fixed test preferences and a fixed date, not your personal settings.", "",
        "| Model | Decisions passed | Median | P95 |", "| --- | --- | --- | --- |",
    ]
    for model in models:
        group = [row for row in rows if row["model"] == model]
        summary = summarize(group)
        median = f"{summary['median_seconds']:.2f}s" if summary["median_seconds"] is not None else "n/a"
        p95 = f"{summary['p95_seconds']:.2f}s" if summary["p95_seconds"] is not None else "n/a"
        lines.append(f"| {model} | {summary['counts'].get('pass', 0)}/{len(group)} | {median} | {p95} |")
    lines += ["", "## Failures", ""]
    for row in rows:
        if row["status"] != "pass":
            lines += [
                f"- **{row['model']} / {row['case_id']} / repeat {row['repeat']}: {row['status']}**",
                f"  - User: {row['prompt']}",
                f"  - Expected: `{json.dumps(row.get('expected', []), ensure_ascii=False)}`",
                f"  - Actual: `{json.dumps(row.get('actual', []), ensure_ascii=False)}`",
            ]
            if row.get("error"):
                lines.append(f"  - Error: {row['error']}")
    if interrupted:
        lines += ["", "Run interrupted. Completed cases are preserved; these results are partial."]
    lines += ["", "Full prompts, responses, errors, and Ollama timings are in results.jsonl.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="Exact installed Ollama name; repeat to compare models. Omit to pick interactively.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Run only the first N cases for a quick check.")
    parser.add_argument("--cases", type=Path, default=CASE_PATH)
    parser.add_argument("--output", type=Path, help="New directory for results; must not already exist.")
    parser.add_argument("--base-url", default=OLLAMA_URL)
    parser.add_argument("--temperature", type=float, help="Override sampling; omit to use the model's installed default.")
    parser.add_argument("--num-predict", type=int, default=512, help="Cap generated tokens per case (default 512).")
    parser.add_argument("--think", choices=["default", "on", "off"], default="default")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.num_predict < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeat, limit, and num-predict must be positive.")
    try:
        cases = load_cases(args.cases)
        if args.limit:
            cases = cases[:args.limit]
        installed = list_models(base_url=args.base_url)
        models = list(dict.fromkeys(args.model or [choose_model(installed)]))
        if any(model not in installed for model in models):
            raise ValueError("Use exact installed model names from ollama list: " + ", ".join(installed))
        output = args.output or Path(__file__).with_name("eval-results") / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        output.mkdir(parents=True, exist_ok=False)
        options = {"num_predict": args.num_predict}
        if args.temperature is not None:
            options["temperature"] = args.temperature
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "models": models, "repeat": args.repeat, "options": options,
            "think": args.think, "base_url": args.base_url,
            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "cases": cases, "tools": TOOLS, "system_instructions": WEATHER_INSTRUCTIONS,
        }
        # Record actual installation metadata instead of guessing model settings.
        metadata["ollama_version"] = request_json("/api/version", base_url=args.base_url)
        metadata["model_info"] = {}
        for model in models:
            info = request_json("/api/show", {"model": model}, base_url=args.base_url)
            metadata["model_info"][model] = {key: info.get(key) for key in ("details", "parameters", "template", "capabilities")}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OllamaError, OSError, ValueError) as exc:
        print(f"Cannot start evaluation: {exc}", file=sys.stderr)
        return 1

    print(f"Testing {len(cases)} cases x {args.repeat} repeats x {len(models)} models. No tools will execute.", flush=True)
    print(f"Reports: {output.resolve()}", flush=True)
    rows = []
    interrupted = False
    try:
        with (output / "results.jsonl").open("w", encoding="utf-8") as results_file:
            for model in models:
                for repetition in range(1, args.repeat + 1):
                    for case in cases:
                        row = {"model": model, "case_id": case["id"], "repeat": repetition, "prompt": case["prompt"], "expected": case["expected"]}
                        messages = build_messages(case)
                        payload = {"model": model, "messages": messages, "tools": TOOLS, "stream": False, "options": options, "keep_alive": "10m"}
                        if args.think != "default":
                            payload["think"] = args.think == "on"
                        print(f"[{model}] {case['id']} ({repetition}/{args.repeat}) ...", flush=True)
                        started = monotonic()
                        try:
                            response = request_json("/api/chat", payload, base_url=args.base_url)
                            row["response"] = response
                            if response.get("error"):
                                raise ValueError(str(response["error"]))
                            if response.get("done") is not True:
                                raise ValueError("Ollama response was incomplete.")
                            if response.get("done_reason") == "length":
                                row.update(status="token_limit", error="Response hit num-predict. Increase the limit or test with thinking off if supported.")
                            else:
                                row.update(grade_response(response, case))
                        except (OllamaError, ValueError, TypeError) as exc:
                            row.update(status="error", error=str(exc))
                        row["wall_seconds"] = round(monotonic() - started, 3)
                        row["messages"] = messages
                        rows.append(row)
                        results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        results_file.flush()
                        print(f"  {row['status']} ({row['wall_seconds']:.2f}s)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped. Saving completed cases.")
    finally:
        write_report(output / "report.md", rows, models, interrupted)
        (output / "summary.json").write_text(json.dumps({
            "interrupted": interrupted,
            "models": {model: summarize([row for row in rows if row["model"] == model]) for model in models},
        }, indent=2), encoding="utf-8")
    print(f"Read: {(output / 'report.md').resolve()}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
