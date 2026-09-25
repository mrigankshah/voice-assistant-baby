"""Run the production router on real Ollama; no workflows execute."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import monotonic
import uuid

from assistant import OllamaError, choose_model, list_models, _day_from_prompt, _weather_location_from_prompt
from evaluate_tools import CASE_PATH, load_cases
from structured_assistant import interpret_request, ROUTER_SCHEMA, ROUTER_INSTRUCTIONS, ROUTER_EXAMPLES


TO_INTENT = {
    "get_weather": "weather", "get_current_datetime": "clock",
    "get_weather_preferences": "read_setting", "set_weather_preferences": "write_setting",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Exact installed model name; omit for the picker.")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("repeat must be positive")
    try:
        models = list_models()
        model = args.model or choose_model(models)
        if model not in models:
            parser.error("Use a model name shown by ollama list.")
        cases = load_cases(CASE_PATH)
    except (OllamaError, OSError, ValueError) as exc:
        print(f"Cannot start: {exc}")
        return 1
    output = Path(__file__).with_name("eval-results") / (
        "routing-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(json.dumps({
        "model": model, "repeat": args.repeat, "cases": cases,
        "schema": ROUTER_SCHEMA, "instructions": ROUTER_INSTRUCTIONS, "examples": ROUTER_EXAMPLES,
    }, indent=2), encoding="utf-8")
    print(f"Routing only: no weather requests or settings changes. Results: {output}", flush=True)
    rows = []
    interrupted = False
    try:
        with (output / "results.jsonl").open("w", encoding="utf-8") as log:
            for repeat in range(1, args.repeat + 1):
                for case in cases:
                    # Existing follow-up fixtures begin with an explicit weather
                    # request. Construct that successful prior task, without executing it.
                    task = None
                    if case.get("history"):
                        previous = case["history"][0]["content"]
                        city = _weather_location_from_prompt(previous, "")
                        if city:
                            task = {"intent": "weather", "location": city, "day": _day_from_prompt(previous) or "today"}
                    expected = TO_INTENT[case["expected"][0]["name"]] if case["expected"] else "chat"
                    row = {"id": case["id"], "repeat": repeat, "prompt": case["prompt"], "expected_intent": expected, "active_task": task}
                    events = []
                    started = monotonic()
                    print(f"{case['id']} ({repeat}/{args.repeat}) ...", flush=True)
                    try:
                        row["plan"] = interpret_request(model, case["prompt"], task, on_tool_debug=events.append)
                        row["status"] = "pass" if row["plan"]["intent"] == expected else "wrong_intent"
                    except OllamaError as exc:
                        row.update(status="error", error=str(exc))
                    row.update(seconds=round(monotonic() - started, 3), debug=events)
                    rows.append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    print(f"  {row['status']} ({row['seconds']}s)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        passed = sum(row["status"] == "pass" for row in rows)
        report = ["# Production routing results", "", f"Intent matches: {passed}/{len(rows)}.",
                  f"Interrupted: {interrupted}.", "",
                  "This includes direct routes and model routes. It grades intent only, not argument accuracy, final answers, or speech recognition.", ""]
        for row in rows:
            if row["status"] != "pass":
                report.append(f"- {row['id']} repeat {row['repeat']}: expected {row['expected_intent']}; got {json.dumps(row.get('plan', row.get('error')))}")
        (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Report: {output / 'report.md'}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
