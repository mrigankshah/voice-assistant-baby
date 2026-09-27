"""Opt-in single-model backend. Tool calls are validated and executed in Python."""

import json
from datetime import datetime

import assistant as core
from finetuning.contract import SYSTEM, TOOLS, validate_call
from local_schedule import Schedule
from settings import SettingsError
from weather import WeatherError


def predict(model, messages, *, tools=None, base_url=core.OLLAMA_URL, on_chunk=None, cancellation=None):
    return core._stream_chat(
        model, [{"role": "system", "content": SYSTEM}, *messages],
        tools=TOOLS if tools is None else tools, base_url=base_url,
        on_chunk=on_chunk, cancellation=cancellation,
        options={"temperature": 0, "num_predict": 384, "num_ctx": 4096},
    )


def execute(call):
    name, args = validate_call(call)
    if name == "get_weather":
        from structured_assistant import _weather_reply
        return _weather_reply(core.get_weather(**args), args.get("day", "today"))
    if name == "get_current_datetime":
        return core._clock_reply(args["part"], core.get_current_datetime())
    if name in {"get_weather_preferences", "set_weather_preferences"}:
        settings = core.load_settings() if name.startswith("get_") else core.update_settings(**args)
        city = settings["default_city"] or "not set"
        return f"Default city: {city}. Temperature unit: {settings['temperature_unit']}."
    schedule = Schedule()
    if name == "create_timer":
        job = schedule.create_timer(**args)
        return f"Timer #{job} set for {args['seconds']} seconds."
    if name == "create_alarm":
        job, due = schedule.create_alarm(**args)
        return f"Alarm #{job} set for {due.strftime('%A, %B %d at %I:%M %p')}, {args.get('recurrence', 'once')}."
    kind = "timer" if "timer" in name else "alarm"
    if name.startswith("list_"):
        jobs = schedule.list_jobs(kind)
        if not jobs:
            return f"You have no active {kind}s."
        return "; ".join(
            f"{j['label']} #{j['id']}: " + (
                f"{max(0, round(j['due_at'] - schedule.clock()))} seconds left" if kind == "timer"
                else datetime.fromtimestamp(j['due_at'], schedule.zone).strftime('%A %I:%M %p')
            ) for j in jobs
        ) + "."
    if name.startswith("cancel_all_"):
        return f"Cancelled {schedule.cancel_all(kind)} {kind}s."
    # Match numeric IDs only, never a different job whose label happens to be numeric.
    if not any(j["id"] == args["id"] for j in schedule.list_jobs(kind)):
        return f"I couldn't find {kind} #{args['id']}."
    count, job = schedule.cancel(kind, str(args["id"]))
    return f"Cancelled {kind} #{job['id']}." if job else f"I couldn't uniquely select {kind} #{args['id']}."


def answer(model, prompt, history, *, on_chunk=None, on_tool_debug=None,
           cancellation=None, base_url=core.OLLAMA_URL):
    messages = list(history) + [{"role": "user", "content": prompt}]
    result = predict(model, messages, base_url=base_url, on_chunk=on_chunk, cancellation=cancellation)
    calls = result.get("tool_calls", [])
    if not calls:
        reply = result.get("content", "")
        if not reply.strip():
            raise core.OllamaError("The model returned neither an answer nor a tool call.")
        return reply, (messages + [{"role": "assistant", "content": reply}])[-core.MAX_HISTORY_MESSAGES:]
    if on_tool_debug:
        on_tool_debug("[model tools] " + json.dumps(calls))
    valid_call = None
    if len(calls) != 1:
        reply = "Please ask for one action at a time. Nothing was executed."
    else:
        try:
            name, arguments = validate_call(calls[0])
            valid_call = {"type": "function", "function": {"name": name, "arguments": arguments}}
            if cancellation:
                cancellation.check()
            reply = execute(valid_call)
        except (ValueError, SettingsError, WeatherError) as exc:
            reply = f"I couldn't complete that request: {exc}"
    if on_tool_debug:
        on_tool_debug("[tool result] " + reply)
    if on_chunk:
        on_chunk(reply)
    # Store one complete turn, including the tool result, so follow-ups see real IDs.
    if valid_call is not None:
        turn = [{"role": "assistant", "content": "", "tool_calls": [valid_call]},
                {"role": "tool", "tool_name": valid_call["function"]["name"], "content": reply},
                {"role": "assistant", "content": reply}]
    else:
        turn = [{"role": "assistant", "content": reply}]
    # Trim whole turns at user boundaries; never leave an orphan tool result.
    updated = messages + turn
    while len(updated) > core.MAX_HISTORY_MESSAGES:
        boundary = next((i for i in range(1, len(updated)) if updated[i]["role"] == "user"), None)
        if boundary is None:
            break
        updated = updated[boundary:]
    return reply, updated
