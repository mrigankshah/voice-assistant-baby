"""Interpret a spoken request once, then let Python manage each capability."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from time import monotonic

import assistant as core
from settings import SettingsError
from weather import WeatherError
from local_schedule import Schedule
from schedule_commands import handle_alarm, handle_timer


CHAT_INSTRUCTIONS = (
    "You are Baby, a helpful conversational assistant on a Raspberry Pi. "
    "Answer ordinary questions and creative requests directly in natural language. "
    "You can explain, tell stories, and tell jokes. Be concise enough for voice. "
    "The application handles live weather, the clock, saved preferences, timers, and alarms separately. "
    "Do not invent live conditions, claim an action happened, or claim to have used a tool."
)


class ConversationHistory(list):
    """Regular user/assistant messages plus task state kept out of Ollama chat history."""

    def __init__(self, messages=(), *, active_task: dict | None = None):
        super().__init__(messages)
        self.active_task = active_task

    def clear(self) -> None:
        super().clear()
        self.active_task = None


def _debug(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _check(cancellation: core.ChatCancellation | None) -> None:
    if cancellation is not None:
        cancellation.check()


def _finish(history: list, prompt: str, reply: str, task: dict | None) -> tuple[str, ConversationHistory]:
    messages = (list(history) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
    ])[-core.MAX_HISTORY_MESSAGES:]
    return reply, ConversationHistory(messages, active_task=task)


def _say(reply: str, on_chunk: Callable[[str], None] | None) -> str:
    if on_chunk is not None:
        on_chunk(reply)
    return reply


def _is_followup(prompt: str, active_task: dict | None) -> bool:
    if not active_task:
        return False
    text = prompt.strip().casefold()
    return bool(active_task.get("pending") or re.match(r"^(?:and\b|what about\b|how about\b|also\b|then\b)", text))


def _weather_day(prompt: str, active_task: dict | None, followup: bool) -> str | None:
    relative = re.findall(r"\b(?:today|tomorrow)\b", prompt, re.IGNORECASE)
    if len(set(word.casefold() for word in relative)) > 1:
        return None
    day = core._day_from_prompt(prompt)
    if day:
        return day
    spoken_iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", prompt)
    if spoken_iso:
        try:
            date.fromisoformat(spoken_iso.group())
        except ValueError:
            return None
        return spoken_iso.group()
    named_date = core.YEARLESS_DATE_PATTERN.search(prompt)
    if named_date:
        suffix = prompt[named_date.end():]
        explicit_year = re.match(r"\s*,?\s*(\d{4})\b", suffix)
        if explicit_year:
            try:
                return date(
                    int(explicit_year.group(1)),
                    core.MONTH_NUMBERS[named_date.group("month").lower()],
                    int(named_date.group("day")),
                ).isoformat()
            except ValueError:
                return None
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|weekend)\b", prompt, re.IGNORECASE):
        return None
    if followup and active_task and active_task.get("intent") == "weather":
        return active_task.get("day", "today")
    return "today"


def _temperature(value: object, unit: str) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return f"{value:g}°{unit}"


def _weather_reply(result: dict, day: str) -> str:
    location = result.get("location", "your city")
    unit = result.get("temperature_unit", "F")
    forecast = result.get("forecast") or {}
    description = forecast.get("condition") or "conditions unavailable"
    label = "Today" if day == "today" else "Tomorrow" if day == "tomorrow" else result.get("date", day)
    parts = [f"{label} in {location}: {description}."]
    current = result.get("current")
    if day == "today" and isinstance(current, dict):
        temperature = _temperature(current.get("temperature"), unit)
        if temperature:
            parts.append(f"Currently {temperature}.")
    high = _temperature(forecast.get("high"), unit)
    low = _temperature(forecast.get("low"), unit)
    if high and low:
        parts.append(f"High {high}, low {low}.")
    elif high:
        parts.append(f"High {high}.")
    rain = forecast.get("precipitation_probability_percent")
    if isinstance(rain, (int, float)) and not isinstance(rain, bool):
        parts.append(f"{rain:g}% chance of rain.")
    return " ".join(parts)


def _setting_reply(settings: dict, prompt: str) -> str:
    asks_city = re.search(r"\b(city|location)\b", prompt, re.IGNORECASE) is not None
    asks_unit = re.search(r"\b(unit|temperature|celsius|celcius|fahrenheit)\b", prompt, re.IGNORECASE) is not None
    city = settings.get("default_city", "")
    city_text = f"Your default city is {city}." if city else "You have no default city set."
    unit_text = f"Your default temperature unit is {settings['temperature_unit'].capitalize()}."
    if asks_city and not asks_unit:
        return city_text
    if asks_unit and not asks_city:
        return unit_text
    return f"{city_text} {unit_text}"


def _authorized_setting_change(prompt: str, name: str, value: str) -> bool:
    if re.search(r"\b(?:don't|do not|never|not yet|leave .* alone|if i asked|if you were)\b", prompt, re.IGNORECASE):
        return False
    if not re.search(r"^\s*(?:please\s+|(?:can|could|would) you\s+)*(?:set|change|use|switch|remember|clear|make|i want)\b", prompt, re.IGNORECASE):
        return False
    if name == "default_city":
        return (not value and re.search(r"\bclear\b", prompt, re.IGNORECASE) is not None) or (
            bool(value) and value.casefold() in prompt.casefold()
        )
    return value.casefold() in prompt.casefold()


def interpret_request(model: str, prompt: str, active_task: dict | None = None, *,
                      cancellation=None, base_url=core.OLLAMA_URL, on_tool_debug=None) -> dict:
    """Route explicit commands without a model call."""
    from command_router import route_command
    _check(cancellation)
    plan = route_command(prompt, active_task)
    _debug(on_tool_debug, "[route] explicit command")
    _check(cancellation)
    return plan


def answer_structured(
    model: str, prompt: str, history: list[dict[str, str]], *,
    on_chunk: Callable[[str], None] | None = None,
    on_tool_debug: Callable[[str], None] | None = None,
    cancellation: core.ChatCancellation | None = None,
    base_url: str = core.OLLAMA_URL,
) -> tuple[str, ConversationHistory]:
    _check(cancellation)
    active_task = getattr(history, "active_task", None)
    plan = interpret_request(model, prompt, active_task, cancellation=cancellation,
                             base_url=base_url, on_tool_debug=on_tool_debug)
    _check(cancellation)
    intent = plan["intent"]
    _debug(on_tool_debug, f"[route] {json.dumps(plan, ensure_ascii=False)}")

    if intent == "chat":
        messages = [
            {"role": "system", "content": CHAT_INSTRUCTIONS},
            *list(history),
            {"role": "user", "content": prompt},
        ]
        reply = core.chat(model, messages, on_chunk=on_chunk, cancellation=cancellation, base_url=base_url)
        _check(cancellation)
        return _finish(history, prompt, reply, None)

    if intent == "clarify":
        reply = plan["question"]
        return _finish(history, prompt, _say(reply, on_chunk), None)

    if intent == "unsupported_action":
        reply = "I can't do that yet. I can check weather, tell the time, manage settings, timers, and alarms."
        return _finish(history, prompt, _say(reply, on_chunk), None)

    if intent in {"timer", "alarm"}:
        execution_prompt = prompt
        if active_task and active_task.get("pending") and active_task.get("intent") == intent:
            plan = {**active_task, **plan}
            if plan.get("operation") == "cancel" and prompt.strip().isdigit():
                execution_prompt = f"{intent} number {prompt.strip()}"
        _check(cancellation)
        schedule = Schedule()
        reply, task = handle_timer(plan, execution_prompt, schedule) if intent == "timer" else handle_alarm(plan, execution_prompt, schedule)
        _debug(on_tool_debug, f"[tool] {intent}: {reply}")
        return _finish(history, prompt, _say(reply, on_chunk), task)

    if intent == "clock":
        part = core._simple_clock_question(prompt) or plan.get("clock_part") or "both"
        if part not in ("time", "date", "both"):
            part = "both"
        _debug(on_tool_debug, f"[tool] request get_current_datetime: {json.dumps({'part': part})}")
        observed = core.get_current_datetime()
        _check(cancellation)
        reply = core._clock_reply(part, observed)
        _debug(on_tool_debug, f"[tool] result get_current_datetime: {reply}")
        return _finish(history, prompt, _say(reply, on_chunk), None)

    if intent == "read_setting":
        _debug(on_tool_debug, "[tool] request get_weather_preferences: {}")
        try:
            settings = core.load_settings()
            reply = _setting_reply(settings, prompt)
            _debug(on_tool_debug, f"[tool] result get_weather_preferences: {json.dumps(settings)}")
        except SettingsError as exc:
            reply = f"I couldn't read your settings: {exc}"
            _debug(on_tool_debug, f"[tool] error get_weather_preferences: {exc}")
        _check(cancellation)
        return _finish(history, prompt, _say(reply, on_chunk), None)

    if intent == "write_setting":
        name, value = plan.get("setting_name"), plan.get("setting_value")
        if name not in ("default_city", "temperature_unit") or not isinstance(value, str):
            reply = "Please say the complete setting change, such as 'set my default city to Boston'."
            return _finish(history, prompt, _say(reply, on_chunk), None)
        if not _authorized_setting_change(prompt, name, value):
            reply = "I didn't change your settings. Please say the change directly if you want it saved."
            _debug(on_tool_debug, "[tool] rejected setting change not explicitly requested")
            return _finish(history, prompt, _say(reply, on_chunk), None)
        _check(cancellation)
        _debug(on_tool_debug, f"[tool] request set_weather_preferences: {json.dumps({name: value})}")
        try:
            saved = core.update_settings(**{name: value})
            reply = (
                f"Default city set to {saved['default_city']}." if saved["default_city"] else "Default city cleared."
            ) if name == "default_city" else f"Default temperature unit set to {saved['temperature_unit'].capitalize()}."
            _debug(on_tool_debug, f"[tool] result set_weather_preferences: {json.dumps(saved)}")
        except SettingsError as exc:
            reply = f"I couldn't save that setting: {exc}"
            _debug(on_tool_debug, f"[tool] error set_weather_preferences: {exc}")
        _check(cancellation)
        return _finish(history, prompt, _say(reply, on_chunk), None)

    followup = _is_followup(prompt, active_task) and (
        not active_task.get("pending") or plan.get("pending")
    )
    named_city = core._weather_location_from_prompt(prompt, plan.get("location"))
    if not named_city and followup and active_task and active_task.get("intent") == "weather":
        short_city = re.fullmatch(r"(?:[Ww]hat|[Hh]ow) about ([A-Z][a-z]+)\??", prompt.strip())
        if short_city:
            named_city = short_city.group(1)
    if named_city:
        city = named_city
        source = "explicit"
    elif followup and active_task and active_task.get("intent") == "weather" and active_task.get("location") and not re.search(
        r"\b(?:home|here|my city|default city)\b", prompt, re.IGNORECASE
    ):
        city = active_task["location"]
        source = "active task"
    else:
        try:
            city = core.load_settings()["default_city"]
        except SettingsError as exc:
            city = ""
            _debug(on_tool_debug, f"[tool] error reading default city: {exc}")
        source = "saved default"
    if not city:
        reply = "Which city should I check?"
        _debug(on_tool_debug, "[route] weather needs a city")
        return _finish(history, prompt, _say(reply, on_chunk), {"intent": "weather", "day": _weather_day(prompt, active_task, followup) or "today", "pending": True})
    day = _weather_day(prompt, active_task, followup)
    if not day:
        reply = "Which day would you like the weather for?"
        _debug(on_tool_debug, "[route] weather needs a day")
        return _finish(history, prompt, _say(reply, on_chunk), {"intent": "weather", "location": city, "pending": True})
    _debug(on_tool_debug, f"[route] weather location={city!r} ({source}), day={day!r}")
    _check(cancellation)
    started = monotonic()
    _debug(on_tool_debug, f"[tool] request get_weather: {json.dumps({'location': city, 'day': day})}")
    try:
        result = core.get_weather(city, day)
        _check(cancellation)
        reply = _weather_reply(result, day)
        task = {"intent": "weather", "location": city, "day": day}
        _debug(on_tool_debug, f"[tool] result get_weather ({monotonic() - started:.2f}s): {json.dumps(result)}")
    except WeatherError as exc:
        reply = f"I couldn't check the weather: {exc}"
        task = active_task
        _debug(on_tool_debug, f"[tool] error get_weather: {exc}")
    _check(cancellation)
    return _finish(history, prompt, _say(reply, on_chunk), task)
