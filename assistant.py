"""A small text chat client for the Ollama server running on the Pi."""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
from datetime import date, datetime
from collections.abc import Callable
from http.client import HTTPConnection, HTTPException
from socket import SHUT_RDWR, socket
from threading import Event, Lock
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from local_clock import get_current_datetime
from settings import SettingsError, load_settings, update_settings
from weather import WeatherError, get_weather


OLLAMA_URL = "http://127.0.0.1:11434"
MAX_HISTORY_MESSAGES = 12
MAX_TOOL_ROUNDS = 3
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather or a forecast for a city. Use for weather, temperature, or rain questions. Location may be omitted to use a configured default.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and optional state or country. Omit to use the configured default location."},
                "day": {"type": "string", "description": "Use today or tomorrow literally. For a named month and day without a year, use MM-DD. For a date with an explicit year, use YYYY-MM-DD. Defaults to today."},
            },
        },
    },
}
CLOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description": "Read the current date and time from this Raspberry Pi's system clock. Use for current time or date questions, and answer with only the part the user requested.",
        "parameters": {"type": "object", "properties": {}},
    },
}
SET_WEATHER_PREFERENCES_TOOL = {
    "type": "function",
    "function": {
        "name": "set_weather_preferences",
        "description": "Save a default city or default temperature unit for future weather questions. Use only when the user asks to change a lasting preference.",
        "parameters": {
            "type": "object",
            "properties": {
                "default_city": {"type": "string", "description": "City and optional state/country. Empty string clears the default city."},
                "temperature_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
        },
    },
}
GET_WEATHER_PREFERENCES_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather_preferences",
        "description": "Read the saved default city and default temperature unit.",
        "parameters": {"type": "object", "properties": {}},
    },
}
WEATHER_INSTRUCTIONS = (
    "You can request get_weather for live weather information. Always use it for weather "
    "questions; never guess current conditions or forecasts. Do not include wind "
    "speed in weather answers. If the tool returns an error, "
    "explain the error instead of inventing weather. Use a city the user actually "
    "named; if none was named, call get_weather without a location so it uses the "
    "saved default. Never assume San Francisco or another city. For today or "
    "tomorrow, pass that exact word as the day; do not "
    "turn it into a calendar date. For a month and day without a year, use MM-DD "
    "and let the weather tool resolve the year. Use get_current_datetime for questions about "
    "the current date or time. If asked only for time, answer with time only; if asked "
    "only for date, answer with date only. Give both only when both were requested. "
    "Use set_weather_preferences when the user asks to "
    "change a default city or temperature unit, and get_weather_preferences when asked "
    "about those settings. Never claim a setting changed unless that tool succeeds. "
    "Do not change a default for a one-time weather question. Do not state a tool "
    "result before the tool returns."
)

MONTH_NUMBERS = {
    name.lower(): number
    for number in range(1, 13)
    for name in (calendar.month_name[number], calendar.month_abbr[number])
}
MONTH_NUMBERS["sept"] = 9
MONTH_PATTERN = "|".join(sorted(MONTH_NUMBERS, key=len, reverse=True))
YEARLESS_DATE_PATTERN = re.compile(
    rf"\b(?P<month>{MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)


class OllamaError(Exception):
    """An Ollama request failed or returned an unexpected response."""


class ChatInterrupted(Exception):
    """The user started speaking during a streamed reply."""


class ChatCancellation:
    """Stop a streamed reply by shutting down its local HTTP connection."""

    def __init__(self) -> None:
        self._cancelled = Event()
        self._lock = Lock()
        self._socket: socket | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            active_socket = self._socket
        if active_socket is not None:
            self._shutdown(active_socket)

    def check(self) -> None:
        if self._cancelled.is_set():
            raise ChatInterrupted()

    def attach(self, active_socket: socket | None) -> None:
        with self._lock:
            self._socket = active_socket
        if self._cancelled.is_set():
            if active_socket is not None:
                self._shutdown(active_socket)
            self.check()

    def detach(self, active_socket: socket | None) -> None:
        with self._lock:
            if self._socket is active_socket:
                self._socket = None

    @staticmethod
    def _shutdown(active_socket: socket) -> None:
        try:
            active_socket.shutdown(SHUT_RDWR)
        except OSError:
            pass


def request_json(path: str, payload: dict | None = None, *, base_url: str = OLLAMA_URL) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )

    try:
        with urlopen(request, timeout=300 if data is not None else 10) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            detail = json.load(exc).get("error", exc.reason)
        except (ValueError, AttributeError):
            detail = exc.reason
        raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise OllamaError(
            "Cannot reach Ollama on this computer. Check that Ollama is running "
            "with `ollama list`, then try again."
        ) from exc
    except ValueError as exc:
        raise OllamaError("Ollama returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise OllamaError("Ollama returned an unexpected response.")
    return result


def list_models(*, base_url: str = OLLAMA_URL) -> list[str]:
    result = request_json("/api/tags", base_url=base_url)
    models = result.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama did not return a model list.")
    return sorted(
        {model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)}
    )


def _stream_chat(
    model: str,
    messages: list[dict],
    *,
    on_chunk: Callable[[str], None] | None = None,
    cancellation: ChatCancellation | None = None,
    base_url: str = OLLAMA_URL,
    tools: list[dict] | None = None,
    format_schema: dict | None = None,
) -> dict:
    if cancellation is not None:
        cancellation.check()
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname is None:
        raise OllamaError("Ollama chat requires a local HTTP address.")
    connection = HTTPConnection(parsed_url.hostname, parsed_url.port, timeout=300)
    path = f"{parsed_url.path.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}
    if tools is not None:
        payload["tools"] = tools
    if format_schema is not None:
        payload["format"] = format_schema
    body = json.dumps(payload)
    parts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict] = []
    finished = False
    active_socket: socket | None = None
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        active_socket = connection.sock
        if cancellation is not None:
            cancellation.attach(active_socket)
        response = connection.getresponse()
        if response.status >= 400:
            try:
                detail = json.loads(response.read()).get("error", response.reason)
            except (ValueError, AttributeError):
                detail = response.reason
            raise OllamaError(f"Ollama returned HTTP {response.status}: {detail}")
        for line in response:
            if cancellation is not None:
                cancellation.check()
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise OllamaError("Ollama returned invalid streaming JSON.") from exc
            if not isinstance(event, dict):
                raise OllamaError("Ollama returned an unexpected streaming response.")
            if event.get("error"):
                raise OllamaError(f"Ollama error: {event['error']}")
            message = event.get("message")
            if message is not None and not isinstance(message, dict):
                raise OllamaError("Ollama returned an unexpected chat message.")
            content = message.get("content") if message else None
            if content is not None and not isinstance(content, str):
                raise OllamaError("Ollama returned unexpected chat content.")
            thinking = message.get("thinking") if message else None
            if thinking is not None and not isinstance(thinking, str):
                raise OllamaError("Ollama returned unexpected thinking content.")
            if thinking:
                thoughts.append(thinking)
            calls = message.get("tool_calls") if message else None
            if calls is not None:
                if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
                    raise OllamaError("Ollama returned unexpected tool calls.")
                tool_calls.extend(calls)
            if content:
                parts.append(content)
                if on_chunk is not None and not tool_calls:
                    on_chunk(content)
            if event.get("done") is True:
                finished = True
                break
    except (OSError, ValueError, HTTPException) as exc:
        if cancellation is not None:
            cancellation.check()
        raise OllamaError("Cannot complete the Ollama request. Check that Ollama is running.") from exc
    finally:
        if cancellation is not None:
            cancellation.detach(active_socket)
        connection.close()

    if cancellation is not None:
        cancellation.check()
    if not finished:
        raise OllamaError("Ollama stopped streaming before the reply finished.")
    message = {"role": "assistant", "content": "".join(parts)}
    if thoughts:
        message["thinking"] = "".join(thoughts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    on_chunk: Callable[[str], None] | None = None,
    cancellation: ChatCancellation | None = None,
    base_url: str = OLLAMA_URL,
) -> str:
    """Stream a plain text chat response without tools."""
    reply = _stream_chat(
        model, messages, on_chunk=on_chunk, cancellation=cancellation, base_url=base_url
    )["content"].strip()
    if not reply:
        raise OllamaError("Ollama returned no text. This model may not support chat.")
    return reply


def _relative_day_from_prompt(prompt: str) -> str | None:
    """Keep an unambiguous today/tomorrow request out of the model's date math."""
    relative_days = re.findall(r"\b(today|tomorrow)\b", prompt, flags=re.IGNORECASE)
    other_days = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tonight|yesterday)\b"
        r"|\b\d{4}-\d{2}-\d{2}\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if len(set(day.lower() for day in relative_days)) == 1 and other_days is None:
        return relative_days[0].lower()
    return None


def _yearless_day_from_prompt(prompt: str) -> str | None:
    """Use the user's named month/day without asking the model to invent a year."""
    if re.search(r"\b\d{4}\b", prompt):
        return None
    matches = list(YEARLESS_DATE_PATTERN.finditer(prompt))
    if len(matches) != 1:
        return None
    match = matches[0]
    month = MONTH_NUMBERS[match.group("month").lower()]
    day = int(match.group("day"))
    try:
        date(2000, month, day)
    except ValueError:
        return None
    return f"{month:02d}-{day:02d}"


def _day_from_prompt(prompt: str) -> str | None:
    return _relative_day_from_prompt(prompt) or _yearless_day_from_prompt(prompt)


def _weather_location_from_prompt(prompt: str, suggested_location: object) -> str:
    """Use a named place from the user's words, otherwise use the saved default."""
    non_locations = {
        "today", "tomorrow", "tonight", "yesterday", "now", "later", "next",
        "morning", "afternoon", "evening", "night", "weekend", "week",
        "my", "our", "your", "this", "here", "there", "home", "local",
        "nearby", "me", "default",
        "celsius", "celcius", "fahrenheit", "minutes", "hours", "days",
    }
    months = set(MONTH_NUMBERS)
    for marker in re.finditer(r"\b(?:in|for|at|near|around)\s+", prompt, re.IGNORECASE):
        phrase = prompt[marker.end():]
        phrase = re.split(
            r"[?!]|\b(?:today|tomorrow|tonight|yesterday|right now|"
            r"on|during|after|before|next|this|in|for|at)\b",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;:")
        words = re.findall(r"[a-z0-9]+", phrase.casefold())
        if not words:
            continue
        if words[0].isdigit() or words[0] in non_locations or words[0] in months:
            continue
        if words[0] == "the" and len(words) > 1:
            if words[1] in {"city", "area"} and len(words) > 3 and words[2] == "of":
                phrase = phrase.split(" ", 3)[3]
            elif words[1] in non_locations | {"city", "area"}:
                continue
        aliases = {"nyc": "New York City", "la": "Los Angeles", "sf": "San Francisco"}
        return aliases.get(phrase.casefold(), phrase)

    if isinstance(suggested_location, str):
        city = suggested_location.split(",", 1)[0].strip()
        if city and re.search(rf"\b{re.escape(city)}\b", prompt, re.IGNORECASE):
            return suggested_location
    return ""


def _simple_setting_command(prompt: str) -> dict[str, str] | None:
    """Handle clear spoken preference commands without depending on model behavior."""
    text = prompt.strip().rstrip(".!? ")
    text = re.sub(r"^(?:can|could|would) you\s+", "", text, flags=re.IGNORECASE)
    city_patterns = (
        r"(?:please\s+)?(?:set|change)(?:\s+my|\s+the)?\s+default\s+(?:city|location)\s+(?:to|as)\s+(.+)",
        r"(?:please\s+)?(?:use|make)\s+(.+?)\s+(?:as\s+)?(?:my|the)\s+default\s+(?:city|location)",
        r"(?:please\s+)?i want (?:my|the) default (?:city|location) (?:to be|set to) (.+)",
    )
    for pattern in city_patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            if not re.search(r"\band\b.*\b(celsius|celcius|fahrenheit)\b", city, re.IGNORECASE):
                return {"default_city": city}
    if re.fullmatch(r"(?:please\s+)?clear(?:\s+my|\s+the)?\s+default\s+(?:city|location)", text, re.IGNORECASE):
        return {"default_city": ""}
    unit_patterns = (
        r"(?:please\s+)?(?:set|change)(?:\s+my|\s+the)?\s+(?:temperature\s+)?units?\s+(?:to\s+)?(celsius|celcius|fahrenheit)",
        r"(?:please\s+)?(?:set|change)(?:\s+my|\s+the)?\s+default\s+(?:temperature\s+)?units?\s+(?:to\s+)?(celsius|celcius|fahrenheit)",
        r"(?:please\s+)?(?:use|switch\s+to)\s+(celsius|celcius|fahrenheit)(?:\s+by\s+default)?",
        r"(?:please\s+)?make\s+(celsius|celcius|fahrenheit)\s+the\s+default",
        r"(?:please\s+)?(?:show|give\s+me)\s+temperatures?\s+in\s+(celsius|celcius|fahrenheit)\s+by\s+default",
        r"(?:please\s+)?i want (?:the\s+)?temperatures?\s+(?:(?:to be|shown)\s+)?in\s+(celsius|celcius|fahrenheit)(?:\s+by\s+default|\s+from\s+now\s+on|\s+instead)",
    )
    for pattern in unit_patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match:
            return {"temperature_unit": match.group(1).lower()}
    return None


def _simple_clock_question(prompt: str) -> str | None:
    """Recognize common current-time questions without using model judgment."""
    text = prompt.strip().lower().rstrip(".!? ")
    text = text.replace("what's", "what is").replace("today's", "today")
    text = re.sub(r"^(?:please\s+|(?:can|could) you\s+)", "", text)
    patterns = {
        "both": (
            r"what is (?:the )?(?:date and time|time and date)(?: right now| now| today)?",
            r"(?:tell|give) me (?:the )?(?:date and time|time and date)",
        ),
        "time": (
            r"what time is it(?: right now| now)?",
            r"what is (?:the )?(?:current )?time(?: right now| now)?",
            r"(?:tell|give) me (?:the )?(?:current )?time(?: right now| now)?",
        ),
        "date": (
            r"what (?:date|day) is it(?: today| now)?",
            r"what day is today",
            r"what is (?:the |today |current )?date(?: today| now)?",
            r"(?:tell|give) me (?:the |today |current )?date",
        ),
    }
    for kind, options in patterns.items():
        if any(re.fullmatch(pattern, text) for pattern in options):
            return kind
    return None


def _clock_reply(kind: str, clock: dict[str, str]) -> str:
    observed = datetime.fromisoformat(clock["iso_datetime"])
    time_text = f"{observed.hour % 12 or 12}:{observed.minute:02d} "
    time_text += "AM" if observed.hour < 12 else "PM"
    date_text = f"{calendar.month_name[observed.month]} {observed.day}, {observed.year}"
    if kind == "time":
        return f"{time_text}."
    if kind == "date":
        return f"{date_text}."
    return f"{time_text} on {date_text}."


def _tool_result(call: dict, prompt: str) -> tuple[str, str]:
    function = call.get("function")
    if not isinstance(function, dict):
        return "unknown", json.dumps({"error": "Unknown tool request."})
    name = function.get("name")
    arguments = function.get("arguments", {})
    if name == "get_current_datetime":
        if arguments != {}:
            return name, json.dumps({"error": "This clock tool takes no arguments."})
        clock = get_current_datetime()
        asks_time = re.search(r"\btime\b", prompt, re.IGNORECASE) is not None
        asks_date = re.search(r"\b(date|day)\b", prompt, re.IGNORECASE) is not None
        if asks_time and not asks_date:
            clock = {"time": clock["time"], "timezone": clock["timezone"]}
        elif asks_date and not asks_time:
            clock = {"date": clock["date"], "weekday": clock["weekday"]}
        return name, json.dumps(clock)
    if name == "get_weather_preferences":
        if arguments != {}:
            return name, json.dumps({"error": "This settings lookup takes no arguments."})
        try:
            return name, json.dumps(load_settings())
        except SettingsError as exc:
            return name, json.dumps({"error": str(exc)})
    if name == "set_weather_preferences":
        if not isinstance(arguments, dict) or set(arguments) - {"default_city", "temperature_unit"}:
            return name, json.dumps({"error": "Invalid settings arguments."})
        try:
            result = update_settings(**arguments)
        except SettingsError as exc:
            result = {"error": str(exc)}
        return name, json.dumps(result)
    if name == "get_weather":
        if not isinstance(arguments, dict) or set(arguments) - {"location", "day"}:
            return name, json.dumps({"error": "Invalid weather arguments."})
        day = _day_from_prompt(prompt) or arguments.get("day", "today")
        location = _weather_location_from_prompt(prompt, arguments.get("location", ""))
        try:
            result = get_weather(location, day)
        except WeatherError as exc:
            result = {"error": str(exc)}
        return name, json.dumps(result)
    return "unknown", json.dumps({"error": "Unknown tool request."})


def _answer_with_history_legacy(
    model: str,
    prompt: str,
    history: list[dict[str, str]],
    *,
    on_chunk: Callable[[str], None] | None = None,
    on_tool_debug: Callable[[str], None] | None = None,
    cancellation: ChatCancellation | None = None,
    base_url: str = OLLAMA_URL,
) -> tuple[str, list[dict[str, str]]]:
    user_message = {"role": "user", "content": prompt}
    simple_command = _simple_setting_command(prompt)
    if simple_command is not None:
        if cancellation is not None:
            cancellation.check()
        if on_tool_debug is not None:
            on_tool_debug(f"[tool] request set_weather_preferences (spoken command): {json.dumps(simple_command)}")
        _, result_text = _tool_result(
            {"function": {"name": "set_weather_preferences", "arguments": simple_command}}, prompt
        )
        if cancellation is not None:
            cancellation.check()
        if on_tool_debug is not None:
            on_tool_debug(f"[tool] result set_weather_preferences: {result_text}")
        result = json.loads(result_text)
        if "error" in result:
            reply = f"I couldn't save that setting: {result['error']}"
        elif "default_city" in simple_command:
            reply = (
                f"Default city set to {result['default_city']}."
                if result["default_city"] else "Default city cleared."
            )
        else:
            reply = f"Default temperature unit set to {result['temperature_unit'].capitalize()}."
        if on_chunk is not None:
            on_chunk(reply)
        updated_history = (history + [user_message, {"role": "assistant", "content": reply}])[
            -MAX_HISTORY_MESSAGES:
        ]
        return reply, updated_history
    clock_kind = _simple_clock_question(prompt)
    if clock_kind is not None:
        if cancellation is not None:
            cancellation.check()
        if on_tool_debug is not None:
            on_tool_debug("[tool] request get_current_datetime (direct clock question): {}")
        clock = get_current_datetime()
        if cancellation is not None:
            cancellation.check()
        if on_tool_debug is not None:
            fields = {"time": clock["time"]} if clock_kind == "time" else {"date": clock["date"]}
            if clock_kind == "both":
                fields = {"time": clock["time"], "date": clock["date"]}
            on_tool_debug(f"[tool] result get_current_datetime: {json.dumps(fields)}")
        reply = _clock_reply(clock_kind, clock)
        if on_chunk is not None:
            on_chunk(reply)
        updated_history = (history + [user_message, {"role": "assistant", "content": reply}])[
            -MAX_HISTORY_MESSAGES:
        ]
        return reply, updated_history
    current_date = get_current_datetime()["date"]
    try:
        weather_defaults = load_settings()
        default_city = weather_defaults["default_city"] or "not set"
        default_unit = weather_defaults["temperature_unit"]
        defaults_context = f"Saved weather defaults: city={default_city!r}, temperature_unit={default_unit}. "
    except SettingsError:
        defaults_context = "Saved weather defaults could not be read. "
    messages: list[dict] = [
        {"role": "system", "content": f"Pi local date: {current_date}. {defaults_context}{WEATHER_INSTRUCTIONS}"},
        *history,
        user_message,
    ]
    for round_number in range(MAX_TOOL_ROUNDS + 1):
        if cancellation is not None:
            cancellation.check()
        response = _stream_chat(
            model,
            messages,
            on_chunk=on_chunk,
            cancellation=cancellation,
            base_url=base_url,
            tools=[WEATHER_TOOL, CLOCK_TOOL, SET_WEATHER_PREFERENCES_TOOL, GET_WEATHER_PREFERENCES_TOOL],
        )
        calls = response.get("tool_calls", [])
        if not calls:
            if on_tool_debug is not None and round_number == 0:
                on_tool_debug("[tool] Ollama replied without requesting a tool")
            reply = response["content"].strip()
            if not reply:
                raise OllamaError("Ollama returned no text. This model may not support chat or tools.")
            break
        if round_number == MAX_TOOL_ROUNDS:
            raise OllamaError("The model requested weather too many times without answering.")
        if len(calls) > 4:
            raise OllamaError("The model requested too many weather lookups at once.")
        messages.append(response)
        for call in calls:
            if cancellation is not None:
                cancellation.check()
            function = call.get("function", {})
            name = function.get("name", "unknown") if isinstance(function, dict) else "unknown"
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if on_tool_debug is not None:
                on_tool_debug(
                    f"[tool] request {name}: "
                    f"{json.dumps(arguments, ensure_ascii=False, default=str)[:1000]}"
                )
                if name == "get_weather" and isinstance(arguments, dict):
                    user_location = _weather_location_from_prompt(prompt, arguments.get("location", ""))
                    if arguments.get("location", "") != user_location:
                        if user_location:
                            on_tool_debug(f"[tool] using city from your words: {user_location!r}")
                        else:
                            on_tool_debug("[tool] no city in your question; using the saved default city")
                    user_day = _day_from_prompt(prompt)
                    if user_day is not None and arguments.get("day", "today") != user_day:
                        on_tool_debug(
                            f"[tool] using day={user_day!r} from your words "
                            "instead of the model's date"
                        )
            started = monotonic()
            tool_name, result = _tool_result(call, prompt)
            if cancellation is not None:
                cancellation.check()
            if on_tool_debug is not None:
                on_tool_debug(f"[tool] result {name} ({monotonic() - started:.2f}s): {result[:2000]}")
            messages.append({"role": "tool", "tool_name": tool_name, "content": result})
    else:
        raise OllamaError("The model requested weather too many times without answering.")
    updated_history = (history + [user_message, {"role": "assistant", "content": reply}])[
        -MAX_HISTORY_MESSAGES:
    ]
    return reply, updated_history


def answer_with_history(
    model: str,
    prompt: str,
    history: list[dict[str, str]],
    *,
    on_chunk: Callable[[str], None] | None = None,
    on_tool_debug: Callable[[str], None] | None = None,
    cancellation: ChatCancellation | None = None,
    base_url: str = OLLAMA_URL,
) -> tuple[str, list[dict[str, str]]]:
    """Use a structured request interpreter followed by Python workflows."""
    from structured_assistant import answer_structured

    return answer_structured(
        model, prompt, history, on_chunk=on_chunk,
        on_tool_debug=on_tool_debug, cancellation=cancellation, base_url=base_url,
    )


def choose_model(models: list[str], *, read=input, write=print) -> str:
    if not models:
        raise OllamaError("No models are installed in Ollama. Check `ollama list` on the Pi.")

    write("Available Ollama models:")
    for number, model in enumerate(models, start=1):
        write(f"  {number}. {model}")

    while True:
        choice = read("Choose a model number: ").strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        write(f"Enter a number from 1 to {len(models)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model.")
    parser.add_argument(
        "--debug-tools", action="store_true",
        help="Print tool requests and results while chatting.",
    )
    args = parser.parse_args()
    print("Looking for models on this Pi...")
    try:
        model = choose_model(list_models())
        print(f"\nChatting with {model}. Type /model to switch, /reset to clear context, or /quit to leave.")
        history: list[dict[str, str]] = []

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return 0

            if not prompt:
                continue
            if prompt.lower() == "/quit":
                print("Goodbye.")
                return 0
            if prompt.lower() == "/reset":
                history.clear()
                print("Conversation cleared.")
                continue
            if prompt.lower() == "/model":
                model = choose_model(list_models())
                history.clear()
                print(f"Now chatting with {model}. Conversation cleared.")
                continue

            try:
                reply_started = False

                def show_chunk(chunk: str) -> None:
                    nonlocal reply_started
                    if not reply_started:
                        print("\nAssistant: ", end="", flush=True)
                        reply_started = True
                    print(chunk, end="", flush=True)

                def show_tool_debug(event: str) -> None:
                    nonlocal reply_started
                    if reply_started:
                        print()
                        reply_started = False
                    print(event, flush=True)

                _, history = answer_with_history(
                    model, prompt, history, on_chunk=show_chunk,
                    on_tool_debug=show_tool_debug if args.debug_tools else None,
                )
            except OllamaError as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                continue
            print()
    except (OllamaError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, OllamaError):
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
