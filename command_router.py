"""Fast, explicit spoken-command routing. No model or network calls."""

import re

import assistant as core


def route_command(prompt: str, active_task: dict | None = None) -> dict:
    text = prompt.strip().replace("’", "'").rstrip(".!? ")
    lower = text.casefold()
    lower = re.sub(r"^what's\b", "what is", lower)
    pending = active_task if active_task and active_task.get("pending") else None

    # A pending workflow may accept a short answer, but a fresh command takes priority.
    if re.search(r"\b(?:weather|forecast|timer|alarm|default city|default location|temperature unit|settings|preferences)\b", lower):
        pending = None

    if re.search(r"\b(?:don't|do not|never|not yet|if i|imagine|suppose)\b", lower) and not re.search(r"\b(?:but|instead)\s+(?:check|tell|show|give)\b", lower):
        return {"intent": "chat"}
    if re.match(r"^(?:what (?:is|are)|how (?:do|does|can)|why|explain|define)\b", lower) and re.search(r"\b(?:timer|alarm|weather|forecast|settings?|preferences?)\b", lower):
        # "What is the weather?" is a request, unlike "What is weather?".
        if not re.match(r"^what (?:is|are) (?:the |today'?s |tomorrow'?s )(?:weather|forecast)\b", lower):
            return {"intent": "chat"}
    if re.search(r"\b(?:just tell me whether you can|are you able to|tell me a story|tell me a joke)\b", lower):
        return {"intent": "chat"}

    if pending:
        from schedule_commands import alarm_time, timer_seconds
        kind, operation = active_task["intent"], active_task.get("operation")
        likely_answer = (
            kind == "weather" and bool(re.match(r"^(?:in|at|for|today|tomorrow|tonight)\b", lower) or re.fullmatch(r"[A-Z][a-z]+", text))
            or kind == "timer" and (timer_seconds(text) is not None if operation == "create" else text.isdecimal())
            or kind == "alarm" and (bool(re.search(r"\b\d+\b|\b(?:am|pm)\b", lower)) if operation == "create" else text.isdecimal())
        )
        if likely_answer:
            return {"intent": kind, "pending": True, "operation": operation}

    # Command words are mandatory for new actions. Bare rain/jacket questions chat.
    has_weather = bool(re.search(r"\b(?:weather|forecast)\b", lower))
    has_timer = bool(re.search(r"\btimers?\b", lower))
    has_alarm = bool(re.search(r"\balarms?\b", lower))
    has_setting = bool(re.search(r"\b(?:default (?:city|location)|temperature units?|settings?|preferences?)\b", lower))
    active = [name for name, present in (("weather", has_weather), ("timer", has_timer), ("alarm", has_alarm)) if present]
    if len(active) > 1 or (has_setting and active and re.search(r"\band\b", lower)
                            and re.search(r"\b(?:set|change|save|use|clear)\b", lower)):
        return {"intent": "clarify", "question": "Please ask for one action at a time."}

    command = core._simple_setting_command(text)
    if command and len(command) == 1:
        name, value = next(iter(command.items()))
        return {"intent": "write_setting", "setting_name": name, "setting_value": value}
    if has_setting:
        if re.search(r"\b(?:set|change|switch|save|clear|remove|forget|remember|use|make|update)\b", lower):
            return {"intent": "write_setting"}
        if re.search(r"\b(?:what|which|show|read|tell|check|list|have i|did i)\b", lower):
            return {"intent": "read_setting"}

    if has_timer:
        if re.search(r"\b(?:cancel|delete|remove|stop)\b", lower):
            return {"intent": "timer", "operation": "cancel"}
        if re.search(r"\b(?:list|show|what|which|how (?:long|much)|remaining|left)\b", lower):
            return {"intent": "timer", "operation": "list"}
        if re.search(r"\b(?:set|start|create|begin)\b", lower):
            return {"intent": "timer", "operation": "create"}
        return {"intent": "clarify", "question": "Would you like to set, list, or cancel a timer?"}
    if has_alarm:
        if re.search(r"\b(?:cancel|delete|remove|stop)\b", lower):
            return {"intent": "alarm", "operation": "cancel"}
        if re.search(r"\b(?:list|show|what|which)\b", lower):
            return {"intent": "alarm", "operation": "list"}
        if re.search(r"\b(?:set|create|wake me)\b", lower):
            return {"intent": "alarm", "operation": "create"}
        return {"intent": "clarify", "question": "Would you like to set, list, or cancel an alarm?"}

    if has_weather:
        if re.match(r"^(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:what|how|check|give|tell|show|will|is|do|need|forecast|weather)\b", lower):
            return {"intent": "weather"}
    part = core._simple_clock_question(text)
    if part:
        return {"intent": "clock", "clock_part": part}
    if re.search(r"\b(?:time|date)\b", lower) and re.search(r"\b(?:current|today|right now|now)\b", lower):
        if re.search(r"\b(?:what|tell|give|show)\b", lower) and not re.search(r"\b(?:time zones?|date fruit|time travel|date of birth)\b", lower):
            return {"intent": "clock", "clock_part": "both" if "time" in lower and "date" in lower else "time" if "time" in lower else "date"}

    if active_task and active_task.get("intent") == "weather" and re.match(r"^(?:and|what about|how about)\b", lower):
        if re.search(r"\b(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|in|for|at)\b", lower) or re.fullmatch(r"(?:[Ww]hat|[Hh]ow) about [A-Z][a-z]+", text):
            return {"intent": "weather"}
    if re.match(r"^(?:(?:please|can you|could you|would you|i want you to|i need you to)\s+)?(?:book|order|send|text|message|play|turn on|turn off|buy|purchase|schedule|remind|write down|make a note)\b", lower):
        return {"intent": "unsupported_action"}
    return {"intent": "chat"}
