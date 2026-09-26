"""Parse the small, explicit timer and alarm command vocabulary."""

import re
from datetime import datetime

from local_schedule import Schedule

WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
         "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
UNITS = "one|two|three|four|five|six|seven|eight|nine"
TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
NUMBER = rf"\d+|(?:{TENS})[- ](?:{UNITS})|(?:{TENS})|nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|eleven|ten|{UNITS}|an?"


def number(text):
    text = text.casefold().strip().replace("-", " ")
    if text.isdecimal():
        return int(text)
    if text in WORDS:
        return WORDS[text]
    parts = text.split()
    if len(parts) == 2 and parts[0] in WORDS and WORDS[parts[0]] >= 20 and parts[1] in WORDS and WORDS[parts[1]] < 10:
        return WORDS[parts[0]] + WORDS[parts[1]]
    return None


def timer_seconds(text):
    matches = list(re.finditer(rf"\b({NUMBER})\s*(seconds?|minutes?|hours?)\b", text, re.IGNORECASE))
    if not matches:
        return None
    total = 0
    for match in matches:
        amount = number(match.group(1))
        if amount is None:
            return None
        unit = match.group(2).casefold()
        total += amount * (3600 if unit.startswith("hour") else 60 if unit.startswith("minute") else 1)
    return total


def identifier(text, kind):
    if text.strip().isdecimal():
        return text.strip()
    match = re.search(rf"\b{kind}\s*(?:number|#)\s*(\d+)\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:called|named)\s+([\w-]+)\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(rf"\b([\w-]+)\s+{kind}\b", text, re.IGNORECASE)
    if match and match.group(1).casefold() not in {"a", "the", "my", "that", "this", "an", "cancel", "stop", "delete", "remove"}:
        return match.group(1)
    return None


def handle_timer(plan, prompt, schedule: Schedule):
    op = plan.get("operation", "create")
    if op == "list":
        jobs = schedule.list_jobs("timer")
        if not jobs:
            return "You have no active timers.", None
        now = schedule.clock()
        return "Active timers: " + "; ".join(
            f"{job['label']} #{job['id']}, {max(0, round(job['due_at'] - now))} seconds left"
            for job in jobs
        ) + ".", None
    if op == "cancel":
        if re.search(r"\ball\b", prompt, re.IGNORECASE):
            count = schedule.cancel_all("timer")
            return f"Cancelled {count} timers." if count else "You have no active timers.", None
        count, job = schedule.cancel("timer", identifier(prompt, "timer"))
        if job:
            return f"Cancelled {job['label']} timer #{job['id']}.", None
        if not count:
            return "I couldn't find that timer.", None
        return "Which timer number should I cancel?", {"intent": "timer", "pending": True, "operation": "cancel"}
    seconds = timer_seconds(prompt)
    if seconds is None:
        return "How long should the timer run?", {"intent": "timer", "pending": True, "operation": "create"}
    if not 1 <= seconds <= 7 * 86400:
        return "Please choose a timer between one second and seven days.", None
    match = re.search(r"\b(?:called|named)\s+([\w-]+)\b|\b([\w-]+)\s+timer\b", prompt, re.IGNORECASE)
    label = (match.group(1) or match.group(2)) if match else "timer"
    if label and label.casefold() in {"a", "the", "my", "new", "one", "another", "minute", "hour", "set", "start", "create", "begin"}:
        label = "timer"
    job_id = schedule.create_timer(seconds, label)
    return f"{label.capitalize()} timer #{job_id} set for {seconds} seconds.", None


def alarm_time(text):
    # Require AM/PM or an unambiguous 24-hour time; bare "seven" asks a question.
    match = re.search(rf"\b({NUMBER}|[01]?\d|2[0-3])(?::([0-5]\d))?\s*(am|pm)\b", text, re.IGNORECASE)
    if match:
        hour = number(match.group(1))
        minute = int(match.group(2) or 0)
        if hour is None or not 1 <= hour <= 12:
            return None
        return (hour % 12 + (12 if match.group(3).lower() == "pm" else 0), minute)
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def handle_alarm(plan, prompt, schedule: Schedule):
    op = plan.get("operation", "create")
    if op == "list":
        jobs = schedule.list_jobs("alarm")
        if not jobs:
            return "You have no active alarms.", None
        return "Active alarms: " + "; ".join(
            f"#{job['id']} at {datetime.fromtimestamp(job['due_at'], schedule.zone).strftime('%a %b %d, %I:%M %p')}"
            + (f", {job['recurrence']}" if job["recurrence"] != "once" else "") for job in jobs
        ) + ".", None
    if op == "cancel":
        if re.search(r"\ball\b", prompt, re.IGNORECASE):
            count = schedule.cancel_all("alarm")
            return f"Cancelled {count} alarms." if count else "You have no active alarms.", None
        count, job = schedule.cancel("alarm", identifier(prompt, "alarm"))
        if job:
            return f"Cancelled alarm #{job['id']}.", None
        if not count:
            return "I couldn't find that alarm.", None
        return "Which alarm number should I cancel?", {"intent": "alarm", "pending": True, "operation": "cancel"}
    when = alarm_time(prompt)
    if when is None:
        return "What time should I set the alarm for? Please include AM or PM.", {"intent": "alarm", "pending": True, "operation": "create"}
    recurrence = "weekdays" if re.search(r"\b(?:weekday|weekdays|every weekday)\b", prompt, re.IGNORECASE) else "daily" if re.search(r"\b(?:daily|every day)\b", prompt, re.IGNORECASE) else "once"
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|next month|january|february|march|april|may|june|july|august|september|october|november|december)\b", prompt, re.IGNORECASE):
        return "I can set an alarm for its next occurrence, tomorrow, every day, or weekdays. Please say one of those.", None
    tomorrow = bool(re.search(r"\btomorrow\b", prompt, re.IGNORECASE))
    if recurrence == "once" and not tomorrow and re.search(r"\b(?:today|tonight)\b", prompt, re.IGNORECASE):
        now = datetime.fromtimestamp(schedule.clock(), schedule.zone)
        if when <= (now.hour, now.minute):
            return "That time has passed today. Please say tomorrow or choose a later time.", None
    job_id, due = schedule.create_alarm(*when, recurrence=recurrence, tomorrow=tomorrow)
    return f"Alarm #{job_id} set for {due.strftime('%A, %B %d at %I:%M %p')}" + (f", {recurrence}." if recurrence != "once" else "."), None
