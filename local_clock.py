"""Read the Raspberry Pi's local system clock for assistant tool calls."""

from __future__ import annotations

from datetime import datetime


def get_current_datetime() -> dict[str, str]:
    now = datetime.now().astimezone()
    return {
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": now.tzname() or "local",
        "utc_offset": now.strftime("%z"),
        "iso_datetime": now.isoformat(timespec="seconds"),
    }
