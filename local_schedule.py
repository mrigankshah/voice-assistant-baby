"""Persistent local timers and alarms using SQLite and the Pi's local timezone."""

from datetime import datetime, timedelta
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DB_PATH = Path(__file__).with_name("assistant_schedule.sqlite3")


def local_zone():
    name = os.environ.get("TZ")
    if not name and Path("/etc/timezone").exists():
        name = Path("/etc/timezone").read_text(encoding="utf-8").strip()
    if not name and Path("/etc/localtime").exists():
        resolved = Path("/etc/localtime").resolve().as_posix()
        if "/zoneinfo/" in resolved:
            name = resolved.split("/zoneinfo/", 1)[1]
    if name:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            pass
    return datetime.now().astimezone().tzinfo


def next_alarm(hour: int, minute: int, recurrence: str, now: datetime, *, tomorrow=False) -> datetime:
    if recurrence not in {"once", "daily", "weekdays"} or not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError("Invalid alarm time or recurrence")
    day = now.date() + timedelta(days=1 if tomorrow else 0)
    for _ in range(8):
        candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=now.tzinfo)
        if candidate > now and (recurrence != "weekdays" or candidate.weekday() < 5):
            return candidate
        day += timedelta(days=1)
    raise ValueError("Could not find the next alarm occurrence")


class Schedule:
    def __init__(self, path=DB_PATH, *, clock=time.time, zone=None):
        self.path, self.clock, self.zone = Path(path), clock, zone or local_zone()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL,
                    due_at REAL NOT NULL, hour INTEGER, minute INTEGER,
                    recurrence TEXT NOT NULL DEFAULT 'once', state TEXT NOT NULL DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL,
                    message TEXT NOT NULL, created_at REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0
                );
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_timer(self, seconds: int, label="timer"):
        if not 1 <= seconds <= 7 * 86400:
            raise ValueError("Timer must be between one second and seven days")
        due = self.clock() + seconds
        with self._connect() as db:
            cursor = db.execute("INSERT INTO jobs(kind,label,due_at) VALUES ('timer',?,?)", (label, due))
            return cursor.lastrowid

    def create_alarm(self, hour: int, minute: int, *, recurrence="once", tomorrow=False, label="alarm"):
        now = datetime.fromtimestamp(self.clock(), self.zone)
        due = next_alarm(hour, minute, recurrence, now, tomorrow=tomorrow)
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO jobs(kind,label,due_at,hour,minute,recurrence) VALUES ('alarm',?,?,?,?,?)",
                (label, due.timestamp(), hour, minute, recurrence),
            )
            return cursor.lastrowid, due

    def list_jobs(self, kind):
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM jobs WHERE kind=? AND state='active' ORDER BY due_at,id", (kind,)
            )]

    def cancel(self, kind, identifier=None):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            jobs = self.list_jobs(kind)
            if identifier is not None:
                key = str(identifier).strip().casefold()
                jobs = [job for job in jobs if str(job["id"]) == key or job["label"].casefold() == key]
            if len(jobs) != 1:
                return len(jobs), None
            job = jobs[0]
            db.execute("UPDATE jobs SET state='cancelled' WHERE id=? AND state='active'", (job["id"],))
            return 1, job

    def cancel_all(self, kind):
        with self._connect() as db:
            cursor = db.execute("UPDATE jobs SET state='cancelled' WHERE kind=? AND state='active'", (kind,))
            return cursor.rowcount

    def fire_due(self):
        now = self.clock()
        fired = []
        with self._connect() as db:
            if db.execute("SELECT 1 FROM jobs WHERE state='active' AND due_at<=? LIMIT 1", (now,)).fetchone() is None:
                return fired
            db.execute("BEGIN IMMEDIATE")
            jobs = [dict(row) for row in db.execute(
                "SELECT * FROM jobs WHERE state='active' AND due_at<=? ORDER BY due_at,id", (now,)
            )]
            for job in jobs:
                message = f"{job['label'].capitalize()} #{job['id']} finished." if job["kind"] == "timer" else f"Alarm #{job['id']} is ringing."
                db.execute("INSERT INTO notifications(job_id,message,created_at) VALUES (?,?,?)", (job["id"], message, now))
                if job["recurrence"] == "once":
                    db.execute("UPDATE jobs SET state='fired' WHERE id=?", (job["id"],))
                else:
                    due = next_alarm(job["hour"], job["minute"], job["recurrence"],
                                     datetime.fromtimestamp(now, self.zone)).timestamp()
                    db.execute("UPDATE jobs SET due_at=? WHERE id=?", (due, job["id"]))
                fired.append(message)
        return fired

    def take_notifications(self):
        with self._connect() as db:
            if db.execute("SELECT 1 FROM notifications WHERE consumed=0 LIMIT 1").fetchone() is None:
                return []
            db.execute("BEGIN IMMEDIATE")
            rows = [dict(row) for row in db.execute("SELECT * FROM notifications WHERE consumed=0 ORDER BY id")]
            if rows:
                db.executemany("UPDATE notifications SET consumed=1 WHERE id=?", [(row["id"],) for row in rows])
            return [row["message"] for row in rows]
