"""Seed a small SQLite analytics DB so the SQL agent answers real questions.

Creates ``data/analytics.db`` (path from ``configs/config.yaml``) with two tiny,
realistic tables and deterministic rows anchored to *today*, so questions like
"How many users signed up yesterday?" return a true, reproducible number.

Run once:  ``python scripts/seed_analytics_db.py``
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from src.core.config import get_settings

# Deterministic, relative to today so "yesterday" always has data.
_TODAY = dt.date.today()
_SIGNUPS = {
    _TODAY - dt.timedelta(days=2): 31,
    _TODAY - dt.timedelta(days=1): 47,  # "yesterday"
    _TODAY: 12,
}
_EVENTS = {"login": 312, "purchase": 28, "export": 64}


def seed(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(db_path).unlink(missing_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE signups (
                id INTEGER PRIMARY KEY,
                user_email TEXT NOT NULL,
                signup_date TEXT NOT NULL   -- ISO date (YYYY-MM-DD)
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL     -- ISO datetime
            );
            """
        )
        uid = 0
        for day, count in _SIGNUPS.items():
            for _ in range(count):
                uid += 1
                conn.execute(
                    "INSERT INTO signups (user_email, signup_date) VALUES (?, ?)",
                    (f"user{uid}@example.com", day.isoformat()),
                )
        eid = 0
        for event_type, count in _EVENTS.items():
            for _ in range(count):
                eid += 1
                conn.execute(
                    "INSERT INTO events (user_id, event_type, created_at) VALUES (?, ?, ?)",
                    (eid % uid + 1, event_type, dt.datetime.now().isoformat(timespec="seconds")),
                )
        conn.commit()

    total = sum(_SIGNUPS.values())
    print(f"✅ Seeded {db_path}: {total} signups, {sum(_EVENTS.values())} events.")


if __name__ == "__main__":
    seed(get_settings().resolve(get_settings().sql.db_path))
