"""SQLite database layer for WESAD stress detection app.

Replaces per-file JSON storage for history, results, and comparisons
with a single ``wesad.db`` SQLite database.

Usage::

    from database import get_db, init_db

    init_db()           # create tables (idempotent)
    db = get_db()       # get a thread-local connection
    db.save_history_entry(tracking_id, entry_dict)
"""

import json
import logging
import os
import sqlite3
import threading

logger = logging.getLogger(__name__)

_DB_DIR = os.path.dirname(__file__)
_DB_NAME = "wesad.db"

# Thread-local storage for connections (SQLite objects are not thread-safe).
_local = threading.local()

# Allow overriding the DB path for tests.
_db_path_override: str | None = None


def set_db_path(path: str | None) -> None:
    """Override the default database path (useful for tests)."""
    global _db_path_override
    _db_path_override = path


def _db_path() -> str:
    if _db_path_override is not None:
        return _db_path_override
    return os.path.join(_DB_DIR, _DB_NAME)


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode enabled."""
    conn = getattr(_local, "conn", None)
    path = _db_path()
    # Reconnect if the path changed (e.g. during tests) or conn is closed.
    if conn is not None:
        try:
            if getattr(_local, "conn_path", None) != path:
                conn.close()
                conn = None
            else:
                conn.execute("SELECT 1")
        except Exception:
            conn = None

    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        _local.conn_path = path
    return conn


def close_connection() -> None:
    """Close the thread-local connection (if any)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
        _local.conn_path = None


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_id   TEXT    NOT NULL,
    captured_at   TEXT,
    subject_id    TEXT,
    method        TEXT,
    is_manual     INTEGER NOT NULL DEFAULT 0,
    stress_ratio  REAL    NOT NULL DEFAULT 0.0,
    stress_level  TEXT,
    overall_stress INTEGER NOT NULL DEFAULT 0,
    predicted_label TEXT,
    total_windows INTEGER NOT NULL DEFAULT 0,
    label_counts  TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_history_tracking
    ON history (tracking_id, captured_at);

CREATE TABLE IF NOT EXISTS results (
    subject_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comparisons (
    subject_id TEXT PRIMARY KEY,
    data       TEXT NOT NULL
);
"""


def init_db() -> None:
    """Create tables and indices if they don't exist yet."""
    conn = get_connection()
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


# ------------------------------------------------------------------
# History CRUD
# ------------------------------------------------------------------

def load_history(tracking_id: str) -> list[dict]:
    """Return all history entries for *tracking_id* ordered by captured_at."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history WHERE tracking_id = ? ORDER BY captured_at",
        (tracking_id,),
    ).fetchall()
    return [_row_to_history_dict(r) for r in rows]


def append_history_entry(tracking_id: str, entry: dict) -> None:
    """Insert a single history entry."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO history
           (tracking_id, captured_at, subject_id, method, is_manual,
            stress_ratio, stress_level, overall_stress, predicted_label,
            total_windows, label_counts)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            tracking_id,
            entry.get("captured_at"),
            entry.get("subject_id"),
            entry.get("method"),
            int(bool(entry.get("is_manual", False))),
            float(entry.get("stress_ratio", 0.0)),
            entry.get("stress_level"),
            int(bool(entry.get("overall_stress", False))),
            entry.get("predicted_label"),
            int(entry.get("total_windows", 0)),
            json.dumps(entry.get("label_counts", {}), ensure_ascii=False),
        ),
    )
    conn.commit()

    # Enforce a cap of 500 entries per tracking_id
    _prune_history(tracking_id, max_entries=500)


def _prune_history(tracking_id: str, max_entries: int = 500) -> None:
    """Delete oldest entries if the count exceeds *max_entries*."""
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE tracking_id = ?",
        (tracking_id,),
    ).fetchone()[0]
    if count > max_entries:
        conn.execute(
            """DELETE FROM history WHERE id IN (
                   SELECT id FROM history
                   WHERE tracking_id = ?
                   ORDER BY captured_at
                   LIMIT ?
               )""",
            (tracking_id, count - max_entries),
        )
        conn.commit()


def filter_history(tracking_id: str,
                   start: str | None = None,
                   end: str | None = None) -> list[dict]:
    """Load history with optional date-range filter (ISO-8601 strings)."""
    clauses = ["tracking_id = ?"]
    params: list = [tracking_id]

    if start:
        clauses.append("captured_at >= ?")
        params.append(start)
    if end:
        clauses.append("captured_at <= ?")
        params.append(end)

    sql = (
        "SELECT * FROM history WHERE "
        + " AND ".join(clauses)
        + " ORDER BY captured_at"
    )
    conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_history_dict(r) for r in rows]


def _row_to_history_dict(row: sqlite3.Row) -> dict:
    """Convert a DB row to the dict format the app expects."""
    return {
        "tracking_id": row["tracking_id"],
        "captured_at": row["captured_at"],
        "subject_id": row["subject_id"],
        "method": row["method"],
        "is_manual": bool(row["is_manual"]),
        "stress_ratio": row["stress_ratio"],
        "stress_level": row["stress_level"],
        "overall_stress": bool(row["overall_stress"]),
        "predicted_label": row["predicted_label"],
        "total_windows": row["total_windows"],
        "label_counts": json.loads(row["label_counts"]) if row["label_counts"] else {},
    }


# ------------------------------------------------------------------
# Results CRUD
# ------------------------------------------------------------------

def save_result(subject_id: str, result: dict) -> None:
    """Upsert an analysis result (stored as JSON blob)."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO results (subject_id, data) VALUES (?, ?)",
        (subject_id, json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()


def load_result(subject_id: str) -> dict | None:
    """Load a saved result. Returns None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT data FROM results WHERE subject_id = ?",
        (subject_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["data"])


def list_result_ids() -> list[str]:
    """Return sorted list of subject IDs that have saved results."""
    import re
    conn = get_connection()
    rows = conn.execute("SELECT subject_id FROM results").fetchall()
    sids = [r["subject_id"] for r in rows]
    return sorted(
        sids,
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)],
    )


def delete_result(subject_id: str) -> None:
    """Delete a saved result."""
    conn = get_connection()
    conn.execute("DELETE FROM results WHERE subject_id = ?", (subject_id,))
    conn.commit()


# ------------------------------------------------------------------
# Comparisons CRUD
# ------------------------------------------------------------------

def save_comparison(subject_id: str, results: list) -> None:
    """Upsert a comparison result (stored as JSON blob)."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO comparisons (subject_id, data) VALUES (?, ?)",
        (subject_id, json.dumps(results, ensure_ascii=False)),
    )
    conn.commit()


def load_comparison(subject_id: str) -> list | None:
    """Load a saved comparison. Returns None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT data FROM comparisons WHERE subject_id = ?",
        (subject_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["data"])
