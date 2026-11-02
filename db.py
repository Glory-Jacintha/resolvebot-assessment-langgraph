"""Small SQLite helpers."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config import get_settings


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open the database; commit on success, roll back on error."""
    conn = sqlite3.connect(get_settings().db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_policy_parameters(conn: sqlite3.Connection) -> dict[str, str]:
    """All business numbers (return_window_days, warranty months, ...)."""
    rows = conn.execute("SELECT key, value FROM policy_parameters").fetchall()
    return {row["key"]: row["value"] for row in rows}


def customer_exists(customer_id: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    return row is not None
