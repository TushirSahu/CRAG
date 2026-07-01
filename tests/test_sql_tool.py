"""Tests for the read-only text-to-SQL agent — no API keys needed.

Exercises the safety guard and execution against a seeded temp SQLite DB; the
LLM-dependent ``query`` path is covered by the manual graph run.
"""

import sqlite3

import pytest

from src.agent.sql_tool import SQLTool, is_safe_select
from src.core.config import get_settings


def test_is_safe_select_accepts_reads():
    assert is_safe_select("SELECT * FROM signups")
    assert is_safe_select("  with t as (select 1) select * from t  ")


def test_is_safe_select_rejects_writes_and_stacking():
    for bad in [
        "DELETE FROM signups",
        "DROP TABLE signups",
        "INSERT INTO signups VALUES (1)",
        "UPDATE signups SET id = 0",
        "SELECT 1; DROP TABLE signups",
        "PRAGMA table_info('signups')",
    ]:
        assert not is_safe_select(bad), bad


def _tool(tmp_path):
    db = tmp_path / "analytics.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE signups (id INTEGER, signup_date TEXT)")
        conn.executemany(
            "INSERT INTO signups VALUES (?, ?)",
            [(1, "2024-01-01"), (2, "2024-01-01"), (3, "2024-01-02")],
        )
        conn.commit()
    settings = get_settings().model_copy(deep=True)
    settings.sql.db_path = str(db)
    return SQLTool(llm=None, settings=settings)


def test_execute_returns_real_rows(tmp_path):
    tool = _tool(tmp_path)
    assert tool.available
    cols, rows = tool.execute("SELECT COUNT(*) AS n FROM signups")
    assert cols == ["n"]
    assert rows[0][0] == 3


def test_schema_introspection(tmp_path):
    tool = _tool(tmp_path)
    assert tool.schema().startswith("signups(")


def test_execute_rejects_writes(tmp_path):
    tool = _tool(tmp_path)
    with pytest.raises(ValueError):
        tool.execute("DELETE FROM signups")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_is_safe_select_accepts_reads()
    test_is_safe_select_rejects_writes_and_stacking()
    with tempfile.TemporaryDirectory() as d:
        test_execute_returns_real_rows(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_schema_introspection(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_execute_rejects_writes(Path(d))
    print("✅ sql_tool tests passed")
