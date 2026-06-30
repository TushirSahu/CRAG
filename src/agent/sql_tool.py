"""Read-only text-to-SQL analytics agent.

Replaces the old hardcoded ``run_sql`` mock. The router sends analytics/count
questions here; this turns the question into a single ``SELECT`` against a real
SQLite database and returns the rows as a context document the rest of the graph
(generate → verify) consumes unchanged.

Safety is enforced in two independent layers so a bad LLM query can never mutate
data:

1. The connection is opened **read-only** (SQLite ``mode=ro`` URI).
2. Generated SQL is validated to be a *single* ``SELECT`` before it is run.

Schema introspection and execution are deliberately separable from the LLM step
so they can be unit-tested without API keys.
"""

from __future__ import annotations

import datetime as _dt
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.agent import prompts
from src.core.config import Settings, get_settings
from src.utils.logger import logger

# Statements we refuse outright, even if smuggled past the single-SELECT check.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|"
    r"detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


class GeneratedSQL(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT statement.")


def is_safe_select(sql: str) -> bool:
    """True only for a single ``SELECT`` (or ``WITH ... SELECT``) statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped or _FORBIDDEN.search(stripped):
        return False
    if ";" in stripped:  # no statement stacking
        return False
    head = stripped.lstrip("(").lower()
    return head.startswith("select") or head.startswith("with")


class SQLTool:
    """Schema-aware, write-guarded SQL helper over one SQLite file."""

    def __init__(self, llm=None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.llm = llm
        self.db_path = self.settings.resolve(self.settings.sql.db_path)
        self.max_rows = self.settings.sql.max_rows

    # -- connection -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Open the database read-only so no query can ever mutate it."""
        uri = f"file:{Path(self.db_path).as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    @property
    def available(self) -> bool:
        return Path(self.db_path).exists()

    # -- introspection ----------------------------------------------------
    def schema(self) -> str:
        """Return a compact ``table(col, col, ...)`` description of every table."""
        with self._connect() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            lines = []
            for table in tables:
                cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{table}')")]
                lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)

    # -- execution --------------------------------------------------------
    def execute(self, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        """Run a validated SELECT and return ``(columns, rows)``. Raises on unsafe SQL."""
        if not is_safe_select(sql):
            raise ValueError(f"Refused non-read-only SQL: {sql!r}")
        with self._connect() as conn:
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(self.max_rows)
        return columns, rows

    # -- end to end -------------------------------------------------------
    def query(self, question: str) -> Dict[str, Any]:
        """Generate SQL for ``question``, run it, and format rows as a context doc."""
        sql = self._generate_sql(question)
        columns, rows = self.execute(sql)
        return {
            "content": self._format(sql, columns, rows),
            "metadata": {"source": "SQL Database", "sql": sql},
        }

    def _generate_sql(self, question: str) -> str:
        if self.llm is None:
            raise RuntimeError("SQLTool needs an llm to generate SQL.")
        parser = JsonOutputParser(pydantic_object=GeneratedSQL)
        result = (prompts.TEXT_TO_SQL | self.llm | parser).invoke(
            {
                "schema": self.schema(),
                "question": question,
                "today": _dt.date.today().isoformat(),
                "format_instructions": parser.get_format_instructions(),
            }
        )
        sql = result["sql"] if isinstance(result, dict) else str(result)
        logger.info("text-to-SQL: %s", sql)
        return sql

    @staticmethod
    def _format(sql: str, columns: List[str], rows: List[Tuple[Any, ...]]) -> str:
        if not rows:
            return f"Query: {sql}\nResult: no rows."
        header = " | ".join(columns)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"Query: {sql}\nColumns: {header}\nRows:\n{body}"
