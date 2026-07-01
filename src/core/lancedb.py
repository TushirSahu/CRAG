"""Single place to open a LanceDB connection.

Both the knowledge store and the semantic cache live in embedded LanceDB
databases; this helper keeps the ``connect`` + path-resolution logic in one spot.
"""

from __future__ import annotations

import lancedb

from src.core.config import get_settings


def connect(path: str):
    """Open (creating if needed) the LanceDB database at a config-relative ``path``."""
    return lancedb.connect(get_settings().resolve(path))
