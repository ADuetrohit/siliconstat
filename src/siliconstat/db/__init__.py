"""Persistence layer (SQLite today, PostgreSQL-ready)."""

from __future__ import annotations

from .schema import SCHEMA_VERSION, schema_statements
from .store import DatabaseError, RunStore, StoredRunInfo

__all__ = ["RunStore", "StoredRunInfo", "DatabaseError", "schema_statements",
           "SCHEMA_VERSION"]
