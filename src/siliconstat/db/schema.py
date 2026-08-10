"""Relational schema for stored Monte Carlo runs.

Portability note
----------------
The DDL below is deliberately plain SQL: ``INTEGER PRIMARY KEY`` surrogate
keys, ``TEXT`` for JSON payloads, explicit foreign keys, no SQLite-only types
or expressions.  Moving to PostgreSQL requires swapping the connection factory
and the ``AUTOINCREMENT``/``SERIAL`` keyword -- which is why every statement is
generated through :func:`schema_statements` with a dialect argument rather than
hard-coded.

Storage policy: **per-sample records are stored, not just aggregates.**  A run
you cannot re-open sample by sample is a run you cannot debug, so ``samples``,
``sample_measurements`` and ``sample_variations`` hold the full raw data and
the aggregate tables are caches derived from them.
"""

from __future__ import annotations

__all__ = ["schema_statements", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1


def _pk(dialect: str) -> str:
    return ("INTEGER PRIMARY KEY AUTOINCREMENT" if dialect == "sqlite"
            else "BIGSERIAL PRIMARY KEY")


def schema_statements(dialect: str = "sqlite") -> list[str]:
    """DDL statements creating the full schema for *dialect*."""
    pk = _pk(dialect)
    return [
        f"""
        CREATE TABLE IF NOT EXISTS schema_info (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS projects (
            id          {pk},
            name        TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at  TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS circuits (
            id          {pk},
            project_id  INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            name        TEXT NOT NULL,
            netlist     TEXT NOT NULL,
            sha256      TEXT NOT NULL,
            summary     TEXT NOT NULL DEFAULT '{{}}',
            created_at  TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_circuits_sha ON circuits(sha256)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS runs (
            id                {pk},
            run_id            TEXT NOT NULL UNIQUE,
            project_id        INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            circuit_id        INTEGER REFERENCES circuits(id) ON DELETE SET NULL,
            circuit_name      TEXT NOT NULL,
            circuit_sha256    TEXT NOT NULL DEFAULT '',
            label             TEXT NOT NULL DEFAULT '',
            config_json       TEXT NOT NULL,
            counters_json     TEXT NOT NULL,
            nominal_json      TEXT NOT NULL DEFAULT '{{}}',
            nominal_status    TEXT NOT NULL DEFAULT '',
            measurement_meta  TEXT NOT NULL DEFAULT '[]',
            slot_meta         TEXT NOT NULL DEFAULT '[]',
            spec_meta         TEXT NOT NULL DEFAULT '[]',
            samples           INTEGER NOT NULL DEFAULT 0,
            seed              INTEGER NOT NULL DEFAULT 0,
            variation_mode    TEXT NOT NULL DEFAULT '',
            pvt_label         TEXT NOT NULL DEFAULT '',
            started_at        TEXT NOT NULL DEFAULT '',
            finished_at       TEXT NOT NULL DEFAULT '',
            duration_s        REAL NOT NULL DEFAULT 0,
            software_version  TEXT NOT NULL DEFAULT '',
            platform_info     TEXT NOT NULL DEFAULT '',
            notes             TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_runs_circuit ON runs(circuit_name)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS samples (
            id             {pk},
            run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            sample_index   INTEGER NOT NULL,
            seed           TEXT NOT NULL,
            status         TEXT NOT NULL,
            failure_reason TEXT NOT NULL DEFAULT '',
            passed         INTEGER,
            dc_iterations  INTEGER NOT NULL DEFAULT 0,
            dc_strategy    TEXT NOT NULL DEFAULT '',
            runtime_s      REAL NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_samples_run ON samples(run_id, sample_index)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS sample_measurements (
            id        {pk},
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            name      TEXT NOT NULL,
            value     REAL,
            ok        INTEGER NOT NULL DEFAULT 1,
            reason    TEXT NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_meas_run_name
            ON sample_measurements(run_id, name)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS sample_variations (
            id        {pk},
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            slot      TEXT NOT NULL,
            value     REAL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_var_run_slot
            ON sample_variations(run_id, slot)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS sample_devices (
            id        {pk},
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            key       TEXT NOT NULL,
            value     REAL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS sample_specs (
            id        {pk},
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            spec_key  TEXT NOT NULL,
            passed    INTEGER NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS run_statistics (
            id          {pk},
            run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            measurement TEXT NOT NULL,
            stats_json  TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_stats_run ON run_statistics(run_id)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS run_yield (
            id           {pk},
            run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            yield_json   TEXT NOT NULL
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS run_analysis (
            id            {pk},
            run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            analysis_json TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
        """,
    ]
