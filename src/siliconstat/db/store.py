"""Persistence for circuits, Monte Carlo runs and their analyses.

SQLite by default (zero-configuration, single file, good enough for millions of
sample rows).  The backend is isolated behind :class:`RunStore` so a PostgreSQL
backend only has to supply a DB-API connection and a parameter style; nothing
above this module knows which database it is talking to.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

from ..core.circuit import Circuit
from ..core.exceptions import SiliconStatError
from ..mc.results import MonteCarloRun, RunCounters, SampleResult
from .schema import SCHEMA_VERSION, schema_statements

__all__ = ["RunStore", "StoredRunInfo", "DatabaseError"]


class DatabaseError(SiliconStatError):
    """Raised for storage-layer failures."""


@dataclass
class StoredRunInfo:
    """Row-level summary used by run listings."""

    run_id: str
    circuit_name: str
    label: str
    samples: int
    seed: int
    variation_mode: str
    pvt_label: str
    started_at: str
    finished_at: str
    duration_s: float
    counters: dict[str, Any]
    software_version: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "circuit_name": self.circuit_name,
            "label": self.label, "samples": self.samples, "seed": self.seed,
            "variation_mode": self.variation_mode, "pvt_label": self.pvt_label,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_s": self.duration_s, "counters": self.counters,
            "software_version": self.software_version, "notes": self.notes,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=True)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class RunStore:
    """Storage front end.

    Thread-safe for the access pattern the API uses (short transactions from
    request handlers) via a per-instance lock plus SQLite's own WAL mode.
    """

    def __init__(self, path: str = "siliconstat.sqlite") -> None:
        self.path = path
        self._lock = threading.RLock()
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_schema()

    # -- connection --------------------------------------------------------
    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(f"database operation failed: {exc}") from exc
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self.connect() as conn:
            for statement in schema_statements("sqlite"):
                conn.execute(statement)
            conn.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES (?, ?)",
                ("version", str(SCHEMA_VERSION)))

    # -- projects ----------------------------------------------------------
    def ensure_project(self, name: str, description: str = "") -> int:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT id FROM projects WHERE name = ?",
                               (name,)).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO projects(name, description, created_at) VALUES (?,?,?)",
                (name, description, _now()))
            return int(cur.lastrowid)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, created_at FROM projects "
                "ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    # -- circuits ----------------------------------------------------------
    def save_circuit(self, circuit: Circuit, *, project: str = "default",
                     netlist_text: str | None = None) -> int:
        text = netlist_text if netlist_text is not None else (circuit.source_text or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        project_id = self.ensure_project(project)
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM circuits WHERE sha256 = ? AND name = ?",
                (digest, circuit.name)).fetchone()
            if row:
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO circuits(project_id, name, netlist, sha256, summary, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (project_id, circuit.name, text, digest,
                 _json(circuit.describe()), _now()))
            return int(cur.lastrowid)

    def list_circuits(self, project: str | None = None) -> list[dict[str, Any]]:
        query = ("SELECT c.id, c.name, c.sha256, c.created_at, p.name AS project, "
                 "LENGTH(c.netlist) AS netlist_bytes FROM circuits c "
                 "LEFT JOIN projects p ON p.id = c.project_id")
        params: tuple[Any, ...] = ()
        if project:
            query += " WHERE p.name = ?"
            params = (project,)
        query += " ORDER BY c.created_at DESC"
        with self._lock, self.connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def get_circuit(self, circuit_id: int) -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT * FROM circuits WHERE id = ?",
                               (circuit_id,)).fetchone()
            if row is None:
                raise DatabaseError(f"no circuit with id {circuit_id}")
            data = dict(row)
            data["summary"] = json.loads(data.get("summary") or "{}")
            return data

    def get_circuit_by_name(self, name: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM circuits WHERE name = ? ORDER BY created_at DESC "
                "LIMIT 1", (name,)).fetchone()
            if row is None:
                return None
            data = dict(row)
            data["summary"] = json.loads(data.get("summary") or "{}")
            return data

    # -- runs --------------------------------------------------------------
    def save_run(self, run: MonteCarloRun, *, project: str = "default",
                 circuit: Circuit | None = None, label: str = "") -> int:
        circuit_id = None
        if circuit is not None:
            circuit_id = self.save_circuit(circuit, project=project)
        project_id = self.ensure_project(project)
        config = run.config or {}
        pvt = config.get("pvt") or {}

        with self._lock, self.connect() as conn:
            existing = conn.execute("SELECT id FROM runs WHERE run_id = ?",
                                    (run.run_id,)).fetchone()
            if existing:
                raise DatabaseError(
                    f"run {run.run_id!r} is already stored; delete it first or "
                    "use a new run id")
            cur = conn.execute(
                """
                INSERT INTO runs(run_id, project_id, circuit_id, circuit_name,
                    circuit_sha256, label, config_json, counters_json,
                    nominal_json, nominal_status, measurement_meta, slot_meta,
                    spec_meta, samples, seed, variation_mode, pvt_label,
                    started_at, finished_at, duration_s, software_version,
                    platform_info, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (run.run_id, project_id, circuit_id, run.circuit_name,
                 run.circuit_sha256, label or config.get("label", ""),
                 _json(config), _json(run.counters.to_dict()),
                 _json(run.nominal), run.nominal_status,
                 _json(run.measurement_meta), _json(run.slot_meta),
                 _json(run.spec_meta), int(config.get("samples", len(run.samples))),
                 int(config.get("seed", 0)),
                 str(config.get("variation", {}).get("mode", "")),
                 str(pvt.get("label", "")), run.started_at, run.finished_at,
                 float(run.duration_s), run.software_version, run.platform_info,
                 run.notes))
            run_pk = int(cur.lastrowid)

            sample_rows = [
                (run_pk, s.index, str(s.seed), s.status, s.failure_reason,
                 None if s.passed is None else int(s.passed),
                 s.dc_iterations, s.dc_strategy, s.runtime_s)
                for s in run.samples
            ]
            conn.executemany(
                "INSERT INTO samples(run_id, sample_index, seed, status, "
                "failure_reason, passed, dc_iterations, dc_strategy, runtime_s) "
                "VALUES (?,?,?,?,?,?,?,?,?)", sample_rows)

            id_rows = conn.execute(
                "SELECT id, sample_index FROM samples WHERE run_id = ?",
                (run_pk,)).fetchall()
            sample_ids = {int(r["sample_index"]): int(r["id"]) for r in id_rows}

            measurements: list[tuple[Any, ...]] = []
            variations: list[tuple[Any, ...]] = []
            devices: list[tuple[Any, ...]] = []
            specs: list[tuple[Any, ...]] = []
            for s in run.samples:
                sid = sample_ids[s.index]
                for name, value in s.measurements.items():
                    measurements.append((
                        sid, run_pk, name, _finite(value),
                        int(s.measurement_ok.get(name, True)),
                        s.measurement_reasons.get(name, "")))
                for slot, value in s.slot_values.items():
                    variations.append((sid, run_pk, slot, _finite(value)))
                for key, value in s.device_values.items():
                    devices.append((sid, run_pk, key, _finite(value)))
                for key, ok in s.spec_pass.items():
                    specs.append((sid, run_pk, key, int(ok)))

            conn.executemany(
                "INSERT INTO sample_measurements(sample_id, run_id, name, value, "
                "ok, reason) VALUES (?,?,?,?,?,?)", measurements)
            conn.executemany(
                "INSERT INTO sample_variations(sample_id, run_id, slot, value) "
                "VALUES (?,?,?,?)", variations)
            conn.executemany(
                "INSERT INTO sample_devices(sample_id, run_id, key, value) "
                "VALUES (?,?,?,?)", devices)
            conn.executemany(
                "INSERT INTO sample_specs(sample_id, run_id, spec_key, passed) "
                "VALUES (?,?,?,?)", specs)
            return run_pk

    def save_analysis(self, run_id: str, analysis: dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT id FROM runs WHERE run_id = ?",
                               (run_id,)).fetchone()
            if row is None:
                raise DatabaseError(f"no stored run with id {run_id!r}")
            run_pk = int(row["id"])
            conn.execute("DELETE FROM run_analysis WHERE run_id = ?", (run_pk,))
            conn.execute(
                "INSERT INTO run_analysis(run_id, analysis_json, created_at) "
                "VALUES (?,?,?)", (run_pk, _json(analysis), _now()))
            conn.execute("DELETE FROM run_statistics WHERE run_id = ?", (run_pk,))
            conn.executemany(
                "INSERT INTO run_statistics(run_id, measurement, stats_json) "
                "VALUES (?,?,?)",
                [(run_pk, name, _json(stats))
                 for name, stats in (analysis.get("statistics") or {}).items()])
            conn.execute("DELETE FROM run_yield WHERE run_id = ?", (run_pk,))
            conn.execute("INSERT INTO run_yield(run_id, yield_json) VALUES (?,?)",
                         (run_pk, _json(analysis.get("yield") or {})))

    def load_analysis(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT a.analysis_json FROM run_analysis a "
                "JOIN runs r ON r.id = a.run_id WHERE r.run_id = ?",
                (run_id,)).fetchone()
            return json.loads(row["analysis_json"]) if row else None

    def list_runs(self, *, project: str | None = None,
                  circuit_name: str | None = None,
                  limit: int = 100) -> list[StoredRunInfo]:
        query = ("SELECT r.* FROM runs r LEFT JOIN projects p "
                 "ON p.id = r.project_id WHERE 1=1")
        params: list[Any] = []
        if project:
            query += " AND p.name = ?"
            params.append(project)
        if circuit_name:
            query += " AND r.circuit_name = ?"
            params.append(circuit_name)
        query += " ORDER BY r.started_at DESC, r.id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            StoredRunInfo(
                run_id=r["run_id"], circuit_name=r["circuit_name"],
                label=r["label"], samples=int(r["samples"]), seed=int(r["seed"]),
                variation_mode=r["variation_mode"], pvt_label=r["pvt_label"],
                started_at=r["started_at"], finished_at=r["finished_at"],
                duration_s=float(r["duration_s"]),
                counters=json.loads(r["counters_json"]),
                software_version=r["software_version"], notes=r["notes"],
            ) for r in rows
        ]

    def load_run(self, run_id: str) -> MonteCarloRun:
        """Rebuild a complete :class:`MonteCarloRun` from storage."""
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?",
                               (run_id,)).fetchone()
            if row is None:
                raise DatabaseError(f"no stored run with id {run_id!r}")
            run_pk = int(row["id"])
            run = MonteCarloRun(
                run_id=row["run_id"], circuit_name=row["circuit_name"],
                config=json.loads(row["config_json"]),
                measurement_meta=json.loads(row["measurement_meta"]),
                slot_meta=json.loads(row["slot_meta"]),
                spec_meta=json.loads(row["spec_meta"]),
                nominal=json.loads(row["nominal_json"]),
                nominal_status=row["nominal_status"],
                started_at=row["started_at"], finished_at=row["finished_at"],
                duration_s=float(row["duration_s"]),
                software_version=row["software_version"],
                platform_info=row["platform_info"],
                circuit_sha256=row["circuit_sha256"], notes=row["notes"],
            )
            counters_data = json.loads(row["counters_json"])
            run.counters = RunCounters(**{
                k: int(v) for k, v in counters_data.items()
                if k in RunCounters.__dataclass_fields__})  # type: ignore[attr-defined]

            samples: dict[int, SampleResult] = {}
            for r in conn.execute(
                    "SELECT * FROM samples WHERE run_id = ? ORDER BY sample_index",
                    (run_pk,)):
                sid = int(r["id"])
                samples[sid] = SampleResult(
                    index=int(r["sample_index"]), seed=int(r["seed"]),
                    status=r["status"], failure_reason=r["failure_reason"],
                    passed=None if r["passed"] is None else bool(r["passed"]),
                    dc_iterations=int(r["dc_iterations"]),
                    dc_strategy=r["dc_strategy"], runtime_s=float(r["runtime_s"]))

            for r in conn.execute(
                    "SELECT * FROM sample_measurements WHERE run_id = ?", (run_pk,)):
                s = samples.get(int(r["sample_id"]))
                if s is None:
                    continue
                s.measurements[r["name"]] = _unfinite(r["value"])
                s.measurement_ok[r["name"]] = bool(r["ok"])
                if r["reason"]:
                    s.measurement_reasons[r["name"]] = r["reason"]

            for r in conn.execute(
                    "SELECT * FROM sample_variations WHERE run_id = ?", (run_pk,)):
                s = samples.get(int(r["sample_id"]))
                if s is not None:
                    s.slot_values[r["slot"]] = _unfinite(r["value"])

            for r in conn.execute(
                    "SELECT * FROM sample_devices WHERE run_id = ?", (run_pk,)):
                s = samples.get(int(r["sample_id"]))
                if s is not None:
                    s.device_values[r["key"]] = _unfinite(r["value"])

            for r in conn.execute(
                    "SELECT * FROM sample_specs WHERE run_id = ?", (run_pk,)):
                s = samples.get(int(r["sample_id"]))
                if s is not None:
                    s.spec_pass[r["spec_key"]] = bool(r["passed"])

            run.samples = sorted(samples.values(), key=lambda s: s.index)
            return run

    def load_netlist(self, run_id: str) -> str | None:
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT c.netlist FROM runs r JOIN circuits c ON c.id = r.circuit_id "
                "WHERE r.run_id = ?", (run_id,)).fetchone()
            return row["netlist"] if row else None

    def delete_run(self, run_id: str) -> bool:
        with self._lock, self.connect() as conn:
            cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            return cur.rowcount > 0

    def run_count(self) -> int:
        with self._lock, self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM runs")
                       .fetchone()["n"])

    def stats(self) -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            def count(table: str) -> int:
                return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
                           .fetchone()["n"])
            size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
            return {
                "path": os.path.abspath(self.path),
                "size_bytes": size,
                "projects": count("projects"), "circuits": count("circuits"),
                "runs": count("runs"), "samples": count("samples"),
                "measurements": count("sample_measurements"),
                "variations": count("sample_variations"),
                "schema_version": SCHEMA_VERSION,
            }


def _finite(value: Any) -> float | None:
    """SQLite stores NaN as NULL; be explicit about the round trip."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _unfinite(value: Any) -> float:
    return float("nan") if value is None else float(value)
