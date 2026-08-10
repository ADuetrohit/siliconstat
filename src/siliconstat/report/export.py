"""Raw-data export.

The exported table is deliberately *wide and complete*: one row per attempted
sample, carrying the seed, the drawn variation values, the resulting device
parameters, every measurement, the pass/fail verdict and the failure reason.
That is enough to reproduce, audit or re-analyse a run in any other tool
without going back to the simulator.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..analysis.summary import RunAnalysis
from ..core.exceptions import SiliconStatError
from ..mc.results import MonteCarloRun

__all__ = ["export_csv", "export_json", "export_parquet", "export_bundle",
           "export_summary_csv", "ExportError"]


class ExportError(SiliconStatError):
    """Raised when an export target is unavailable or fails."""


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def export_csv(run: MonteCarloRun, path: str) -> str:
    """Per-sample CSV: ``sample_id, seed, var.*, dev.*, meas.*, spec.*, status``."""
    _ensure_dir(path)
    frame = run.to_dataframe()
    frame.to_csv(path, index=False)
    return path


def export_parquet(run: MonteCarloRun, path: str) -> str:
    """Same table as :func:`export_csv`, in Parquet (requires pyarrow)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ExportError(
            "Parquet export needs the optional 'pyarrow' dependency; install it "
            "with `pip install siliconstat[parquet]` or export to CSV instead"
        ) from exc
    _ensure_dir(path)
    run.to_dataframe().to_parquet(path, index=False)
    return path


def export_json(run: MonteCarloRun, path: str,
                analysis: RunAnalysis | None = None, *,
                include_samples: bool = True) -> str:
    """Complete run record (and optionally its analysis) as JSON."""
    _ensure_dir(path)
    payload: dict[str, Any] = {"run": run.to_dict(include_samples=include_samples)}
    if analysis is not None:
        payload["analysis"] = analysis.to_dict()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_default, allow_nan=True)
    return path


def export_summary_csv(analysis: RunAnalysis, path: str) -> str:
    """Compact statistics-per-measurement CSV for spreadsheets."""
    import pandas as pd

    _ensure_dir(path)
    rows = []
    for name, stats in analysis.statistics.items():
        row = {"measurement": name}
        row.update(stats.to_dict())
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_bundle(run: MonteCarloRun, analysis: RunAnalysis, directory: str, *,
                  netlist: str = "", parquet: bool = True,
                  html: bool = True) -> dict[str, str]:
    """Write the full artefact set for a run into *directory*.

    Returns a mapping of artefact name to path.  Parquet is skipped (with the
    reason recorded) rather than failing the whole bundle when pyarrow is
    missing.
    """
    from .html_report import write_html_report

    os.makedirs(directory, exist_ok=True)
    stem = os.path.join(directory, f"{run.run_id}")
    written: dict[str, str] = {}
    written["csv"] = export_csv(run, f"{stem}_samples.csv")
    written["summary_csv"] = export_summary_csv(analysis, f"{stem}_statistics.csv")
    written["json"] = export_json(run, f"{stem}_run.json", analysis)
    if parquet:
        try:
            written["parquet"] = export_parquet(run, f"{stem}_samples.parquet")
        except ExportError as exc:
            written["parquet_skipped"] = str(exc)
    if html:
        written["html"] = write_html_report(
            f"{stem}_report.html", analysis, run, netlist=netlist)
    if netlist:
        netlist_path = f"{stem}_circuit.net"
        with open(netlist_path, "w", encoding="utf-8") as fh:
            fh.write(netlist)
        written["netlist"] = netlist_path
    return written


def _default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
