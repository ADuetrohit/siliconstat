"""Exception hierarchy for SiliconStat.

The solver must never silently return an invalid result.  Every failure mode
raises a specific exception carrying enough context to debug it.
"""

from __future__ import annotations

from typing import Any


class SiliconStatError(Exception):
    """Base class for every error raised by SiliconStat."""


class NetlistSyntaxError(SiliconStatError):
    """Raised when a netlist cannot be parsed.

    Carries the offending line number and text so the CLI/API can report a
    precise, human-readable diagnostic.
    """

    def __init__(self, message: str, line_no: int | None = None, line: str | None = None,
                 source: str | None = None) -> None:
        self.message = message
        self.line_no = line_no
        self.line = line
        self.source = source
        parts = []
        if source:
            parts.append(str(source))
        if line_no is not None:
            parts.append(f"line {line_no}")
        location = ":".join(parts)
        full = f"{location}: {message}" if location else message
        if line is not None:
            full += f"\n    | {line.strip()}"
        super().__init__(full)


class CircuitError(SiliconStatError):
    """Raised when a circuit is structurally invalid (topology / references)."""


class SimulationError(SiliconStatError):
    """Base class for runtime simulation failures."""


class ConvergenceError(SimulationError):
    """Raised when the Newton-Raphson iteration fails to converge.

    Attributes
    ----------
    iterations:
        Number of Newton iterations performed before giving up.
    residual:
        Infinity-norm of the KCL residual (amperes) at the last iterate.
    max_dv:
        Largest node-voltage update (volts) at the last iterate.
    node:
        Name of the node with the largest remaining update, if known.
    strategies:
        Continuation strategies that were attempted, in order.
    """

    def __init__(self, message: str, *, iterations: int = 0, residual: float = float("nan"),
                 max_dv: float = float("nan"), node: str | None = None,
                 strategies: list[str] | None = None) -> None:
        self.iterations = iterations
        self.residual = residual
        self.max_dv = max_dv
        self.node = node
        self.strategies = strategies or []
        detail = (
            f"{message} (iterations={iterations}, |KCL residual|={residual:.3e} A, "
            f"max|dV|={max_dv:.3e} V"
        )
        if node:
            detail += f", worst node='{node}'"
        if self.strategies:
            detail += f", strategies tried={self.strategies}"
        detail += ")"
        super().__init__(detail)


class NumericalError(SimulationError):
    """Raised on singular matrices, NaN/Inf iterates or ill-conditioned systems."""

    def __init__(self, message: str, *, condition: float | None = None,
                 detail: Any = None) -> None:
        self.condition = condition
        self.detail = detail
        msg = message
        if condition is not None and condition == condition:  # not NaN
            msg += f" (matrix condition estimate ~ {condition:.3e})"
        super().__init__(msg)


class MeasurementError(SiliconStatError):
    """Raised when a measurement cannot be evaluated from a simulation result."""


class VariationError(SiliconStatError):
    """Raised for invalid statistical / variation configuration."""


class AnalysisError(SiliconStatError):
    """Raised for invalid statistical analysis requests (e.g. too few samples)."""
