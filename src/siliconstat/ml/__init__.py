"""Layer 9: optional ML acceleration, built on top of real simulation data."""

from __future__ import annotations

from .surrogate import MODEL_KINDS, Surrogate, fit_surrogate, surrogate_experiment

__all__ = ["Surrogate", "fit_surrogate", "surrogate_experiment", "MODEL_KINDS"]
