"""SiliconStat -- Monte Carlo Mismatch Analysis and Statistical Verification
Platform for Analog ICs.

Layer hierarchy (bottom-up):

    circuit physics  ->  simulator  ->  mismatch model  ->  Monte Carlo
                     ->  statistics ->  yield          ->  visualization
                     ->  ML acceleration

Each layer only depends on the layers beneath it.  See ``docs/architecture.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
