"""Pelgrom-style area scaling of local device mismatch.

Pelgrom's law states that the mismatch between two identical, closely spaced
devices scales inversely with the square root of their active area::

    sigma(dVT)     = AVT   / sqrt(W * L)
    sigma(db / b)  = ABETA / sqrt(W * L)

Convention (important -- this is a factor of sqrt(2))
-----------------------------------------------------
``AVT`` and ``ABETA`` as quoted by foundries and in the original Pelgrom paper
describe the standard deviation of the **difference between a matched pair**.
The per-device standard deviation that a Monte Carlo engine must inject is
therefore smaller by sqrt(2)::

    sigma_device = AVT / sqrt(2 * W * L)

so that two independently perturbed devices reproduce
``sigma(dVT) = AVT / sqrt(W*L)``.  This is verified empirically in
``tests/test_pelgrom.py``.

Set ``pair_convention=False`` if your model card quotes AVT as a single-device
sigma instead; the difference is documented in ``docs/pelgrom.md``.

Units: ``AVT`` in V*um, ``ABETA`` dimensionless*um, ``W``/``L`` in metres.
"""

from __future__ import annotations

import math

from ..core.exceptions import VariationError

__all__ = ["area_um2", "sigma_pair", "sigma_device", "sigma_vth_device",
           "sigma_beta_device", "area_for_target_sigma"]

_SQRT2 = math.sqrt(2.0)


def area_um2(w: float, length: float) -> float:
    """Active area in square micrometres."""
    if w <= 0 or length <= 0:
        raise VariationError(
            f"Pelgrom scaling needs positive geometry (got W={w}, L={length})")
    return (w * 1e6) * (length * 1e6)


def sigma_pair(coefficient: float, w: float, length: float) -> float:
    """``A / sqrt(W*L)`` -- the sigma of the *difference* across a matched pair."""
    return coefficient / math.sqrt(area_um2(w, length))


def sigma_device(coefficient: float, w: float, length: float, *,
                 pair_convention: bool = True) -> float:
    """Per-device sigma to inject so a pair reproduces the quoted mismatch."""
    s = sigma_pair(coefficient, w, length)
    return s / _SQRT2 if pair_convention else s


def sigma_vth_device(avt: float, w: float, length: float, *,
                     pair_convention: bool = True) -> float:
    """Local threshold mismatch sigma for one device [V]."""
    return sigma_device(avt, w, length, pair_convention=pair_convention)


def sigma_beta_device(abeta: float, w: float, length: float, *,
                      pair_convention: bool = True) -> float:
    """Local current-factor mismatch sigma for one device (relative)."""
    return sigma_device(abeta, w, length, pair_convention=pair_convention)


def area_for_target_sigma(coefficient: float, target_sigma: float, *,
                          pair_convention: bool = True) -> float:
    """Inverse Pelgrom: active area (um^2) needed to reach *target_sigma*.

    The design question the law is actually used to answer -- "how big must
    this pair be to hold offset under 2 mV?".
    """
    if target_sigma <= 0:
        raise VariationError("target sigma must be > 0")
    effective = coefficient / (_SQRT2 if pair_convention else 1.0)
    return (effective / target_sigma) ** 2
