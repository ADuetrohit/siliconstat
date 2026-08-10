"""Time-domain source waveforms for transient analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["Waveform", "DCWave", "PulseWave", "SinWave", "PWLWave"]


class Waveform:
    """Base class: a deterministic value-vs-time function."""

    def value(self, t: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def dc_value(self) -> float:
        return self.value(0.0)

    def describe(self) -> str:  # pragma: no cover - trivial
        return self.__class__.__name__


@dataclass
class DCWave(Waveform):
    dc: float = 0.0

    def value(self, t: float) -> float:
        return self.dc

    def describe(self) -> str:
        return f"DC {self.dc}"


@dataclass
class PulseWave(Waveform):
    """``PULSE(v1 v2 td tr tf pw per)`` -- the classic SPICE trapezoidal pulse."""

    v1: float = 0.0
    v2: float = 1.0
    td: float = 0.0
    tr: float = 1e-9
    tf: float = 1e-9
    pw: float = 1e-6
    per: float = 2e-6

    def value(self, t: float) -> float:
        tr = max(self.tr, 1e-18)
        tf = max(self.tf, 1e-18)
        if t < self.td:
            return self.v1
        local = t - self.td
        if self.per > 0:
            local = math.fmod(local, self.per)
        if local < tr:
            return self.v1 + (self.v2 - self.v1) * (local / tr)
        if local < tr + self.pw:
            return self.v2
        if local < tr + self.pw + tf:
            return self.v2 + (self.v1 - self.v2) * ((local - tr - self.pw) / tf)
        return self.v1

    def dc_value(self) -> float:
        return self.v1

    def describe(self) -> str:
        return (f"PULSE({self.v1} {self.v2} {self.td} {self.tr} "
                f"{self.tf} {self.pw} {self.per})")


@dataclass
class SinWave(Waveform):
    """``SIN(vo va freq td theta)`` -- damped sinusoid."""

    vo: float = 0.0
    va: float = 1.0
    freq: float = 1e6
    td: float = 0.0
    theta: float = 0.0

    def value(self, t: float) -> float:
        if t < self.td:
            return self.vo
        local = t - self.td
        damp = math.exp(-self.theta * local) if self.theta else 1.0
        return self.vo + self.va * damp * math.sin(2.0 * math.pi * self.freq * local)

    def dc_value(self) -> float:
        return self.vo

    def describe(self) -> str:
        return f"SIN({self.vo} {self.va} {self.freq} {self.td} {self.theta})"


@dataclass
class PWLWave(Waveform):
    """``PWL(t1 v1 t2 v2 ...)`` -- piecewise-linear, held flat outside range."""

    points: Sequence[tuple[float, float]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        pts = sorted(self.points, key=lambda p: p[0])
        if not pts:
            raise ValueError("PWL waveform needs at least one (time, value) point")
        self.points = tuple(pts)

    def value(self, t: float) -> float:
        pts = self.points
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for i in range(1, len(pts)):
            t0, v0 = pts[i - 1]
            t1, v1 = pts[i]
            if t <= t1:
                if t1 == t0:
                    return v1
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        return pts[-1][1]  # pragma: no cover

    def dc_value(self) -> float:
        return self.points[0][1]

    def describe(self) -> str:
        body = " ".join(f"{t} {v}" for t, v in self.points)
        return f"PWL({body})"
