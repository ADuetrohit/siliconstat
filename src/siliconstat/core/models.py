"""Device model cards (the "technology" layer of the physics stack).

A model card holds the *process* parameters shared by every device instance
that references it.  Per-instance geometry (W, L) and per-instance statistical
offsets live on the device itself -- that separation is what makes global
process variation and local mismatch expressible independently.

Unit conventions
----------------
All electrical quantities are SI (volts, amperes, farads, metres) with two
deliberate exceptions, because these are universally quoted this way in the
mismatch literature:

``AVT``   threshold-mismatch Pelgrom coefficient, in **V*um**
          (e.g. ``AVT=3.5m`` means 3.5 mV*um)
``ABETA`` current-factor mismatch coefficient, **dimensionless * um**
          (e.g. ``ABETA=0.01`` means 1 %*um)

See ``docs/pelgrom.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .exceptions import NetlistSyntaxError

# Physical constants
BOLTZMANN = 1.380649e-23        # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
EPSILON_0 = 8.8541878128e-12    # F/m
EPSILON_OX = 3.9 * EPSILON_0    # SiO2 permittivity, F/m
KELVIN_OFFSET = 273.15


def thermal_voltage(temp_c: float) -> float:
    """kT/q in volts at *temp_c* degrees Celsius."""
    return BOLTZMANN * (temp_c + KELVIN_OFFSET) / ELEMENTARY_CHARGE


@dataclass
class MosfetModel:
    """Simplified (SPICE level-1 style) MOSFET process card.

    This is an **educational / engineering-grade** model: square-law with
    channel-length modulation, body effect and a first-order temperature
    dependence.  It is *not* BSIM and makes no claim of industrial accuracy.
    Short-channel effects (velocity saturation, DIBL, subthreshold conduction)
    are not modelled -- see ``docs/limitations.md``.
    """

    name: str
    mtype: str  # 'nmos' or 'pmos'

    #: SPICE model level.  Only 1 is implemented; the parameter is accepted so
    #: that a card written for any SPICE loads unchanged, and a card asking for
    #: a model this tool does not have fails with a message that says so.
    level: float = 1.0

    # --- DC core ---------------------------------------------------------
    vto: float = 0.45          # zero-bias threshold *magnitude* [V]
    kp: float = 200e-6         # transconductance parameter mu*Cox [A/V^2]
    lambda_: float = 0.10      # channel-length modulation [1/V]
    gamma: float = 0.0         # body-effect coefficient [V^0.5]
    phi: float = 0.8           # surface potential 2*phi_F [V]

    # --- geometry corrections -------------------------------------------
    ld: float = 0.0            # lateral diffusion per side [m]
    wd: float = 0.0            # width narrowing per side [m]

    # --- capacitances (used by AC / transient only) ----------------------
    tox: float = 4.0e-9        # gate oxide thickness [m]
    cox: float | None = None   # override oxide capacitance [F/m^2]
    cgso: float = 2.0e-10      # gate-source overlap capacitance [F/m]
    cgdo: float = 2.0e-10      # gate-drain overlap capacitance [F/m]
    cgbo: float = 0.0          # gate-bulk overlap capacitance [F/m]
    cjd: float = 0.0           # drain-bulk junction capacitance [F/m of width]
    cjs: float = 0.0           # source-bulk junction capacitance [F/m of width]

    # --- temperature -----------------------------------------------------
    tnom: float = 27.0         # nominal temperature [degC]
    tcv: float = -1.0e-3       # dVth/dT [V/K] (threshold falls with temperature)
    bex: float = -1.5          # mobility temperature exponent, KP ~ (T/Tnom)^BEX

    # --- statistical (Pelgrom) ------------------------------------------
    avt: float = 3.5e-3        # threshold mismatch coefficient [V*um]
    abeta: float = 1.0e-2      # current-factor mismatch coefficient [1 * um]

    extra: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mtype = self.mtype.lower()
        if self.mtype not in ("nmos", "pmos"):
            raise NetlistSyntaxError(f"unknown MOSFET type {self.mtype!r} (expected NMOS or PMOS)")
        # VTO is stored as a magnitude; accept the SPICE convention of a
        # negative VTO for PMOS.
        self.vto = abs(self.vto)
        if self.kp <= 0:
            raise NetlistSyntaxError(f"model {self.name}: KP must be > 0 (got {self.kp})")
        if self.phi <= 0:
            raise NetlistSyntaxError(f"model {self.name}: PHI must be > 0 (got {self.phi})")
        if self.gamma < 0:
            raise NetlistSyntaxError(f"model {self.name}: GAMMA must be >= 0 (got {self.gamma})")
        if int(self.level) != 1:
            raise NetlistSyntaxError(
                f"model {self.name}: LEVEL={self.level:g} is not implemented. "
                "SiliconStat provides only the square-law LEVEL=1 model; see "
                "docs/mosfet_model.md for exactly what that does and does not "
                "include.")

    @property
    def sign(self) -> float:
        """+1 for NMOS, -1 for PMOS -- maps real voltages to the n-type frame."""
        return 1.0 if self.mtype == "nmos" else -1.0

    @property
    def cox_area(self) -> float:
        """Oxide capacitance per unit area [F/m^2]."""
        if self.cox is not None:
            return self.cox
        return EPSILON_OX / self.tox

    # -- temperature scaling ---------------------------------------------
    def at_temperature(self, temp_c: float) -> "MosfetModel":
        """Return a copy of this card evaluated at *temp_c*.

        ``VTH(T) = VTO + TCV * (T - Tnom)``  (linear, first order)
        ``KP(T)  = KP * (T/Tnom)^BEX``       (mobility degradation)
        """
        if abs(temp_c - self.tnom) < 1e-12:
            return self
        t_abs = temp_c + KELVIN_OFFSET
        tnom_abs = self.tnom + KELVIN_OFFSET
        ratio = max(t_abs / tnom_abs, 1e-6)
        return replace(
            self,
            vto=max(self.vto + self.tcv * (temp_c - self.tnom), 1e-3),
            kp=self.kp * ratio ** self.bex,
            tnom=temp_c,
        )

    # -- Pelgrom helpers ---------------------------------------------------
    def sigma_vth(self, w: float, length: float) -> float:
        """Pelgrom threshold mismatch sigma [V] for a W x L device (metres)."""
        area_um2 = (w * 1e6) * (length * 1e6)
        if area_um2 <= 0:
            raise NetlistSyntaxError(f"model {self.name}: device area must be > 0")
        return self.avt / math.sqrt(area_um2)

    def sigma_beta_rel(self, w: float, length: float) -> float:
        """Pelgrom relative current-factor mismatch sigma (dimensionless)."""
        area_um2 = (w * 1e6) * (length * 1e6)
        if area_um2 <= 0:
            raise NetlistSyntaxError(f"model {self.name}: device area must be > 0")
        return self.abeta / math.sqrt(area_um2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "type": self.mtype, "vto": self.vto, "kp": self.kp,
            "level": self.level,
            "lambda": self.lambda_, "gamma": self.gamma, "phi": self.phi,
            "tox": self.tox, "cgso": self.cgso, "cgdo": self.cgdo,
            "avt": self.avt, "abeta": self.abeta, "tnom": self.tnom,
            "tcv": self.tcv, "bex": self.bex,
        }


@dataclass
class DiodeModel:
    """Shockley diode model card with series resistance and junction capacitance."""

    name: str
    is_: float = 1e-14      # saturation current [A]
    n: float = 1.0          # emission coefficient
    rs: float = 0.0         # series resistance [ohm]
    cj0: float = 0.0        # zero-bias junction capacitance [F]
    vj: float = 0.7         # junction potential [V]
    m: float = 0.5          # grading coefficient
    tnom: float = 27.0
    eg: float = 1.11        # bandgap [eV]
    xti: float = 3.0        # saturation-current temperature exponent
    extra: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.is_ <= 0:
            raise NetlistSyntaxError(f"model {self.name}: IS must be > 0")
        if self.n <= 0:
            raise NetlistSyntaxError(f"model {self.name}: N must be > 0")

    def at_temperature(self, temp_c: float) -> "DiodeModel":
        if abs(temp_c - self.tnom) < 1e-12:
            return self
        t_abs = temp_c + KELVIN_OFFSET
        tnom_abs = self.tnom + KELVIN_OFFSET
        vt = thermal_voltage(temp_c)
        ratio = t_abs / tnom_abs
        is_t = self.is_ * (ratio ** (self.xti / self.n)) * math.exp(
            -self.eg / (self.n * vt) * (1.0 - ratio)
        )
        return replace(self, is_=is_t, tnom=temp_c)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": "d", "is": self.is_, "n": self.n,
                "rs": self.rs, "cj0": self.cj0, "vj": self.vj, "m": self.m}


# Aliases accepted in ``.model`` cards, mapped to dataclass field names.
MOS_PARAM_ALIASES: dict[str, str] = {
    "level": "level",
    "vto": "vto", "vt0": "vto", "vth": "vto", "vth0": "vto",
    "kp": "kp", "u0cox": "kp", "beta": "kp",
    "lambda": "lambda_", "lam": "lambda_",
    "gamma": "gamma", "phi": "phi",
    "ld": "ld", "wd": "wd",
    "tox": "tox", "cox": "cox", "cgso": "cgso", "cgdo": "cgdo", "cgbo": "cgbo",
    "cjd": "cjd", "cjs": "cjs",
    "tnom": "tnom", "tcv": "tcv", "bex": "bex",
    "avt": "avt", "abeta": "abeta", "ab": "abeta",
}

DIODE_PARAM_ALIASES: dict[str, str] = {
    "is": "is_", "n": "n", "rs": "rs", "cj0": "cj0", "cjo": "cj0",
    "vj": "vj", "m": "m", "tnom": "tnom", "eg": "eg", "xti": "xti",
}
