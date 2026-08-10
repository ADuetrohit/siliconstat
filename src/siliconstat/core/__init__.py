"""Layer 1 + 2: circuit physics and the SPICE-like simulator core."""

from __future__ import annotations

from .circuit import Circuit, MeasureSpec, SpecLimit
from .devices import (
    Capacitor,
    CurrentSource,
    Device,
    Diode,
    Inductor,
    Mosfet,
    Resistor,
    VoltageSource,
)
from .exceptions import (
    ConvergenceError,
    NetlistSyntaxError,
    NumericalError,
    SiliconStatError,
    SimulationError,
)
from .models import DiodeModel, MosfetModel
from .netlist import parse_netlist, parse_netlist_file
from .solver import (
    ACResult,
    OperatingPoint,
    TransientResult,
    ac_analysis,
    solve_dc,
    transient_analysis,
)

__all__ = [
    "Circuit",
    "MeasureSpec",
    "SpecLimit",
    "Device",
    "Resistor",
    "Capacitor",
    "Inductor",
    "VoltageSource",
    "CurrentSource",
    "Diode",
    "Mosfet",
    "MosfetModel",
    "DiodeModel",
    "parse_netlist",
    "parse_netlist_file",
    "solve_dc",
    "ac_analysis",
    "transient_analysis",
    "OperatingPoint",
    "ACResult",
    "TransientResult",
    "SiliconStatError",
    "NetlistSyntaxError",
    "ConvergenceError",
    "SimulationError",
    "NumericalError",
]
