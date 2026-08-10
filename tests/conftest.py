"""Shared pytest fixtures.

Circuits are re-parsed per test rather than shared: parsing an example costs a
couple of milliseconds, and a shared mutable :class:`Circuit` would let one
test's finalisation leak into another's assertions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from siliconstat.core import parse_netlist_file
from siliconstat.core.circuit import Circuit
from siliconstat.core.solver import SolverOptions, solve_dc
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import default_mismatch_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = PROJECT_ROOT / "examples"

EXAMPLE_NAMES = (
    "rc_divider",
    "diode_bias",
    "current_mirror",
    "diff_pair",
    "two_stage_opamp",
    "inverter_transient",
)


def example_path(name: str) -> Path:
    path = EXAMPLES / f"{name}.net"
    if not path.is_file():
        raise FileNotFoundError(f"missing example netlist: {path}")
    return path


def load(name: str) -> Circuit:
    return parse_netlist_file(str(example_path(name)))


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture(params=EXAMPLE_NAMES)
def any_example(request) -> Circuit:
    """Parametrised over every shipped demo circuit."""
    return load(request.param)


@pytest.fixture
def rc_divider() -> Circuit:
    return load("rc_divider")


@pytest.fixture
def diode_bias() -> Circuit:
    return load("diode_bias")


@pytest.fixture
def current_mirror() -> Circuit:
    return load("current_mirror")


@pytest.fixture
def diff_pair() -> Circuit:
    return load("diff_pair")


@pytest.fixture
def two_stage_opamp() -> Circuit:
    return load("two_stage_opamp")


@pytest.fixture
def inverter() -> Circuit:
    return load("inverter_transient")


@pytest.fixture
def mirror_op(current_mirror: Circuit):
    return solve_dc(current_mirror)


@pytest.fixture
def solver_options() -> SolverOptions:
    return SolverOptions()


@pytest.fixture
def small_mirror_run(current_mirror: Circuit):
    """A modest, deterministic Monte Carlo run used by the analysis tests."""
    model = default_mismatch_model(current_mirror)
    config = MonteCarloConfig(variation=model, samples=200, seed=20240517,
                              workers=1)
    return run_monte_carlo(current_mirror, config)


@pytest.fixture
def tmp_db(tmp_path) -> str:
    return str(tmp_path / "test.sqlite")


def write_netlist(tmp_path, text: str, name: str = "circuit.net") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


MINIMAL_MOS_NETLIST = """
.title Minimal
.model NCH NMOS VTO=0.45 KP=200u LAMBDA=0.1 GAMMA=0.4 PHI=0.8 AVT=3.5m ABETA=0.01
VDD vdd 0 1.8
IREF vdd g 10u
M1 g g 0 0 NCH W=10u L=1u
.op
.measure id1 I(M1)
.measure vg  V(g)
.end
"""
