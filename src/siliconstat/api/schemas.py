"""Pydantic request/response models for the REST API.

Validation happens here rather than deeper in the stack so that a malformed
request is rejected at the edge with a precise message, and so the OpenAPI
schema published at ``/docs`` is a faithful contract for the frontend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..mc.config import MAX_SAMPLES
from ..variation.pvt import CORNERS

MAX_NETLIST_BYTES = 512 * 1024
VariationMode = Literal["nominal", "process", "mismatch", "both"]


class NetlistIn(BaseModel):
    netlist: str = Field(..., description="Netlist source text")
    name: str | None = Field(None, description="Optional circuit name override")

    @field_validator("netlist")
    @classmethod
    def _size(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("netlist is empty")
        if len(value.encode("utf-8")) > MAX_NETLIST_BYTES:
            raise ValueError(
                f"netlist exceeds the {MAX_NETLIST_BYTES // 1024} kB limit")
        return value


class CircuitRef(BaseModel):
    """Either inline netlist text, a stored circuit id, or a built-in example."""

    netlist: str | None = None
    circuit_id: int | None = None
    example: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "CircuitRef":
        provided = [x for x in (self.netlist, self.circuit_id, self.example)
                    if x not in (None, "")]
        if len(provided) != 1:
            raise ValueError(
                "provide exactly one of 'netlist', 'circuit_id' or 'example'")
        if self.netlist and len(self.netlist.encode("utf-8")) > MAX_NETLIST_BYTES:
            raise ValueError(
                f"netlist exceeds the {MAX_NETLIST_BYTES // 1024} kB limit")
        return self


class PVTIn(BaseModel):
    corner: str = "TT"
    supply: float | None = None
    temp_c: float = 27.0

    @field_validator("corner")
    @classmethod
    def _corner(cls, value: str) -> str:
        key = value.strip().upper()
        if key not in CORNERS:
            raise ValueError(f"unknown corner {value!r}; expected one of "
                             f"{', '.join(CORNERS)}")
        return key

    @field_validator("supply")
    @classmethod
    def _supply(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 < value <= 20.0):
            raise ValueError("supply voltage must be in (0, 20] V")
        return value

    @field_validator("temp_c")
    @classmethod
    def _temp(cls, value: float) -> float:
        if not (-273.0 < value <= 400.0):
            raise ValueError("temperature must be in (-273, 400] degrees Celsius")
        return value


class VariationIn(BaseModel):
    mode: VariationMode = "both"
    sigma_vth_global: float = Field(0.025, ge=0.0, le=1.0)
    sigma_beta_global_pct: float = Field(3.0, ge=0.0, le=100.0)
    include_passives: bool = True
    pelgrom: bool = True
    sigma_vth_local: float = Field(0.003, ge=0.0, le=1.0)
    sigma_beta_local_pct: float = Field(1.0, ge=0.0, le=100.0)
    distribution: Literal["gaussian", "uniform", "lognormal"] = "gaussian"
    truncate_sigma: float | None = Field(None, ge=1.0, le=10.0)


class SimulateIn(CircuitRef):
    pvt: PVTIn | None = None


class MonteCarloIn(CircuitRef):
    samples: int = Field(1000, ge=1, le=MAX_SAMPLES)
    seed: int = Field(12345, ge=0, le=2 ** 63 - 1)
    workers: int = Field(1, ge=1, le=64)
    sampling: Literal["standard", "latin_hypercube"] = "standard"
    variation: VariationIn = Field(default_factory=VariationIn)
    pvt: PVTIn | None = None
    label: str = ""
    project: str = "default"
    store: bool = True


class PVTSweepIn(CircuitRef):
    samples: int = Field(200, ge=1, le=100_000)
    seed: int = Field(12345, ge=0)
    corners: list[str] = Field(default_factory=lambda: list(CORNERS))
    supplies: list[float] | None = None
    supply_tolerance: float = Field(0.10, ge=0.0, le=0.5)
    temperatures: list[float] = Field(default_factory=lambda: [-40.0, 27.0, 125.0])
    variation: VariationIn = Field(default_factory=VariationIn)
    metric: str | None = None
    project: str = "default"

    @field_validator("corners")
    @classmethod
    def _corners(cls, values: list[str]) -> list[str]:
        out = []
        for v in values:
            key = v.strip().upper()
            if key not in CORNERS:
                raise ValueError(f"unknown corner {v!r}")
            out.append(key)
        if not out:
            raise ValueError("at least one corner is required")
        return out

    @model_validator(mode="after")
    def _grid_size(self) -> "PVTSweepIn":
        n_supplies = len(self.supplies) if self.supplies else 3
        total = len(self.corners) * n_supplies * len(self.temperatures)
        if total > 200:
            raise ValueError(
                f"the requested PVT grid has {total} conditions; keep it under 200")
        if total * self.samples > 2_000_000:
            raise ValueError(
                f"the requested grid would need {total * self.samples} "
                "simulations; reduce the sample count or the grid")
        return self


class SurrogateIn(CircuitRef):
    metric: str | None = None
    train_samples: int = Field(400, ge=20, le=50_000)
    test_samples: int = Field(200, ge=10, le=50_000)
    seed: int = Field(4242, ge=0)
    model: Literal["auto", "linear", "quadratic", "gradient_boosting",
                   "random_forest"] = "auto"
    variation: VariationIn = Field(default_factory=VariationIn)


class CompareIn(BaseModel):
    run_ids: list[str] = Field(..., min_length=2, max_length=8)


class JobOut(BaseModel):
    job_id: str
    run_id: str | None = None
    status: str
    message: str = ""


class JobStatusOut(BaseModel):
    job_id: str
    kind: str
    status: str
    run_id: str | None = None
    completed: int = 0
    total: int = 0
    successful: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    eta_s: float | None = None
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = ""
    finished_at: str | None = None


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    runs_stored: int
    active_jobs: int
    python: str


class ACSweepIn(BaseModel):
    fstart: float = Field(1.0, gt=0.0, le=1e15)
    fstop: float = Field(1e9, gt=0.0, le=1e15)
    points: int = Field(20, ge=2, le=200)
    sweep: Literal["dec", "oct", "lin"] = "dec"

    @model_validator(mode="after")
    def _range(self) -> "ACSweepIn":
        if self.fstop <= self.fstart:
            raise ValueError("fstop must be greater than fstart")
        return self


class TransientIn(BaseModel):
    tstep: float = Field(..., gt=0.0)
    tstop: float = Field(..., gt=0.0)
    tstart: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def _window(self) -> "TransientIn":
        if self.tstop <= self.tstart:
            raise ValueError("tstop must be greater than tstart")
        if (self.tstop - self.tstart) / self.tstep > 500_000:
            raise ValueError(
                "that window needs more than 500 000 timesteps; increase tstep")
        return self


class WaveformIn(CircuitRef):
    """Raw simulation traces: the AC response and/or the transient waveform."""

    pvt: PVTIn | None = None
    ac: ACSweepIn | None = None            # defaults to the netlist's .ac card
    tran: TransientIn | None = None        # defaults to the netlist's .tran card
    nodes: list[str] | None = None         # defaults to every non-ground node
    max_points: int = Field(2000, ge=50, le=20_000)
