"""Persistence: a stored run must come back byte-for-byte."""

from __future__ import annotations

import math

import numpy as np
import pytest

from siliconstat.analysis import analyse_run
from siliconstat.db import DatabaseError, RunStore
from siliconstat.mc import MonteCarloConfig, run_monte_carlo
from siliconstat.variation import ParameterVariation, VariationModel, default_mismatch_model

from conftest import load


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(str(tmp_path / "runs.sqlite"))


@pytest.fixture
def mirror_run():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=default_mismatch_model(circuit), samples=60, seed=808,
        workers=1))
    return circuit, run


# ---------------------------------------------------------------------------
# schema and lifecycle
# ---------------------------------------------------------------------------

def test_a_fresh_database_is_created_and_reports_its_schema(tmp_path):
    store = RunStore(str(tmp_path / "nested" / "dir" / "runs.sqlite"))
    stats = store.stats()
    assert stats["runs"] == 0
    assert stats["schema_version"] >= 1
    assert stats["size_bytes"] > 0


def test_projects_are_created_once(store):
    first = store.ensure_project("demo", "a description")
    assert store.ensure_project("demo") == first
    names = {p["name"] for p in store.list_projects()}
    assert "demo" in names


def test_circuits_are_deduplicated_by_content(store):
    circuit = load("current_mirror")
    a = store.save_circuit(circuit, project="demo")
    b = store.save_circuit(circuit, project="demo")
    assert a == b
    assert len(store.list_circuits(project="demo")) == 1


def test_get_circuit_and_by_name(store):
    circuit = load("current_mirror")
    circuit_id = store.save_circuit(circuit)
    record = store.get_circuit(circuit_id)
    assert record["name"] == circuit.name
    assert record["netlist"].strip()
    assert record["summary"]["n_nodes"] == circuit.n_nodes
    assert store.get_circuit_by_name(circuit.name)["id"] == circuit_id
    assert store.get_circuit_by_name("nothing here") is None


def test_unknown_circuit_id_is_reported(store):
    with pytest.raises(DatabaseError):
        store.get_circuit(9999)


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------

def test_every_sample_field_survives_the_round_trip(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit, project="demo", label="round trip")
    restored = store.load_run(run.run_id)

    assert restored.run_id == run.run_id
    assert restored.circuit_name == run.circuit_name
    assert restored.counters.to_dict() == run.counters.to_dict()
    assert restored.measurement_meta == run.measurement_meta
    assert restored.slot_meta == run.slot_meta
    assert restored.spec_meta == run.spec_meta
    assert restored.nominal == pytest.approx(run.nominal)
    assert restored.circuit_sha256 == run.circuit_sha256
    assert len(restored.samples) == len(run.samples)

    for a, b in zip(run.samples, restored.samples):
        assert a.index == b.index
        assert a.seed == b.seed
        assert a.status == b.status
        assert a.failure_reason == b.failure_reason
        assert a.passed == b.passed
        assert a.dc_strategy == b.dc_strategy
        assert a.measurements == pytest.approx(b.measurements)
        assert a.slot_values == pytest.approx(b.slot_values)
        assert a.device_values == pytest.approx(b.device_values)
        assert a.spec_pass == b.spec_pass


def test_nan_measurements_round_trip_through_null(store):
    """SQLite has no NaN, so NaN is stored as NULL and must come back as NaN --
    never as 0.0, which would silently corrupt a distribution."""
    circuit = load("current_mirror")
    model = VariationModel(variations=[
        ParameterVariation(parameter="beta", scope="local", sigma_pct=60.0,
                           targets="type:mosfet")])
    run = run_monte_carlo(circuit, MonteCarloConfig(
        variation=model, samples=120, seed=7, workers=1))
    store.save_run(run, circuit=circuit)
    restored = store.load_run(run.run_id)

    for a, b in zip(run.samples, restored.samples):
        for name, value in a.measurements.items():
            other = b.measurements[name]
            if math.isnan(value):
                assert math.isnan(other), f"{name} came back as {other}"
            else:
                assert other == pytest.approx(value)


def test_reloaded_run_supports_the_full_analysis(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    restored = store.load_run(run.run_id)
    original = analyse_run(run)
    again = analyse_run(restored)
    assert again.yield_report.combined_passing == \
        original.yield_report.combined_passing
    assert again.statistics["ierr"].mean == pytest.approx(
        original.statistics["ierr"].mean, rel=1e-12)


def test_value_arrays_match_after_reload(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    restored = store.load_run(run.run_id)
    assert np.array_equal(run.values("ierr"), restored.values("ierr"))


# ---------------------------------------------------------------------------
# analysis caching
# ---------------------------------------------------------------------------

def test_analysis_can_be_stored_and_reloaded(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    analysis = analyse_run(run).to_dict()
    store.save_analysis(run.run_id, analysis)
    cached = store.load_analysis(run.run_id)
    assert cached["run_id"] == run.run_id
    assert cached["yield"]["combined_passing"] == \
        analysis["yield"]["combined_passing"]


def test_saving_an_analysis_twice_replaces_it(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    store.save_analysis(run.run_id, analyse_run(run).to_dict())
    store.save_analysis(run.run_id, analyse_run(run).to_dict())
    assert store.load_analysis(run.run_id) is not None


def test_analysis_for_an_unknown_run_is_reported(store):
    assert store.load_analysis("nope") is None
    with pytest.raises(DatabaseError):
        store.save_analysis("nope", {})


# ---------------------------------------------------------------------------
# listing, netlists and deletion
# ---------------------------------------------------------------------------

def test_runs_are_listed_with_their_metadata(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit, project="demo", label="tagged")
    listing = store.list_runs(project="demo")
    assert len(listing) == 1
    info = listing[0]
    assert info.run_id == run.run_id
    assert info.label == "tagged"
    assert info.samples == 60
    assert info.seed == 808
    assert info.variation_mode == "process+mismatch"
    assert info.counters["successful"] == run.counters.successful
    assert info.to_dict()["run_id"] == run.run_id


def test_runs_can_be_filtered_by_circuit(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    assert store.list_runs(circuit_name=circuit.name)
    assert store.list_runs(circuit_name="something else") == []


def test_the_netlist_is_recoverable_from_the_run(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    netlist = store.load_netlist(run.run_id)
    assert netlist and ".model NCH NMOS" in netlist


def test_storing_the_same_run_id_twice_is_refused(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    with pytest.raises(DatabaseError) as excinfo:
        store.save_run(run, circuit=circuit)
    assert "already stored" in str(excinfo.value)


def test_deleting_a_run_cascades_to_its_samples(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    assert store.stats()["samples"] == 60
    assert store.delete_run(run.run_id) is True
    assert store.run_count() == 0
    assert store.stats()["samples"] == 0
    assert store.stats()["measurements"] == 0
    with pytest.raises(DatabaseError):
        store.load_run(run.run_id)


def test_deleting_an_unknown_run_returns_false(store):
    assert store.delete_run("nope") is False


def test_loading_an_unknown_run_is_reported(store):
    with pytest.raises(DatabaseError) as excinfo:
        store.load_run("nope")
    assert "nope" in str(excinfo.value)


def test_stats_counts_every_table(store, mirror_run):
    circuit, run = mirror_run
    store.save_run(run, circuit=circuit)
    stats = store.stats()
    assert stats["runs"] == 1
    assert stats["circuits"] == 1
    assert stats["samples"] == 60
    assert stats["measurements"] == 60 * len(run.measurement_names)
    assert stats["variations"] == 60 * len(run.slot_names)


def test_two_runs_of_the_same_circuit_share_one_circuit_row(store):
    circuit = load("current_mirror")
    config = MonteCarloConfig(variation=default_mismatch_model(circuit),
                              samples=5, seed=1, workers=1)
    for _ in range(2):
        store.save_run(run_monte_carlo(circuit, config), circuit=circuit)
    assert store.stats()["runs"] == 2
    assert store.stats()["circuits"] == 1
