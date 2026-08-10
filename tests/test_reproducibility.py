"""Reproducibility.

Sample *i* draws from ``SeedSequence(entropy=seed, spawn_key=(i,))``, which is
a pure function of the master seed and the sample index.  Three consequences
are asserted here, and they are the reason the scheme was chosen:

* the same seed reproduces a run exactly;
* worker count and completion order cannot change any result;
* a single sample can be re-simulated in isolation for debugging and will
  match the value it had inside the full run.
"""

from __future__ import annotations

import numpy as np
import pytest

from siliconstat.mc import (
    MonteCarloConfig,
    run_monte_carlo,
    sample_seed,
    sample_seed_sequence,
    simulate_sample,
)
from siliconstat.variation import VariationSampler, default_mismatch_model

from conftest import load


def make_config(**kwargs) -> MonteCarloConfig:
    circuit = kwargs.pop("circuit")
    defaults = dict(variation=default_mismatch_model(circuit), samples=120,
                    seed=12345, workers=1)
    defaults.update(kwargs)
    return MonteCarloConfig(**defaults)


# ---------------------------------------------------------------------------
# the seeding scheme itself
# ---------------------------------------------------------------------------

def test_sample_seed_is_a_pure_function_of_seed_and_index():
    assert sample_seed(12345, 7) == sample_seed(12345, 7)
    assert sample_seed(12345, 7) != sample_seed(12345, 8)
    assert sample_seed(12345, 7) != sample_seed(12346, 7)


def test_seed_sequences_are_independent_across_indices():
    a = np.random.default_rng(sample_seed_sequence(999, 0)).standard_normal(50)
    b = np.random.default_rng(sample_seed_sequence(999, 1)).standard_normal(50)
    assert not np.allclose(a, b)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.4    # 50 samples is noisy


def test_seed_sequence_is_order_independent():
    """Generating sample 5 before sample 2 must not change either."""
    first = np.random.default_rng(sample_seed_sequence(7, 5)).standard_normal(10)
    _ = np.random.default_rng(sample_seed_sequence(7, 2)).standard_normal(10)
    again = np.random.default_rng(sample_seed_sequence(7, 5)).standard_normal(10)
    assert np.array_equal(first, again)


# ---------------------------------------------------------------------------
# run-level reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_every_sample_exactly():
    circuit = load("current_mirror")
    config = make_config(circuit=circuit)
    first = run_monte_carlo(circuit, config)
    second = run_monte_carlo(circuit, config)
    assert len(first.samples) == len(second.samples)
    for a, b in zip(first.samples, second.samples):
        assert a.seed == b.seed
        assert a.status == b.status
        assert a.slot_values == b.slot_values
        for name, value in a.measurements.items():
            assert value == b.measurements[name], name


def test_different_seed_gives_different_samples():
    circuit = load("current_mirror")
    a = run_monte_carlo(circuit, make_config(circuit=circuit, seed=1))
    b = run_monte_carlo(circuit, make_config(circuit=circuit, seed=2))
    va = a.values("ierr")
    vb = b.values("ierr")
    assert not np.allclose(va, vb)
    # ... but they must describe the same distribution.
    assert va.std(ddof=1) == pytest.approx(vb.std(ddof=1), rel=0.25)


def test_a_single_sample_can_be_re_simulated_in_isolation():
    """The debugging workflow: pull sample 37 out of a 120-sample run and
    reproduce it on its own."""
    circuit = load("current_mirror")
    config = make_config(circuit=circuit)
    run = run_monte_carlo(circuit, config)
    sampler = VariationSampler(circuit, config.variation)

    target = run.samples[37]
    isolated = simulate_sample(circuit, sampler, config, index=37,
                               specs=circuit.specs)
    assert isolated.seed == target.seed
    assert isolated.status == target.status
    assert isolated.slot_values == target.slot_values
    assert isolated.measurements == target.measurements
    assert isolated.spec_pass == target.spec_pass


def test_sample_index_selects_the_draw_not_the_position():
    circuit = load("current_mirror")
    config = make_config(circuit=circuit, samples=10)
    sampler = VariationSampler(circuit, config.variation)
    out_of_order = [simulate_sample(circuit, sampler, config, index=i)
                    for i in (7, 2, 9, 0)]
    run = run_monte_carlo(circuit, config)
    for sample in out_of_order:
        assert sample.slot_values == run.samples[sample.index].slot_values


def test_a_prefix_of_a_longer_run_matches_a_shorter_run():
    """Because sample i is independent of the sample count, running 200
    samples reproduces the first 100 of a 100-sample run exactly."""
    circuit = load("current_mirror")
    short = run_monte_carlo(circuit, make_config(circuit=circuit, samples=60))
    long = run_monte_carlo(circuit, make_config(circuit=circuit, samples=150))
    for a, b in zip(short.samples, long.samples[:60]):
        assert a.slot_values == b.slot_values
        assert a.measurements == b.measurements


@pytest.mark.slow
def test_parallel_execution_matches_sequential_exactly():
    """Whether or not the process pool actually starts (it may fall back to
    sequential on a constrained machine), the results must be identical."""
    circuit = load("current_mirror")
    sequential = run_monte_carlo(circuit, make_config(circuit=circuit,
                                                      samples=200, workers=1))
    parallel = run_monte_carlo(circuit, make_config(circuit=circuit,
                                                    samples=200, workers=2))
    assert [s.index for s in parallel.samples] == list(range(200))
    for a, b in zip(sequential.samples, parallel.samples):
        assert a.seed == b.seed
        assert a.slot_values == b.slot_values
        assert a.measurements == b.measurements


def test_latin_hypercube_is_reproducible_too():
    circuit = load("current_mirror")
    config = make_config(circuit=circuit, samples=80, sampling="latin_hypercube")
    a = run_monte_carlo(circuit, config)
    b = run_monte_carlo(circuit, config)
    assert np.array_equal(a.values("ierr"), b.values("ierr"))


# ---------------------------------------------------------------------------
# the reproduction record
# ---------------------------------------------------------------------------

def test_reproduction_record_contains_everything_needed():
    circuit = load("current_mirror")
    run = run_monte_carlo(circuit, make_config(circuit=circuit, samples=20))
    record = run.reproduction_record()
    for key in ("run_id", "seed", "samples", "sampling", "variation", "solver",
                "software_version", "platform", "circuit_sha256", "timestamp"):
        assert key in record, key
    assert record["seed"] == 12345
    assert record["samples"] == 20
    assert record["circuit_sha256"]


def test_the_circuit_hash_changes_when_the_netlist_changes():
    from siliconstat.core.netlist import parse_netlist

    original = load("current_mirror")
    text = original.source_text or ""
    modified = parse_netlist(text.replace("W={WM}", "W={WM}", 1) + "\n* tweak\n",
                             allow_include=False)
    a = run_monte_carlo(original, make_config(circuit=original, samples=5))
    b = run_monte_carlo(modified, make_config(circuit=modified, samples=5))
    assert a.circuit_sha256 != b.circuit_sha256


def test_run_ids_are_unique_between_runs():
    circuit = load("current_mirror")
    config = make_config(circuit=circuit, samples=5)
    ids = {run_monte_carlo(circuit, config).run_id for _ in range(3)}
    assert len(ids) == 3
