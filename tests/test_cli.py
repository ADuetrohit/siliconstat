"""The ``siliconstat`` command-line interface, driven in process."""

from __future__ import annotations

import json
import os

import pytest

from siliconstat.cli.main import main

from conftest import example_path


def run_cli(*args: str) -> int:
    return main(list(args))


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "cli.sqlite")


# ---------------------------------------------------------------------------
# circuit validate
# ---------------------------------------------------------------------------

def test_validate_accepts_a_good_netlist(db, capsys):
    code = run_cli("--db", db, "circuit", "validate",
                   str(example_path("current_mirror")))
    out = capsys.readouterr().out
    assert code == 0
    assert "netlist is valid" in out
    assert "DC operating point converged" in out
    assert "MIRROR" in out            # the matched group is reported
    assert "M1" in out and "M2" in out


def test_validate_reports_measurements_and_specs(db, capsys):
    run_cli("--db", db, "circuit", "validate", str(example_path("diff_pair")))
    out = capsys.readouterr().out
    assert "vos" in out                    # the offset measurement is listed
    assert "srcp=VINP" in out              # ... with its resolved arguments
    assert "specification" in out


def test_validate_rejects_a_broken_netlist(db, tmp_path, capsys):
    bad = tmp_path / "bad.net"
    bad.write_text("R1 a b\n", encoding="utf-8")
    code = run_cli("--db", db, "circuit", "validate", str(bad))
    assert code == 1
    assert "resistance value" in capsys.readouterr().err


def test_validate_reports_a_missing_file(db, capsys):
    code = run_cli("--db", db, "circuit", "validate", "does_not_exist.net")
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_validate_can_skip_the_solve(db, capsys):
    code = run_cli("--db", db, "circuit", "validate",
                   str(example_path("current_mirror")), "--no-solve")
    assert code == 0
    assert "DC operating point converged" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_prints_the_operating_point_and_measurements(db, capsys):
    code = run_cli("--db", db, "simulate", str(example_path("current_mirror")))
    out = capsys.readouterr().out
    assert code == 0
    assert "operating point" in out
    assert "saturation" in out
    assert "gm/Id" in out
    assert "iref" in out and "ierr" in out
    assert "PASS" in out


def test_simulate_reports_the_analytical_divider_exactly(db, capsys):
    run_cli("--db", db, "simulate", str(example_path("rc_divider")))
    out = capsys.readouterr().out
    assert "3.75" in out
    assert "1.25 mA" in out or "0.00125" in out


def test_simulate_honours_a_pvt_condition(db, capsys):
    run_cli("--db", db, "simulate", str(example_path("current_mirror")),
            "--corner", "SS", "--supply", "1.62", "--temp", "125")
    out = capsys.readouterr().out
    assert "SS/1.62V/125C" in out
    assert "1.62" in out


def test_simulate_can_write_json(db, tmp_path, capsys):
    target = tmp_path / "sim.json"
    run_cli("--db", db, "simulate", str(example_path("current_mirror")),
            "--json", str(target))
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["operating_point"]["node_voltages"]["nref"] == pytest.approx(
        0.5383, abs=1e-3)
    assert payload["measurements"]["values"]["iref"] == pytest.approx(1e-5,
                                                                     rel=1e-4)


# ---------------------------------------------------------------------------
# monte-carlo
# ---------------------------------------------------------------------------

def test_monte_carlo_end_to_end(db, tmp_path, capsys):
    out_dir = tmp_path / "exports"
    code = run_cli("--db", db, "monte-carlo",
                   "--circuit", str(example_path("current_mirror")),
                   "--samples", "80", "--seed", "12345", "--mode", "both",
                   "--out", str(out_dir), "--quiet")
    out = capsys.readouterr().out
    assert code == 0
    assert "simulation status" in out
    assert "statistics" in out
    assert "yield" in out
    assert "COMBINED YIELD" in out
    assert "sensitivity" in out
    assert "M1.vth_local" in out
    assert "stored in" in out

    written = sorted(p.name for p in out_dir.iterdir())
    assert any(name.endswith("_samples.csv") for name in written)
    assert any(name.endswith("_report.html") for name in written)
    assert any(name.endswith("_run.json") for name in written)


def test_monte_carlo_mode_mismatch_only(db, capsys):
    run_cli("--db", db, "monte-carlo",
            "--circuit", str(example_path("current_mirror")),
            "--samples", "40", "--mode", "mismatch", "--quiet")
    out = capsys.readouterr().out
    assert "mismatch" in out
    assert "_local" in out
    assert "_global" not in out


def test_monte_carlo_nominal_mode_has_no_random_variables(db, capsys):
    run_cli("--db", db, "monte-carlo",
            "--circuit", str(example_path("rc_divider")),
            "--samples", "5", "--mode", "nominal", "--quiet")
    out = capsys.readouterr().out
    assert "random variables" in out


def test_monte_carlo_rejects_a_bad_sample_count(db, capsys):
    code = run_cli("--db", db, "monte-carlo",
                   "--circuit", str(example_path("rc_divider")),
                   "--samples", "0", "--quiet")
    assert code == 1
    assert "sample count" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# runs and reports
# ---------------------------------------------------------------------------

def seed_a_run(db: str) -> str:
    from siliconstat.db import RunStore

    main(["--db", db, "monte-carlo", "--circuit",
          str(example_path("current_mirror")), "--samples", "40", "--quiet"])
    return RunStore(db).list_runs()[0].run_id


def test_runs_list(db, capsys):
    seed_a_run(db)
    capsys.readouterr()
    assert run_cli("--db", db, "runs", "list") == 0
    out = capsys.readouterr().out
    assert "run id" in out
    assert "NMOS Current Mirror" in out


def test_runs_list_when_empty(db, capsys):
    assert run_cli("--db", db, "runs", "list") == 0
    assert "no stored runs" in capsys.readouterr().out


def test_runs_compare(db, capsys):
    first = seed_a_run(db)
    second = seed_a_run(db)
    capsys.readouterr()
    assert run_cli("--db", db, "runs", "compare", first, second) == 0
    out = capsys.readouterr().out
    assert "run comparison" in out
    assert first in out and second in out
    assert "measurement ierr" in out


def test_report_command_writes_html(db, tmp_path, capsys):
    run_id = seed_a_run(db)
    capsys.readouterr()
    target = tmp_path / "report.html"
    assert run_cli("--db", db, "report", "--run", run_id, "--out",
                   str(target)) == 0
    assert target.exists() and target.stat().st_size > 10_000
    assert run_id in target.read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out


def test_report_command_can_also_export_raw_data(db, tmp_path, capsys):
    run_id = seed_a_run(db)
    capsys.readouterr()
    run_cli("--db", db, "report", "--run", run_id,
            "--out", str(tmp_path / "r.html"), "--export", str(tmp_path / "raw"))
    assert any(p.name.endswith("_samples.csv")
               for p in (tmp_path / "raw").iterdir())


def test_report_for_an_unknown_run_fails_cleanly(db, capsys):
    assert run_cli("--db", db, "report", "--run", "nope") == 1
    assert "nope" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# db stats and misc
# ---------------------------------------------------------------------------

def test_db_stats(db, capsys):
    seed_a_run(db)
    capsys.readouterr()
    assert run_cli("--db", db, "db", "stats") == 0
    out = capsys.readouterr().out
    assert "runs" in out and "samples" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "siliconstat" in capsys.readouterr().out


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("circuit", "simulate", "monte-carlo", "pvt", "report",
                    "runs", "ml", "bench", "serve", "db"):
        assert command in out


@pytest.mark.slow
def test_pvt_sweep_command(db, capsys):
    code = run_cli("--db", db, "pvt", "--circuit",
                   str(example_path("current_mirror")),
                   "--samples", "20", "--metric", "ierr",
                   "--corners", "TT", "SS", "--temps", "27",
                   "--supplies", "1.8", "--quiet")
    out = capsys.readouterr().out
    assert code == 0
    assert "PVT x Monte Carlo" in out
    assert "worst-case condition" in out
    assert "TT" in out and "SS" in out


@pytest.mark.slow
def test_bench_command(db, tmp_path, capsys):
    target = tmp_path / "bench.json"
    code = run_cli("--db", db, "bench", "--circuit",
                   str(example_path("rc_divider")),
                   "--samples-list", "20", "--json", str(target))
    assert code == 0
    assert "samples/s" in capsys.readouterr().out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload[0]["samples"] == 20
    assert payload[0]["wall_s"] > 0
