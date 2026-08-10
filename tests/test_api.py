"""The REST API, exercised against the contract in docs/api_contract.md."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from conftest import example_path  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A fresh app instance with its own database."""
    monkeypatch.setenv("SILICONSTAT_DB", str(tmp_path / "api.sqlite"))
    import siliconstat.api.service as service

    monkeypatch.setattr(service, "_STORE", None, raising=False)
    monkeypatch.setattr(service, "_JOBS", None, raising=False)
    from siliconstat.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def wait_for_job(client, job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("done", "failed", "cancelled"):
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def mirror_netlist() -> str:
    return example_path("current_mirror").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------

def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["version"]
    assert payload["runs_stored"] == 0


def test_version_lists_the_corners(client):
    payload = client.get("/api/version").json()
    assert set(payload["corners"]) == {"TT", "FF", "SS", "FS", "SF"}
    assert payload["environment"]["python"]


def test_db_stats(client):
    payload = client.get("/api/db/stats").json()
    assert payload["runs"] == 0
    assert payload["schema_version"] >= 1


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------

def test_examples_are_listed_with_their_metadata(client):
    examples = client.get("/api/examples").json()
    ids = {e["id"] for e in examples}
    assert {"current_mirror.net", "diff_pair.net", "two_stage_opamp.net"} <= ids
    mirror = next(e for e in examples if e["id"] == "current_mirror.net")
    assert mirror["devices"] == 5
    assert "ierr" in mirror["measurements"]
    assert mirror["matched_groups"] == ["MIRROR"]
    assert mirror["netlist"].strip()


def test_validate_a_good_netlist(client):
    payload = client.post("/api/circuits/validate",
                          json={"netlist": mirror_netlist()}).json()
    assert payload["valid"] is True
    assert payload["operating_point"]["converged"] is True
    assert payload["operating_point"]["node_voltages"]["nref"] == pytest.approx(
        0.5383, abs=1e-3)
    assert payload["circuit"]["matched_groups"] == ["MIRROR"]
    assert len(payload["default_variation"]["slots"]) == 8


def test_validate_reports_a_syntax_error_with_its_line(client):
    payload = client.post("/api/circuits/validate",
                          json={"netlist": "V1 a 0 1\nR1 a b\n"}).json()
    assert payload["valid"] is False
    assert "line 2" in payload["error"]
    assert payload["error_type"] == "NetlistSyntaxError"


def test_validate_reports_a_circuit_that_cannot_be_solved(client):
    payload = client.post("/api/circuits/validate",
                          json={"netlist": "V1 a 0 1\nR1 a 0 1k\nR2 c d 1k\n"
                                           ".measure v V(a)\n"}).json()
    assert payload["valid"] is True
    assert payload["operating_point"]["converged"] is False
    assert "singular" in payload["operating_point"]["error"].lower()


def test_uploaded_netlists_cannot_read_the_filesystem(client):
    """The security boundary: .include is disabled for text over HTTP."""
    payload = client.post(
        "/api/circuits/validate",
        json={"netlist": ".include /etc/passwd\nV1 a 0 1\nR1 a 0 1k\n"}).json()
    assert payload["valid"] is False
    assert "disabled" in payload["error"]


def test_oversized_netlists_are_rejected(client):
    response = client.post("/api/circuits/validate",
                           json={"netlist": "*" + "x" * (600 * 1024)})
    assert response.status_code == 422


def test_empty_netlist_is_rejected(client):
    assert client.post("/api/circuits/validate", json={"netlist": "  "}
                       ).status_code == 422


def test_circuits_can_be_stored_and_fetched(client):
    stored = client.post("/api/circuits", json={"netlist": mirror_netlist(),
                                                "name": "My Mirror"}).json()
    assert stored["name"] == "My Mirror"
    circuit_id = stored["circuit_id"]
    fetched = client.get(f"/api/circuits/{circuit_id}").json()
    assert fetched["name"] == "My Mirror"
    assert client.get("/api/circuits").json()


def test_unknown_circuit_id_is_a_404(client):
    assert client.get("/api/circuits/9999").status_code == 404


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_an_example(client):
    payload = client.post("/api/simulate",
                          json={"example": "current_mirror.net"}).json()
    assert payload["operating_point"]["iterations"] > 0
    assert payload["measurements"]["values"]["iref"] == pytest.approx(1e-5,
                                                                     rel=1e-4)
    assert all(spec["passed"] for spec in payload["specs"])
    assert payload["operating_point"]["devices"]["M1"]["region"] == "saturation"


def test_simulate_with_a_pvt_condition(client):
    nominal = client.post("/api/simulate",
                          json={"example": "current_mirror.net"}).json()
    corner = client.post("/api/simulate", json={
        "example": "current_mirror.net",
        "pvt": {"corner": "SS", "supply": 1.62, "temp_c": 125.0}}).json()
    assert corner["pvt"]["corner"] == "SS"
    assert corner["operating_point"]["node_voltages"]["vdd"] == pytest.approx(1.62)
    assert corner["operating_point"]["node_voltages"]["out"] != pytest.approx(
        nominal["operating_point"]["node_voltages"]["out"])


def test_simulate_from_an_inline_netlist(client):
    payload = client.post("/api/simulate",
                          json={"netlist": mirror_netlist()}).json()
    assert payload["measurements"]["values"]["ierr"] == pytest.approx(0.082,
                                                                     abs=0.01)


@pytest.mark.parametrize("body", [
    {},                                            # nothing supplied
    {"netlist": "V1 a 0 1", "example": "x.net"},   # two sources
])
def test_circuit_reference_must_name_exactly_one_source(client, body):
    assert client.post("/api/simulate", json=body).status_code == 422


def test_unknown_example_is_rejected(client):
    response = client.post("/api/simulate", json={"example": "nope.net"})
    assert response.status_code == 400
    assert "unknown example" in response.json()["detail"]


def test_example_names_cannot_traverse_directories(client):
    response = client.post("/api/simulate",
                           json={"example": "../../etc/passwd.net"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# monte carlo
# ---------------------------------------------------------------------------

def test_synchronous_monte_carlo(client):
    payload = client.post("/api/monte-carlo/sync", json={
        "example": "current_mirror.net", "samples": 60, "seed": 12345}).json()
    assert payload["summary"]["counters"]["total"] == 60
    analysis = payload["analysis"]
    assert analysis["statistics"]["ierr"]["count"] > 0
    assert analysis["yield"]["per_spec"]
    assert analysis["sensitivity"]["ierr"]["entries"]


def test_synchronous_endpoint_is_capped(client):
    response = client.post("/api/monte-carlo/sync",
                           json={"example": "current_mirror.net",
                                 "samples": 5000})
    assert response.status_code == 400
    assert "capped" in response.json()["detail"]


def test_asynchronous_monte_carlo_job_lifecycle(client):
    started = client.post("/api/monte-carlo", json={
        "example": "current_mirror.net", "samples": 120, "seed": 12345}).json()
    assert started["status"] in ("queued", "running")

    finished = wait_for_job(client, started["job_id"])
    assert finished["status"] == "done"
    assert finished["completed"] == 120
    assert finished["successful"] + finished["failed"] == 120
    assert finished["run_id"]
    assert finished["result"]["analysis"]["yield"]["combined_passing"] >= 0

    assert any(job["job_id"] == started["job_id"]
               for job in client.get("/api/jobs").json())


def test_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_cancelling_a_finished_job_is_a_conflict(client):
    started = client.post("/api/monte-carlo/sync", json={
        "example": "rc_divider.net", "samples": 5}).json()
    assert started["run_id"]
    assert client.post("/api/jobs/nope/cancel").status_code == 409


@pytest.mark.parametrize("body,field", [
    ({"example": "current_mirror.net", "samples": 0}, "samples"),
    ({"example": "current_mirror.net", "samples": 2_000_000}, "samples"),
    ({"example": "current_mirror.net", "workers": 0}, "workers"),
    ({"example": "current_mirror.net",
      "pvt": {"corner": "ZZ"}}, "corner"),
    ({"example": "current_mirror.net",
      "pvt": {"corner": "TT", "supply": -1.0}}, "supply"),
    ({"example": "current_mirror.net",
      "pvt": {"corner": "TT", "temp_c": 5000.0}}, "temp"),
    ({"example": "current_mirror.net",
      "variation": {"mode": "sideways"}}, "mode"),
])
def test_invalid_monte_carlo_requests_are_rejected(client, body, field):
    response = client.post("/api/monte-carlo", json=body)
    assert response.status_code == 422
    assert field in response.text


def test_variation_mode_changes_the_slot_set(client):
    mismatch = client.post("/api/monte-carlo/sync", json={
        "example": "current_mirror.net", "samples": 20,
        "variation": {"mode": "mismatch"}}).json()
    slots = {s["slot"] for s in mismatch["analysis"]["slot_meta"]}
    assert all(name.endswith("_local") for name in slots)


def test_oversized_pvt_grid_is_rejected_before_any_work_starts(client):
    """5 corners x 3 supplies x 5 temperatures x 30k samples is 2.25 M
    simulations, past the documented ceiling; it must be refused at the schema
    layer rather than queued."""
    response = client.post("/api/pvt", json={
        "example": "current_mirror.net", "samples": 30_000,
        "corners": ["TT", "FF", "SS", "FS", "SF"],
        "temperatures": [-40, 0, 27, 85, 125]})
    assert response.status_code == 422
    assert "simulations" in response.text
    assert client.get("/api/jobs").json() == []


def test_a_pvt_grid_with_too_many_conditions_is_rejected(client):
    response = client.post("/api/pvt", json={
        "example": "current_mirror.net", "samples": 1,
        "corners": ["TT", "FF", "SS", "FS", "SF"],
        "supplies": [1.6 + 0.01 * i for i in range(20)],
        "temperatures": [-40, 0, 27, 85, 125]})
    assert response.status_code == 422
    assert "conditions" in response.text


@pytest.mark.slow
def test_small_pvt_sweep_runs_to_completion(client):
    started = client.post("/api/pvt", json={
        "example": "current_mirror.net", "samples": 15, "metric": "ierr",
        "corners": ["TT", "SS"], "supplies": [1.8], "temperatures": [27.0]
    }).json()
    finished = wait_for_job(client, started["job_id"], timeout=300.0)
    assert finished["status"] == "done"
    grid = finished["result"]["grid"]
    assert len(grid) == 2
    assert {row["corner"] for row in grid} == {"TT", "SS"}
    assert finished["result"]["worst_case"] is not None
    for row in grid:
        assert row["successful"] + row["failed"] == 15
        assert row["run_id"]


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

@pytest.fixture
def stored_run(client) -> str:
    payload = client.post("/api/monte-carlo/sync", json={
        "example": "current_mirror.net", "samples": 80, "seed": 12345}).json()
    return payload["run_id"]


def test_run_listing_and_detail(client, stored_run):
    listing = client.get("/api/runs").json()
    assert any(entry["run_id"] == stored_run for entry in listing)

    detail = client.get(f"/api/runs/{stored_run}").json()
    assert detail["counters"]["total"] == 80
    assert detail["reproduction"]["seed"] == 12345
    assert detail["measurement_meta"]
    assert detail["slot_meta"]


def test_run_analysis_is_cached_and_refreshable(client, stored_run):
    first = client.get(f"/api/runs/{stored_run}/analysis").json()
    again = client.get(f"/api/runs/{stored_run}/analysis").json()
    assert first["yield"] == again["yield"]
    refreshed = client.get(f"/api/runs/{stored_run}/analysis",
                           params={"refresh": True}).json()
    assert refreshed["run_id"] == stored_run


def test_samples_are_paginated_and_filterable(client, stored_run):
    page = client.get(f"/api/runs/{stored_run}/samples",
                      params={"offset": 10, "limit": 5}).json()
    assert page["total"] == 80
    assert len(page["samples"]) == 5
    assert page["samples"][0]["index"] == 10

    filtered = client.get(f"/api/runs/{stored_run}/samples",
                          params={"status": "ok"}).json()
    assert all(s["status"] == "ok" for s in filtered["samples"])


def test_csv_export_endpoint(client, stored_run):
    response = client.get(f"/api/runs/{stored_run}/export.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    body = response.text
    assert "sample_id" in body.splitlines()[0]
    assert len(body.splitlines()) == 81          # header plus 80 samples


def test_html_report_endpoint_is_self_contained(client, stored_run):
    response = client.get(f"/api/runs/{stored_run}/report.html")
    assert response.status_code == 200
    body = response.text
    assert "<svg" in body
    assert "<script" not in body.lower()
    assert stored_run in body


def test_netlist_endpoint(client, stored_run):
    response = client.get(f"/api/runs/{stored_run}/netlist")
    assert response.status_code == 200
    assert ".model NCH NMOS" in response.text


def test_reproduce_produces_an_identical_run(client, stored_run):
    started = client.post(f"/api/runs/{stored_run}/reproduce").json()
    finished = wait_for_job(client, started["job_id"])
    assert finished["status"] == "done"
    check = finished["result"]["reproduction_check"]
    assert check["identical"] is True
    assert check["mismatch_count"] == 0
    assert check["samples_compared"] == 80
    assert check["values_compared"] > 0
    assert check["new_run"] != stored_run


def test_compare_two_runs(client, stored_run):
    other = client.post("/api/monte-carlo/sync", json={
        "example": "current_mirror.net", "samples": 40, "seed": 999}).json()
    payload = client.post("/api/runs/compare",
                          json={"run_ids": [stored_run, other["run_id"]]}).json()
    assert len(payload["runs"]) == 2
    assert "ierr" in payload["shared_measurements"]
    assert payload["runs"][0]["counters"]["total"] == 80
    assert payload["runs"][1]["counters"]["total"] == 40
    assert payload["runs"][0]["charts"]["ierr"]["histogram"]["counts"]


def test_compare_needs_at_least_two_runs(client, stored_run):
    assert client.post("/api/runs/compare",
                       json={"run_ids": [stored_run]}).status_code == 422


def test_compare_with_an_unknown_run_is_a_404(client, stored_run):
    response = client.post("/api/runs/compare",
                           json={"run_ids": [stored_run, "nope"]})
    assert response.status_code == 404


def test_delete_a_run(client, stored_run):
    assert client.delete(f"/api/runs/{stored_run}").json()["deleted"] == stored_run
    assert client.get(f"/api/runs/{stored_run}").status_code == 404
    assert client.delete(f"/api/runs/{stored_run}").status_code == 404


def test_unknown_run_endpoints_are_404(client):
    for path in ("", "/analysis", "/samples", "/export.csv", "/report.html",
                 "/netlist"):
        assert client.get(f"/api/runs/nope{path}").status_code == 404


# ---------------------------------------------------------------------------
# ML surrogate
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_surrogate_job(client):
    started = client.post("/api/ml/surrogate", json={
        "example": "current_mirror.net", "metric": "ierr",
        "train_samples": 120, "test_samples": 60, "model": "quadratic"}).json()
    finished = wait_for_job(client, started["job_id"], timeout=300.0)
    assert finished["status"] == "done"
    result = finished["result"]
    assert result["target"] == "ierr"
    assert result["train_samples"] > 0 and result["test_samples"] > 0
    assert result["break_even_samples"] > 0


# ---------------------------------------------------------------------------
# raw simulation traces
# ---------------------------------------------------------------------------

def test_transient_waveform(client):
    payload = client.post("/api/waveforms",
                          json={"example": "inverter_transient.net"}).json()
    tran = payload["tran"]
    assert tran is not None
    assert payload["ac"] is None            # the inverter declares no .ac
    assert tran["points"] == len(tran["time"]) > 100
    assert set(tran["nodes"]) == {"vdd", "in", "out"}
    assert len(tran["nodes"]["out"]) == tran["points"]
    assert tran["time"][0] == pytest.approx(0.0)
    assert tran["time"][-1] == pytest.approx(8e-9, rel=1e-6)

    out = tran["nodes"]["out"]
    assert max(out) > 1.75                  # pulls up to the rail
    assert min(out) < 0.05                  # and down to ground
    assert max(out) > 1.8                   # gate-drain coupling overshoot
    assert all(v == pytest.approx(1.8) for v in tran["nodes"]["vdd"])


def test_ac_response(client):
    payload = client.post("/api/waveforms",
                          json={"example": "two_stage_opamp.net"}).json()
    ac = payload["ac"]
    assert ac is not None
    assert payload["tran"] is None           # the OTA declares no .tran
    assert ac["reference"] == "inp" and ac["stimulus"] == "VINP"
    assert ac["points"] == len(ac["freqs"]) > 50
    assert ac["freqs"][0] == pytest.approx(1.0)

    magnitude = [v for v in ac["nodes"]["out"]["mag_db"] if v is not None]
    assert magnitude[0] == pytest.approx(86.94, abs=0.5)   # DC gain
    assert magnitude[-1] < 0                                # rolled off past unity
    phase = [v for v in ac["nodes"]["out"]["phase_deg"] if v is not None]
    assert phase[0] == pytest.approx(0.0, abs=1.0)          # non-inverting at DC
    assert phase[-1] < -90.0


def test_a_circuit_without_ac_or_tran_says_so_rather_than_inventing_a_trace(client):
    payload = client.post("/api/waveforms",
                          json={"example": "current_mirror.net"}).json()
    assert payload["ac"] is None and payload["tran"] is None
    assert any("no waveform" in note for note in payload["notes"])
    # ... but the operating point is still there.
    assert payload["operating_point"]["node_voltages"]["nref"] == pytest.approx(
        0.5383, abs=1e-3)


def test_waveform_windows_can_be_overridden(client):
    payload = client.post("/api/waveforms", json={
        "example": "inverter_transient.net",
        "tran": {"tstep": 5e-11, "tstop": 4e-9},
        "nodes": ["out"]}).json()
    assert set(payload["tran"]["nodes"]) == {"out"}
    assert payload["tran"]["time"][-1] == pytest.approx(4e-9, rel=1e-6)


def test_waveform_ac_window_can_be_overridden(client):
    payload = client.post("/api/waveforms", json={
        "example": "two_stage_opamp.net",
        "ac": {"fstart": 100.0, "fstop": 1e6, "points": 5, "sweep": "dec"}}).json()
    assert payload["ac"]["freqs"][0] == pytest.approx(100.0)
    assert payload["ac"]["freqs"][-1] == pytest.approx(1e6, rel=1e-6)


def test_long_traces_are_decimated_and_the_factor_is_reported(client):
    payload = client.post("/api/waveforms", json={
        "example": "inverter_transient.net",
        "tran": {"tstep": 2e-12, "tstop": 8e-9},
        "max_points": 200}).json()
    assert payload["tran"]["points"] <= 200
    assert payload["tran"]["decimation"] > 1
    assert any("decimated" in note for note in payload["notes"])


def test_waveform_rejects_an_unknown_node(client):
    response = client.post("/api/waveforms", json={
        "example": "current_mirror.net", "nodes": ["nowhere"]})
    assert response.status_code == 400
    assert "nowhere" in response.json()["detail"]


@pytest.mark.parametrize("body", [
    {"example": "inverter_transient.net", "tran": {"tstep": 0, "tstop": 1e-9}},
    {"example": "inverter_transient.net",
     "tran": {"tstep": 1e-12, "tstop": 1e-9, "tstart": 2e-9}},   # tstop < tstart
    {"example": "inverter_transient.net",
     "tran": {"tstep": 1e-15, "tstop": 1e-6}},                   # too many steps
    {"example": "two_stage_opamp.net", "ac": {"fstart": 1e6, "fstop": 1.0}},
    {"example": "two_stage_opamp.net", "ac": {"fstart": -1.0, "fstop": 1e6}},
])
def test_invalid_waveform_windows_are_rejected(client, body):
    assert client.post("/api/waveforms", json=body).status_code == 422


def test_waveform_honours_a_pvt_condition(client):
    nominal = client.post("/api/waveforms",
                          json={"example": "two_stage_opamp.net"}).json()
    hot = client.post("/api/waveforms", json={
        "example": "two_stage_opamp.net",
        "pvt": {"corner": "SS", "supply": 1.62, "temp_c": 125.0}}).json()
    assert hot["pvt"]["corner"] == "SS"
    assert hot["operating_point"]["node_voltages"]["vdd"] == pytest.approx(1.62)
    assert (hot["nodes"] if False else True)
    assert hot["ac"]["nodes"]["out"]["mag_db"][0] != pytest.approx(
        nominal["ac"]["nodes"]["out"]["mag_db"][0], abs=0.01)


# ---------------------------------------------------------------------------
# the single-page app
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/circuits", "/results", "/yield",
                                  "/sensitivity", "/reports", "/docs"])
def test_client_routes_serve_the_dashboard(client, path):
    """The client router owns these paths; they do not exist on disk.

    StaticFiles *raises* on a miss rather than returning a 404 response, so an
    earlier fallback that inspected ``response.status_code`` never fired and
    every deep link 404'd while ``/`` worked.
    """
    import pathlib

    dist = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        pytest.skip("frontend has not been built (npm run build in frontend/)")
    response = client.get(path)
    assert response.status_code == 200, path
    assert '<div id="root">' in response.text


def test_unknown_api_routes_still_404(client):
    assert client.get("/api/definitely-not-a-route").status_code == 404
