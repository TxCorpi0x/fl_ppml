"""The comparison harness: mode gating, run supervision, reporting and result merging."""

import json

import pytest


def test_skipped_modes_are_reported_and_fail_the_run(tmp_path, monkeypatch):
    import ppflx_bench.compare.runner as runner
    from ppflx_bench.compare.registry import MODES

    monkeypatch.setattr(MODES["dp"], "check_prerequisites", lambda: "Missing prerequisite: keys/dp/dp_params.json")
    monkeypatch.setattr(MODES["baseline"], "check_prerequisites", lambda: None)
    monkeypatch.setattr(
        runner,
        "run_experiment",
        lambda **kw: {"mode": kw["display_mode"], "success": True, "benchmark": {"rounds": 1, "num_clients": 2}},
    )
    monkeypatch.setattr(runner, "create_plots", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="dp.*skipped"):
        runner.run_comparison(
            dataset="healthcare", modes=["baseline", "dp"], num_clients=2, num_rounds=1,
            output_dir=str(tmp_path), chain_backend="none",
        )

    [report_path] = list(tmp_path.glob("healthcare/*/comparison_report.json"))
    by_mode = {r["mode"]: r for r in json.loads(report_path.read_text())}
    assert by_mode["dp"]["success"] is False and "Missing prerequisite" in by_mode["dp"]["skipped"]


def test_failed_results_never_replace_stored_dataset_entries(tmp_path):
    from ppflx_bench.compare.runner import _merge_into_dataset_report

    stored = [{"mode": "zkp", "success": True, "benchmark": {"rounds": 20}}]
    (tmp_path / "healthcare").mkdir()
    (tmp_path / "healthcare" / "comparison_report.json").write_text(json.dumps(stored))

    _merge_into_dataset_report([{"mode": "zkp", "success": False, "benchmark": None}], str(tmp_path), "healthcare")

    assert json.loads((tmp_path / "healthcare" / "comparison_report.json").read_text()) == stored


def test_harness_stops_a_run_that_outlives_its_clients():
    """A stuck run is stopped after the grace period, not after hours."""
    import time

    from ppflx_bench.launch import wait_for_run

    stopped = []
    t0 = time.monotonic()
    status, reason = wait_for_run(lambda: ("running", ""), lambda: False, lambda: stopped.append(1), timeout=0, grace=1, poll=0.1)
    assert status == "finished:stopped" and "after every client exited" in reason
    assert stopped == [1] and time.monotonic() - t0 < 5

    status, reason = wait_for_run(lambda: ("running", ""), lambda: True, lambda: stopped.append(2), timeout=0.5, grace=600, poll=0.1)
    assert status == "finished:stopped" and "timeout" in reason and stopped == [1, 2]

    done = wait_for_run(lambda: ("finished:completed", ""), lambda: False, lambda: pytest.fail("stopped a finished run"), timeout=0, grace=0, poll=0.1)
    assert done == ("finished:completed", "")


def test_zkp_mode_is_not_run_without_a_healthy_proof_service(tmp_path, monkeypatch):
    import ppflx_bench.compare.experiment as experiment
    from ppflx_bench.compare.registry import MODES

    monkeypatch.setattr(experiment, "_ensure_gnark_service", lambda log_dir: False)
    monkeypatch.setattr(experiment, "run_distributed", lambda *a, **k: pytest.fail("mode ran without a proof service"))

    result = experiment.run_experiment(MODES["zkp"], "zkp", {}, str(tmp_path), use_simulation=False)

    assert result["success"] is False and "gnark" in result["error"]


def test_harness_does_not_force_partial_encryption():
    from ppflx_bench.compare.experiment import _grpc_env

    assert "FL_ENCRYPT_LAYERS" not in _grpc_env({})
    assert _grpc_env({"FL_ENCRYPT_LAYERS": "ALL"})["FL_ENCRYPT_LAYERS"] == "ALL"


def test_report_marks_simulated_he_results(capsys):
    from ppflx_bench.compare.report import print_summary

    bm = lambda transport: {"transport": transport, "rounds": 1, "num_clients": 2, "timing": {}, "model_quality": {}}
    print_summary([
        {"mode": "he_tenseal", "success": True, "benchmark": bm("simulated")},
        {"mode": "zkp", "success": True, "benchmark": bm("network")},
    ])
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.startswith(("he_tenseal", "zkp"))}
    assert "[SIM]" in lines["he_tenseal"] and "[SIM]" not in lines["zkp"]


def test_simulated_result_never_replaces_a_networked_one(tmp_path):
    from ppflx_bench.compare.runner import _merge_into_dataset_report

    networked = {"mode": "he_tenseal", "success": True, "benchmark": {"transport": "network", "rounds": 20}}
    (tmp_path / "healthcare").mkdir()
    report = tmp_path / "healthcare" / "comparison_report.json"
    report.write_text(json.dumps([networked]))

    simulated = {"mode": "he_tenseal", "success": True, "benchmark": {"transport": "simulated", "rounds": 2}}
    _merge_into_dataset_report([simulated], str(tmp_path), "healthcare")
    assert json.loads(report.read_text()) == [networked]

    newer = {"mode": "he_tenseal", "success": True, "benchmark": {"transport": "network", "rounds": 5}}
    _merge_into_dataset_report([newer], str(tmp_path), "healthcare")
    assert json.loads(report.read_text()) == [newer]
