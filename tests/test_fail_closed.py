"""Fail-closed behaviour of the security-relevant paths.

Each test pins one error path that previously degraded silently. Tests that
need proofs use the real gnark service; outage tests point it at a dead port.
"""

import json
from collections import OrderedDict
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import fl.core.zkp_gnark as zkp_gnark
from tests.conftest import break_gnark, requires_gnark, use_gnark

DEAD = "http://127.0.0.1:1"

GLOBAL = OrderedDict(
    [
        ("model.0.weight", np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)),
        ("model.0.bias", np.array([0.05, -0.05], dtype=np.float32)),
    ]
)
HONEST = OrderedDict((k, (v + np.float32(0.01)).astype(np.float32)) for k, v in GLOBAL.items())  # small update


class TinyModel(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        if weights is not None:
            self.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})


def _fit(params, metrics, num_examples=10):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(list(params)), num_examples, metrics)


def _config(**overrides):
    return NS(**{"zkp_backend": "gnark", "sim_mode": False, "min_fit_clients": 1, **overrides})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FL_ZKP_BACKEND", "FL_ZKP_LAYERS", "FL_ZKP_ALLOW_PEDERSEN_STUB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FL_ZKP_MAX_NORM", "1.0")


@pytest.fixture
def live(monkeypatch, gnark):
    use_gnark(monkeypatch, gnark)
    return gnark


def _client_context(server):
    """A client that downloaded the server's global model and bound."""
    from fl.privacy.zkp import ZKPMode

    client, ctx = ZKPMode(), {"backend": "gnark"}
    client.on_fit_config(ctx, {"server_round": 1, **server.fit_config(1)})
    client.receive_parameters(TinyModel(), list(server._global), ctx, sim_mode=False)
    return client, ctx


def _upload(weights=HONEST):
    client, ctx = _client_context(_zkp_server())
    params = client.send_parameters(TinyModel(weights), ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def _zkp_server():
    from fl.privacy.zkp import ZKPMode

    mode = ZKPMode()
    mode.setup_server_context(_config())
    mode.bind_server_model(None, TinyModel(GLOBAL))
    return mode


def _composite_server():
    from fl.privacy.he_zkp import HeTensealZKPMode

    mode = HeTensealZKPMode()
    mode._zkp_mode.setup_server_context(_config())  # the HE context needs keys these tests don't use
    mode.bind_server_model(None, TinyModel(GLOBAL))
    return mode


# ─── ZKP client ──────────────────────────────────────────────────────────────


def test_client_proof_generation_failure_raises(monkeypatch):
    client, ctx = _client_context(_zkp_server())
    break_gnark(monkeypatch, DEAD)
    with pytest.raises(RuntimeError):
        client.send_parameters(TinyModel(HONEST), ctx, sim_mode=False)


def test_client_refuses_to_prove_without_global_model_or_bound():
    from fl.privacy.zkp import ZKPMode

    with pytest.raises(RuntimeError, match="no global model"):
        ZKPMode().send_parameters(TinyModel(HONEST), {"backend": "gnark", "max_update_norm": 1.0}, sim_mode=False)
    with pytest.raises(RuntimeError, match="no update-norm bound"):
        ZKPMode().on_fit_config({"backend": "gnark"}, {"server_round": 1})


def test_pedersen_stub_is_refused_unless_explicitly_allowed(monkeypatch):
    from fl.privacy.zkp import resolve_backend

    with pytest.raises(RuntimeError, match="no verification"):
        resolve_backend(_config(zkp_backend="pedersen"))
    monkeypatch.setenv("FL_ZKP_ALLOW_PEDERSEN_STUB", "1")
    assert resolve_backend(_config(zkp_backend="pedersen")) == "pedersen"


# ─── Plaintext ZKP server ────────────────────────────────────────────────────


@requires_gnark
def test_honest_clients_are_aggregated_and_anchored(live):
    mode = _zkp_server()
    results = [(NS(cid=c), _fit(*_upload())) for c in ("a", "b")]

    params, metrics = mode.aggregate_fit_override(1, results, [], None, _config())

    assert params is not None and metrics["round_outcome"] == "aggregated"
    assert mode.last_round_report["admitted"] == ["a", "b"]
    assert mode._last_anchor_data["client_ids"] == ["a", "b"]
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], HONEST["model.0.weight"], atol=1e-6)


@requires_gnark
def test_all_clients_rejected_leaves_model_unchanged(live):
    params, metrics = _upload()
    tampered = [p + 1.0 for p in params]
    mode = _zkp_server()
    results = [(NS(cid=c), _fit(tampered, metrics)) for c in ("a", "b")]

    aggregated, out = mode.aggregate_fit_override(1, results, [], None, _config())

    assert aggregated is None
    assert out["round_outcome"] == "no_quorum"
    assert set(mode.last_round_report["rejected"]) == {"a", "b"}
    assert mode._last_anchor_data is None


@requires_gnark
@pytest.mark.parametrize(
    "tamper",
    ["drop_layer_proof", "off_policy_bound", "missing_scale", "no_proofs", "wrong_shape_upload"],
)
def test_bad_uploads_are_rejected_without_raising(live, tamper):
    params, metrics = _upload()
    proofs = json.loads(metrics["zkp_proofs_json"])
    if tamper == "drop_layer_proof":
        proofs = proofs[:1]
    elif tamper == "off_policy_bound":
        proofs = [dict(p, bound_sq=str(zkp_gnark.policy_bound_sq(1.0, 6) + 1)) for p in proofs]
    elif tamper == "missing_scale":
        proofs = [{k: v for k, v in p.items() if k != "scale"} for p in proofs]
    elif tamper == "no_proofs":
        proofs = []
    elif tamper == "wrong_shape_upload":
        params = [np.zeros((3, 3), dtype=np.float32), params[1]]
    metrics = dict(metrics, zkp_proofs_json=json.dumps(proofs))
    mode = _zkp_server()

    aggregated, out = mode.aggregate_fit_override(1, [(NS(cid="c"), _fit(params, metrics))], [], None, _config())

    assert aggregated is None
    assert out["round_outcome"] == "no_quorum"
    assert "c" in mode.last_round_report["rejected"]


@requires_gnark
def test_quorum_below_min_fit_clients_leaves_model_unchanged(live):
    honest = _upload()
    params, metrics = _upload()
    mode = _zkp_server()
    results = [
        (NS(cid="honest"), _fit(*honest)),
        (NS(cid="tampered"), _fit([p + 1.0 for p in params], metrics)),
    ]

    aggregated, out = mode.aggregate_fit_override(1, results, [], None, _config(min_fit_clients=2))

    assert aggregated is None
    assert out == {"round_outcome": "no_quorum", "admitted": 1, "rejected": 1}


@requires_gnark
def test_verification_service_outage_aborts_round(live, monkeypatch):
    upload = _upload()
    mode = _zkp_server()
    break_gnark(monkeypatch, DEAD)

    aggregated, out = mode.aggregate_fit_override(1, [(NS(cid="c"), _fit(*upload))], [], None, _config())

    assert aggregated is None
    assert out == {"round_outcome": "infrastructure_abort"}
    assert mode.last_round_report["outcome"] == "infrastructure_abort"
    assert mode._last_anchor_data is None


def test_malformed_payload_counts_as_verification_failure(monkeypatch):
    break_gnark(monkeypatch, DEAD)
    params = list(HONEST.values())
    ok, failed = zkp_gnark.verify_gnark_proofs(params, list(HONEST), [{"layer": "model.0.weight", "shape": [2, 2]}])
    assert not ok and "model.0.weight" in failed
    ok, failed = zkp_gnark.verify_gnark_proofs(
        params, list(HONEST), [{"layer": "model.0.weight__chunk_0", "shape": [9], "scale": "1", "bound_sq": "1", "proof_b64": "AA=="}]
    )
    assert not ok


# ─── HE + ZKP composite ──────────────────────────────────────────────────────


@requires_gnark
def test_composite_never_aggregates_rejected_clients(live):
    honest_params, honest_metrics = _upload()
    mode = _composite_server()
    poisoned = [np.full_like(p, 1000.0) for p in honest_params]
    results = [
        (NS(cid="honest"), _fit(honest_params, honest_metrics)),
        (NS(cid="no_proofs"), _fit(poisoned, {})),
    ]

    # Simulation config: the HE backend defers to FedAvg, the old re-entry path.
    params, out = mode.aggregate_fit_override(1, results, [], None, _config(sim_mode=True))

    assert out["round_outcome"] == "aggregated" and out["admitted"] == 1
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], HONEST["model.0.weight"], atol=1e-6)


@requires_gnark
def test_composite_service_outage_aborts_round(live, monkeypatch):
    upload = _upload()
    mode = _composite_server()
    break_gnark(monkeypatch, DEAD)

    params, out = mode.aggregate_fit_override(1, [(NS(cid="c"), _fit(*upload))], [], None, _config())

    assert params is None and out == {"round_outcome": "infrastructure_abort"}


# ─── Strategy: chain ledger and round outcomes ───────────────────────────────


class _Chain:
    def __init__(self, fail_save=False):
        self.events, self.fail_save = [], fail_save

    def commit_model(self, **kw):
        self.events.append(("ModelCommit", kw["round"], len(kw["client_hashes"])))

    def anchor_proofs(self, **kw):
        self.events.append(("ProofAnchor", kw["round"], kw["client_ids"]))

    def save(self, path):
        if self.fail_save:
            raise OSError("disk full")


class _Mode:
    name = "fake"
    last_round_report = None

    def __init__(self, params, report):
        self._params, self._report = params, report

    def aggregate_fit_override(self, server_round, results, failures, ctx, config, bm):
        self.last_round_report = self._report
        return self._params, {}


def _strategy(mode, chain, ledger_path=None):
    from fl.core.benchmark import BenchmarkMetrics
    from fl.server import FedPrivate

    s = FedPrivate.__new__(FedPrivate)
    s.mode, s.chain, s.server_context = mode, chain, None
    s.config = NS(is_he=True, chain_ledger_path=ledger_path)
    s.benchmark = BenchmarkMetrics()
    return s


def test_no_model_commit_or_anchor_for_a_round_without_update():
    report = {"round": 1, "outcome": "no_quorum", "admitted": [], "rejected": {"x": "bad proof"}}
    chain = _Chain()
    s = _strategy(_Mode(None, report), chain)

    out = s.aggregate_fit(1, [(NS(cid="x"), _fit([np.zeros(2, np.float32)], {}))], [RuntimeError("client died")])

    assert out[0] is None
    assert chain.events == []
    [recorded] = s.benchmark.round_outcomes
    assert recorded["outcome"] == "no_quorum" and recorded["flower_failures"] == 1
    assert s.benchmark.summary()["round_outcomes"] == [recorded]


def test_round_outcome_records_update_clipping():
    report = {"round": 1, "outcome": "aggregated", "admitted": ["a", "b"], "rejected": {}}
    s = _strategy(_Mode(ndarrays_to_parameters([np.ones(2, np.float32)]), report), _Chain())
    results = [
        (NS(cid="a"), _fit([np.zeros(2, np.float32)], {"zkp_update_norm": 0.02, "zkp_update_clipped": 0})),
        (NS(cid="b"), _fit([np.zeros(2, np.float32)], {"zkp_update_norm": 0.09, "zkp_update_clipped": 1})),
    ]

    s.aggregate_fit(1, results, [])

    [recorded] = s.benchmark.round_outcomes
    assert recorded["clipped"] == ["b"] and recorded["update_norms"] == {"a": 0.02, "b": 0.09}


def test_model_commit_hashes_only_admitted_clients():
    report = {"round": 2, "outcome": "aggregated", "admitted": ["a"], "rejected": {"b": "bad proof"}}
    chain = _Chain()
    s = _strategy(_Mode(ndarrays_to_parameters([np.ones(2, np.float32)]), report), chain)
    results = [(NS(cid=c), _fit([np.zeros(2, np.float32)], {})) for c in ("a", "b")]

    s.aggregate_fit(2, results, [])

    assert chain.events == [("ModelCommit", 2, 1)]


class _Grid:
    """Node ids a SuperLink reports; the count changes between calls as nodes connect."""

    def __init__(self, *counts):
        self.counts = list(counts)

    def get_node_ids(self):
        count = self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]
        return list(range(1, count + 1))


def _sampling_strategy():
    from fl.server import FedPrivate

    s = FedPrivate.__new__(FedPrivate)
    s.fraction_fit = s.fraction_evaluate = 1.0
    s.min_fit_clients = s.min_evaluate_clients = 2
    s.min_available_clients = 3
    s.config = NS(local_epochs=1, learning_rate=0.001, batch_size=16)
    s.mode = NS(fit_config=lambda r: {}, evaluates_this_round=lambda r: True)
    s._sampled = {}
    return s


def test_first_round_samples_every_client_that_connects_while_waiting():
    """A client still starting when round 1 is configured must not be left out of it."""
    from flwr.app import ArrayRecord, ConfigRecord

    messages = _sampling_strategy().configure_train(1, ArrayRecord(), ConfigRecord(), _Grid(2, 3))

    assert sorted(m.metadata.dst_node_id for m in messages) == [1, 2, 3]


@pytest.mark.parametrize("phase", ["fit", "evaluate"])
def test_server_stops_instead_of_waiting_forever_for_departed_clients(monkeypatch, phase):
    """After every client exited, the server blocked in an unbounded wait."""
    import time

    from flwr.app import ArrayRecord, ConfigRecord

    monkeypatch.setenv("FL_CLIENT_WAIT_TIMEOUT", "1")
    s = _sampling_strategy()
    configure = s.configure_train if phase == "fit" else s.configure_evaluate

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="only 0 of 3 required clients available after 1s"):
        configure(1, ArrayRecord(), ConfigRecord(), _Grid(0))
    assert time.monotonic() - t0 < 5


def test_ledger_save_failure_raises(tmp_path):
    report = {"round": 1, "outcome": "aggregated", "admitted": ["a"], "rejected": {}}
    s = _strategy(_Mode(ndarrays_to_parameters([np.ones(2, np.float32)]), report), _Chain(fail_save=True), str(tmp_path / "l.json"))

    with pytest.raises(RuntimeError, match="could not save ledger"):
        s.aggregate_fit(1, [(NS(cid="a"), _fit([np.zeros(2, np.float32)], {}))], [])


# ─── HE and DP ──────────────────────────────────────────────────────────────


def test_tenseal_server_without_public_key_refuses_real_mode():
    from fl.privacy.he_tenseal import HeTensealMode

    missing = "/nonexistent/public_context.bin"
    with pytest.raises(FileNotFoundError):
        HeTensealMode().setup_server_context(NS(he_tenseal_public_path=missing, sim_mode=False))
    assert HeTensealMode().setup_server_context(NS(he_tenseal_public_path=missing, sim_mode=True)) is None


def test_tenseal_aggregation_without_context_raises():
    from fl.privacy.he_tenseal import HeTensealMode

    with pytest.raises(RuntimeError, match="no server context"):
        HeTensealMode().aggregate_fit_override(1, [], [], None, NS(sim_mode=False))


def test_tenseal_decryption_failure_raises():
    from fl.core.security import make_tenseal_context
    from fl.privacy.he_tenseal import HeTensealMode

    garbage = [np.frombuffer(b"not a ciphertext", dtype=np.uint8)] * 2
    with pytest.raises(Exception):
        HeTensealMode().receive_parameters(TinyModel(), garbage, make_tenseal_context(), sim_mode=False)


def test_undecodable_uint8_payload_is_never_averaged():
    from fl.privacy.he_tenseal import _decompress_cte2_results

    results = [(NS(cid="c"), _fit([np.frombuffer(b"\x00\x01garbage", dtype=np.uint8)], {}))]
    with pytest.raises(ValueError, match="not a decodable"):
        _decompress_cte2_results(results)


def test_tfhe_aggregation_without_context_or_on_error_raises():
    from fl.privacy.he_concrete_tfhe import HeConcreteThfeMode

    with pytest.raises(RuntimeError, match="no server context"):
        HeConcreteThfeMode().aggregate_fit_override(1, [], [], None, NS(sim_mode=False))
    results = [(NS(cid="c"), _fit([np.frombuffer(b"garbage", dtype=np.uint8)], {}))]
    with pytest.raises(RuntimeError, match="Encrypted aggregation failed"):
        HeConcreteThfeMode().aggregate_fit_override(1, results, [], object(), NS(sim_mode=False))


def test_dp_without_params_file_requires_explicit_epsilon():
    from fl.privacy.dp import DifferentialPrivacyMode

    base = dict(dp_params_path="/nonexistent/dp.pkl", dp_delta=1e-5, dp_max_grad_norm=1.0, dp_noise_multiplier=0.1)
    with pytest.raises(FileNotFoundError):
        DifferentialPrivacyMode().setup_client_context(NS(dp_epsilon=10.0, **base))
    params = DifferentialPrivacyMode().setup_client_context(NS(dp_epsilon=1.0, **base))
    assert params.epsilon == 1.0
    assert params.noise_multiplier == pytest.approx(np.sqrt(2 * np.log(1.25 / 1e-5)))


# ─── Harness ─────────────────────────────────────────────────────────────────


def test_skipped_modes_are_reported_and_fail_the_run(tmp_path, monkeypatch):
    import fl.compare.runner as runner
    from fl.compare.registry import MODES

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
    from fl.compare.runner import _merge_into_dataset_report

    stored = [{"mode": "zkp", "success": True, "benchmark": {"rounds": 20}}]
    (tmp_path / "healthcare").mkdir()
    (tmp_path / "healthcare" / "comparison_report.json").write_text(json.dumps(stored))

    _merge_into_dataset_report([{"mode": "zkp", "success": False, "benchmark": None}], str(tmp_path), "healthcare")

    assert json.loads((tmp_path / "healthcare" / "comparison_report.json").read_text()) == stored


def test_harness_stops_a_run_that_outlives_its_clients():
    """A stuck run is stopped after the grace period, not after hours."""
    import time

    from fl.launch import wait_for_run

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
    import fl.compare.experiment as experiment
    from fl.compare.registry import MODES

    monkeypatch.setattr(experiment, "_ensure_gnark_service", lambda log_dir: False)
    monkeypatch.setattr(experiment, "run_distributed", lambda *a, **k: pytest.fail("mode ran without a proof service"))

    result = experiment.run_experiment(MODES["zkp"], "zkp", {}, str(tmp_path), use_simulation=False)

    assert result["success"] is False and "gnark" in result["error"]
