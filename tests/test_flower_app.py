"""Flower Message API layer: run config, records, client state, strategy replies and the ClientApp.

No gnark service or network needed.
"""

import dataclasses
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Error, Message, MessageType, MetricRecord, RecordDict
from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]


# ─── Run config ──────────────────────────────────────────────────────────────


def test_pyproject_run_config_covers_every_field_and_round_trips():
    from ppflx.config import RUN_CONFIG_FIELDS, FLConfig

    defaults = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    assert set(defaults) == set(RUN_CONFIG_FIELDS)

    config = FLConfig.from_run_config(defaults)
    assert config.min_eval_clients is None and config.dirichlet_alpha is None and config.dp_epsilon == 10.0
    assert config.chain_ledger_path == "./results/ledger.json"
    assert FLConfig.from_run_config(config.to_run_config()) == config

    with pytest.raises(ValueError, match="unknown run config keys"):
        FLConfig.from_run_config({**defaults, "rounds": 3})


def test_launcher_casts_overrides_to_the_declared_types(tmp_path):
    from ppflx_bench.launch import make_run_config, write_run_config

    run_config = make_run_config({"num-rounds": 2.0, "learning-rate": 1, "sim-mode": True, "mode": "dp"})
    assert run_config["num-rounds"] == 2 and type(run_config["num-rounds"]) is int
    assert type(run_config["learning-rate"]) is float and run_config["sim-mode"] is True
    assert tomllib.loads(write_run_config(tmp_path / "rc.toml", run_config).read_text()) == run_config

    with pytest.raises(ValueError, match="unknown run config keys"):
        make_run_config({"rounds": 2})
    with pytest.raises(TypeError):
        make_run_config({"benchmark": "yes"})


def test_launched_processes_never_install_app_dependencies(tmp_path, monkeypatch):
    """The SuperLink defaults to `uv sync`-ing pyproject.toml from PyPI into a fresh env per run."""
    import ppflx_bench.launch as launch

    monkeypatch.setenv("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "0")
    env = launch.flwr_env(tmp_path / ".flwr")
    assert env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] == "1"
    assert env["FLWR_TELEMETRY_ENABLED"] == "0" and env["FLWR_DISABLE_UPDATE_CHECK"] == "1"

    spawned = []
    federation = launch.Federation(tmp_path, 2)
    monkeypatch.setattr(launch, "_require_free", lambda port: None)
    monkeypatch.setattr(launch, "_wait_for_port", lambda *a, **k: None)
    monkeypatch.setattr(federation, "_spawn", lambda cmd, log: spawned.append([str(c) for c in cmd]))
    federation._start_superlink()
    assert "--disable-runtime-dependency-installation" in spawned[0]


def test_harness_arguments_map_to_run_config_keys(tmp_path):
    from ppflx_bench.compare.experiment import run_config_for
    from ppflx_bench.launch import make_run_config

    base_args = {"dataset": "healthcare", "number_clients": 4, "rounds": 2, "max_epochs": 1, "dirichlet_alpha": None}
    run_config = make_run_config(run_config_for("zkp_sampled", base_args, str(tmp_path), simulation=False))
    assert run_config["mode"] == "zkp_sampled" and run_config["min-avail-clients"] == 4
    assert run_config["results-dir"] == str(tmp_path) and run_config["dirichlet-alpha"] == 0.0

    with pytest.raises(ValueError, match="without a run config key"):
        run_config_for("dp", {"frac_eval": 0.5}, str(tmp_path), simulation=True)


def test_a_completed_run_with_crashed_clients_is_not_a_success():
    """A ClientApp crash leaves the SuperNode up; the run completes with no aggregate."""
    from ppflx_bench.compare.experiment import round_failures

    clean = {"round_outcomes": [{"round": 1, "outcome": "committed", "flower_failures": 0}, {"round": 2, "outcome": "aggregated", "flower_failures": 0}]}
    assert round_failures(clean) == [] and round_failures(None) == []

    crashed = {"round_outcomes": [{"round": 1, "outcome": "no_results", "flower_failures": 2}, {"round": 2, "outcome": "aggregated", "flower_failures": 1}]}
    assert [p.split(":")[0] for p in round_failures(crashed)] == ["round 1", "round 2"]


def test_fab_carries_only_the_app_manifest_and_readme():
    """Run state under results/ (installed FAB copies in .flwr) once nested into every later FAB."""
    import io
    import zipfile

    from flwr.cli.build import build_fab_from_files

    files = {name: (REPO / name).read_bytes() for name in ("pyproject.toml", ".gitignore", "README.md")}
    files["ppflx/server.py"] = (REPO / "ppflx" / "server.py").read_bytes()
    files["results/healthcare/run/baseline/.flwr/apps/txcorpi0x.ppflx-bench.1.0.0.x/README.md"] = b"installed copy"
    files["docs/README.md"] = b"nested readme"

    fab, _ = build_fab_from_files(files)

    assert sorted(zipfile.ZipFile(io.BytesIO(fab)).namelist()) == [".info/CONTENT", "README.md", "pyproject.toml"]


# ─── Records and client state ────────────────────────────────────────────────


def test_records_take_numpy_scalars_and_reject_other_objects():
    from ppflx.records import config_record, metric_record

    assert dict(config_record({"a": np.int64(3), "b": np.float32(0.5), "c": "x", "d": None})) == {"a": 3, "b": 0.5, "c": "x"}
    with pytest.raises(TypeError):
        config_record({"x": object()})
    assert dict(metric_record({"n": np.int64(2), "ok": True, "s": "text"})) == {"n": 2, "ok": 1}


def test_client_state_round_trips_without_pickle():
    from ppflx.core.elgamal_gnark import GlobalModel
    from ppflx.records import pack_state, unpack_state

    commitment = {
        "round": 3,
        "q": np.arange(5, dtype=np.int64),
        "rand": b"\x00\x01",
        "max_update_norm": 0.5,
        "global": GlobalModel(weight=4, ct=b"ct", sums=np.array([1, -2], dtype=np.int64)),
        "missing": None,
    }
    out = unpack_state(*pack_state({"commitment": commitment}))["commitment"]

    assert out["round"] == 3 and out["rand"] == b"\x00\x01" and out["missing"] is None
    assert out["q"].dtype == np.int64 and out["q"].tolist() == [0, 1, 2, 3, 4]
    assert isinstance(out["global"], GlobalModel) and (out["global"].weight, out["global"].ct, out["global"].plain) == (4, b"ct", None)
    assert out["global"].sums.tolist() == [1, -2]


def test_client_state_only_rebuilds_allowlisted_types():
    from ppflx.records import pack_state, unpack_state

    @dataclasses.dataclass
    class GlobalModel:  # same name as the allowlisted class, different type
        weight: int

    with pytest.raises(TypeError, match="not the allowlisted"):
        pack_state({"x": GlobalModel(1)})
    with pytest.raises(TypeError, match="cannot be persisted"):
        pack_state({"x": object()})

    forged = ConfigRecord({"tree": json.dumps({"dict": {"x": {"dataclass": "Popen", "fields": {}}}})})
    with pytest.raises(ValueError, match="not allowlisted"):
        unpack_state(forged, ArrayRecord())


# ─── Strategy: replies ───────────────────────────────────────────────────────


def _instruction(node, message_type=MessageType.TRAIN):
    return Message(RecordDict({"config": ConfigRecord()}), dst_node_id=node, message_type=message_type, group_id="1")


def test_train_replies_become_results_and_failures():
    from ppflx.server import FedPrivate

    s = FedPrivate.__new__(FedPrivate)
    s._sampled = {("train", 1): [1, 2, 3, 4]}
    ok = Message(
        RecordDict(
            {
                "arrays": ArrayRecord.from_numpy_ndarrays([np.ones(2, np.float32)]),
                "fit_metrics": ConfigRecord({"zkp_proofs_json": "[]"}),
                "metrics": MetricRecord({"num-examples": 7}),
            }
        ),
        reply_to=_instruction(1),
    )
    error = Message(Error(code=0, reason="boom"), reply_to=_instruction(2))
    malformed = Message(RecordDict({"metrics": MetricRecord({"num-examples": 1})}), reply_to=_instruction(3))

    results, failures = s._fit_results(1, [ok, error, malformed])

    [(node, fit_res)] = results
    assert node.cid == "1" and fit_res.num_examples == 7 and fit_res.metrics == {"zkp_proofs_json": "[]"}
    assert parameters_to_ndarrays(fit_res.parameters)[0].tolist() == [1.0, 1.0]
    reasons = " | ".join(map(str, failures))
    assert len(failures) == 3 and "boom" in reasons and "malformed" in reasons and "node 4: no train reply" in reasons


# ─── ClientApp ───────────────────────────────────────────────────────────────


def _remembering_mode():
    """A plaintext mode that, like commit–challenge modes, needs its previous round's state."""
    from ppflx.privacy import get_privacy_mode

    class Remembering(type(get_privacy_mode("baseline"))):
        def setup_client_context(self, config):
            return {}

        def on_fit_config(self, context, fit_config):
            context["round"] = fit_config["server_round"]

        def send_parameters(self, net, context, **kwargs):
            params = super().send_parameters(net, context, **kwargs)
            previous = context.get("commitment")
            context["commitment"] = {"round": context["round"], "previous": previous and previous["round"], "values": params[0]}
            return params

        def post_fit_metrics(self, context, benchmark=None):
            return {"previous_round": context["commitment"]["previous"] or 0}

    return Remembering()


def test_client_app_carries_state_between_messages(tmp_path, monkeypatch):
    import ppflx_bench.app as app_module
    import ppflx.privacy
    from ppflx_bench.launch import make_run_config
    from ppflx.models import get_model_for_batch

    data = TensorDataset(torch.randn(24, 13), torch.randint(0, 2, (24,)))
    loaders = [DataLoader(data, batch_size=8) for _ in range(2)]

    def load(config):
        config.num_classes = 2
        return loaders, loaders

    mode = _remembering_mode()
    monkeypatch.setattr(app_module, "_load_data", load)
    monkeypatch.setattr(ppflx.privacy, "get_privacy_mode", lambda name: mode)

    run_config = make_run_config({"results-dir": str(tmp_path), "num-clients": 2})
    context = Context(run_id=1, node_id=11, node_config={"partition-id": 1, "num-partitions": 2}, state=RecordDict(), run_config=run_config)
    arrays = ArrayRecord.from_numpy_ndarrays([v.numpy() for v in get_model_for_batch(next(iter(loaders[0])), 2).state_dict().values()])

    for server_round in (1, 2):
        config = ConfigRecord({"server_round": server_round, "local_epochs": 1, "learning_rate": 0.01, "batch_size": 8})
        request = Message(RecordDict({"arrays": arrays, "config": config}), dst_node_id=11, message_type=MessageType.TRAIN, group_id=str(server_round))
        reply = app_module.client_app(request, context)
        assert not reply.has_error()
        assert reply.content["fit_metrics"]["previous_round"] == server_round - 1
        assert reply.content["metrics"]["num-examples"] == len(loaders[1])

    evaluation = app_module.client_app(
        Message(RecordDict({"arrays": arrays, "config": ConfigRecord({"server_round": 2})}), dst_node_id=11, message_type=MessageType.EVALUATE, group_id="2"),
        context,
    )
    assert {"loss", "num-examples", "accuracy"} <= set(evaluation.content["metrics"])

    raw = json.loads(context.state["ppflx.client.benchmark"]["raw"])
    assert len(raw["client_fit_time"]) == 2 and len(raw["test_accuracy"]) == 1
    assert json.loads((tmp_path / "client_1_benchmark.json").read_text())["mode"] == "baseline"


def test_client_app_rejects_a_node_config_that_does_not_match_the_run(tmp_path):
    import ppflx_bench.app as app_module
    from ppflx_bench.launch import make_run_config

    context = Context(
        run_id=1, node_id=11, node_config={"partition-id": 0, "num-partitions": 5}, state=RecordDict(),
        run_config=make_run_config({"results-dir": str(tmp_path), "num-clients": 2}),
    )
    with pytest.raises(ValueError, match="does not match num-clients=2"):
        app_module._client_for(context)
