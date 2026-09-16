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

def _repo_root() -> Path:
    """The checkout root, found by marker so the tests work at any depth."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() or (parent / "ppflx").is_dir():
            return parent
    raise RuntimeError("could not locate the repository root")


REPO = _repo_root()


# ─── Run config ──────────────────────────────────────────────────────────────








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





