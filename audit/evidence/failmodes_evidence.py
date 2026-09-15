"""Step 4 evidence: exercise ZKP-flow error paths with the gnark service unreachable.

Run from the repository root:

    PYTHONPATH=. python audit/evidence/failmodes_evidence.py

On commit b49c4c9 (Phase 1) this reproduces the fail-open rows quoted in
audit/failmodes.md. On current code it shows the fail-closed outcomes recorded
in that file's Phase 2 section. No source files are modified.
"""

import contextlib
import io
import json
from collections import OrderedDict
from types import SimpleNamespace as NS

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import fl.core.zkp_gnark as zg

DEAD = "http://127.0.0.1:1"
zg.DEFAULT_PROVER_URL = zg.DEFAULT_VERIFIER_URL = DEAD  # before Step 6: zg.DEFAULT_SERVICE_URL


def fit(params, metrics, n=10):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), n, metrics)


def run(label, fn):
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            result = fn()
        print(f"\n[{label}] -> {result}")
    except Exception as exc:
        print(f"\n[{label}] -> RAISED {type(exc).__name__}: {str(exc)[:160]}")
    logs = [line for line in out.getvalue().splitlines() if line.strip()]
    for line in logs[-4:]:
        print("   log:", line[:160])


W = OrderedDict(w=np.array([[0.1, -0.2]], dtype=np.float32), b=np.array([0.05], dtype=np.float32))


class Net:
    def state_dict(self):
        return OrderedDict((k, torch.from_numpy(v.copy())) for k, v in W.items())


class SchemaModel(torch.nn.Module):
    """Server model with the same layer names and shapes as W."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, 2))
        self.b = torch.nn.Parameter(torch.zeros(1))


FAKE_PROOFS = [
    {"layer": k, "shape": list(v.shape), "scale": str(float(zg.DEFAULT_SCALE)), "bound_sq": str(zg.policy_bound_sq()) if hasattr(zg, "policy_bound_sq") else "1", "hash_hex": "ab", "proof_b64": "AA=="}
    for k, v in W.items()
]
cfg = NS(zkp_backend="gnark", sim_mode=False)


def bind(mode):
    if hasattr(mode, "bind_server_model"):
        mode.bind_server_model(None, SchemaModel())
    return mode


def summarize(mode, ret, results, config):
    admitted = (getattr(mode, "last_round_report", None) or {}).get("admitted")
    if admitted is None:  # Phase 1 code: admission is the pre_aggregate side effect
        admitted = [cp.cid for cp, _ in mode.pre_aggregate(results, config)]
    anchor = getattr(mode, "_last_anchor_data", None)
    return {
        "override_return": ret if ret is None or ret[0] is None else ("<parameters>", ret[1]),
        "admitted": admitted,
        "outcome": (getattr(mode, "last_round_report", None) or {}).get("outcome"),
        "anchor": anchor and {k: (len(v) if isinstance(v, list) else v) for k, v in anchor.items()},
    }


from fl.privacy.zkp import ZKPMode  # noqa: E402


# a. client: proof generation with unreachable service
def a():
    r = ZKPMode()._generate_proofs(Net(), {"backend": "gnark"}, phase="send")
    return {"backend": r[0], "num_proofs": len(r[1])}


run("a ZKP client, service unreachable", a)


def a2():
    m = ZKPMode()
    params = m.send_parameters(Net(), {"backend": "gnark"}, sim_mode=False)
    return {"params_returned": len(params), "post_fit_metrics": m.post_fit_metrics(None)}


run("a2 ZKP client send_parameters then metrics", a2)


# b. plaintext zkp server: every client unverifiable
def b():
    m = bind(ZKPMode())
    results = [(NS(cid=c), fit(list(W.values()), {"zkp_proofs_json": json.dumps(FAKE_PROOFS), "zkp_layer_names_json": json.dumps(list(W))})) for c in ("c1", "c2")]
    return summarize(m, m.aggregate_fit_override(1, results, [], None, cfg), results, cfg)


run("b zkp server, all proofs fail (service unreachable)", b)


def b2():
    m = bind(ZKPMode())
    results = [(NS(cid=c), fit(list(W.values()), {})) for c in ("c1", "c2")]
    return summarize(m, m.aggregate_fit_override(1, results, [], None, cfg), results, cfg)


run("b2 zkp server, no client sent proofs", b2)


# c. HE+ZKP composite: all proofs fail
from fl.privacy.he_zkp import HeTensealZKPMode  # noqa: E402


def c():
    m = bind(HeTensealZKPMode())
    results = [(NS(cid=x), fit(list(W.values()), {"zkp_proofs_json": json.dumps(FAKE_PROOFS)})) for x in ("c1", "c2")]
    ret = m.aggregate_fit_override(1, results, [], None, cfg)
    return {"override_return": ret, "outcome": (getattr(m, "last_round_report", None) or {}).get("outcome"), "anchor": m._last_anchor_data}


run("c he_tenseal_zkp server, all proofs fail", c)


# d. pedersen backend (former CLI default)
def d():
    m = bind(ZKPMode())
    results = [(NS(cid="x"), fit(list(W.values()), {}))]
    config = NS(zkp_backend="pedersen", sim_mode=False)
    return summarize(m, m.aggregate_fit_override(1, results, [], None, config), results, config)


run("d zkp with pedersen backend (former CLI default)", d)


# e. TenSEAL server without public context in non-sim
from fl.core.security import aggregate_custom, make_tenseal_context  # noqa: E402
from fl.privacy.he_tenseal import HeTensealMode  # noqa: E402


def e():
    he = HeTensealMode()
    enc = he._encrypt_params(Net(), make_tenseal_context())
    results = [(NS(cid=x), fit(enc, {})) for x in ("c1", "c2")]
    ret = he.aggregate_fit_override(1, results, [], None, cfg)
    pre = he.pre_aggregate(results, cfg)
    arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in pre]
    agg = aggregate_custom([(arr, fr.num_examples) for arr, (_, fr) in zip(arrays, pre)])
    return {"override_return": ret, "fedavg_output_first_values": [x.ravel()[:3].tolist() for x in agg][:1]}


run("e he_tenseal server, server_context=None, non-sim", e)


# f. malformed proof payload in plaintext verification
def f():
    return zg.verify_gnark_proofs(list(W.values()), list(W), [{"layer": "w", "shape": [1, 2]}])


run("f verify_gnark_proofs, proof missing fields", f)


def f2():
    bad = [{"layer": "w__chunk_0", "shape": [5], "scale": "1.0", "bound_sq": "1", "proof_b64": "AA=="}]
    return zg.verify_gnark_proofs(list(W.values()), list(W), bad)


run("f2 verify_gnark_proofs, chunk shape larger than layer", f2)


# g. DP without params file
from fl.privacy.dp import DifferentialPrivacyMode  # noqa: E402


def g():
    p = DifferentialPrivacyMode().setup_client_context(
        NS(dp_params_path="/nonexistent/dp.pkl", dp_epsilon=10.0, dp_delta=1e-5, dp_max_grad_norm=1.0, dp_noise_multiplier=0.484481)
    )
    return {"epsilon": p.epsilon, "noise_multiplier": p.noise_multiplier}


run("g dp, params file missing", g)


# h. strategy ledger write after a no-update round
from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode  # noqa: E402
from fl.server import FedPrivate  # noqa: E402


def h():
    class Chain:
        events = []

        def commit_model(self, **kw):
            self.events.append(("ModelCommit", kw["round"], len(kw.get("client_hashes", []))))

        def anchor_proofs(self, **kw):
            self.events.append(("ProofAnchor", kw["round"]))

        def save(self, path):
            pass

    s = FedPrivate.__new__(FedPrivate)
    s.chain, s.mode, s.config = Chain(), HeElGamalZKPMode(), NS(chain_ledger_path=None)
    s._chain_commit(3, None, [(NS(cid="x"), fit([np.zeros(3, dtype=np.uint8)], {}))])
    return Chain.events


run("h FedPrivate._chain_commit after a no-update round", h)
