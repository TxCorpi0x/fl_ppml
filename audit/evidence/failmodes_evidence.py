"""Step 4 Phase 1 evidence: exercise error paths against current code. No source edits."""
import io, json, contextlib, traceback, zlib
from collections import OrderedDict
from types import SimpleNamespace as NS

import numpy as np, torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

import fl.core.zkp_gnark as zg

DEAD = "http://127.0.0.1:1"
zg.DEFAULT_SERVICE_URL = DEAD

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
    logs = [l for l in out.getvalue().splitlines() if l.strip()]
    for l in logs[-4:]:
        print("   log:", l[:160])

W = OrderedDict(w=np.array([[0.1, -0.2]], dtype=np.float32), b=np.array([0.05], dtype=np.float32))
class Net:
    def state_dict(self): return OrderedDict((k, torch.from_numpy(v.copy())) for k, v in W.items())
FAKE_PROOFS = [{"layer": k, "shape": list(v.shape), "scale": "1000000.0", "bound_sq": "1", "hash_hex": "ab", "proof_b64": "AA=="} for k, v in W.items()]
cfg = NS(zkp_backend="gnark", sim_mode=False)

# a. client: proof generation with unreachable service
from fl.privacy.zkp import ZKPMode
def a():
    m = ZKPMode(); r = m._generate_proofs(Net(), {"backend": "gnark"}, phase="send")
    return {"backend": r[0], "num_proofs": len(r[1]), "metrics_sent": m.post_fit_metrics(None) if False else "n/a"}
run("a ZKP client, service unreachable", a)
def a2():
    m = ZKPMode(); params = m.send_parameters(Net(), {"backend": "gnark"}, sim_mode=False)
    return {"params_returned": len(params), "post_fit_metrics": m.post_fit_metrics(None)}
run("a2 ZKP client send_parameters then metrics", a2)

# b. plaintext zkp server: every client unverifiable
def b():
    m = ZKPMode()
    results = [(NS(cid=c), fit(list(W.values()), {"zkp_proofs_json": json.dumps(FAKE_PROOFS), "zkp_layer_names_json": json.dumps(list(W))})) for c in ("c1", "c2")]
    ret = m.aggregate_fit_override(1, results, [], None, cfg)
    admitted = [cp.cid for cp, _ in m.pre_aggregate(results, cfg)]
    return {"override_return": ret, "admitted_to_fedavg": admitted, "anchor": m._last_anchor_data and {k: (len(v) if isinstance(v, list) else v) for k, v in m._last_anchor_data.items()}}
run("b zkp server, all proofs fail (service unreachable)", b)
def b2():
    m = ZKPMode()
    results = [(NS(cid=c), fit(list(W.values()), {})) for c in ("c1", "c2")]
    m.aggregate_fit_override(1, results, [], None, cfg)
    return {"admitted_to_fedavg": [cp.cid for cp, _ in m.pre_aggregate(results, cfg)], "anchor": m._last_anchor_data}
run("b2 zkp server, no client sent proofs", b2)

# c. HE+ZKP composite: all excluded
from fl.privacy.he_zkp import HeTensealZKPMode
def c():
    m = HeTensealZKPMode()
    results = [(NS(cid=x), fit(list(W.values()), {"zkp_proofs_json": json.dumps(FAKE_PROOFS)})) for x in ("c1", "c2")]
    ret = m.aggregate_fit_override(1, results, [], None, cfg)  # server_context None
    return {"override_return": ret, "anchor_client_ids": m._last_anchor_data["client_ids"], "anchor_proofs": len(m._last_anchor_data["proof_hashes"])}
run("c he_tenseal_zkp server, all proofs fail", c)

# d. pedersen default
import main_server, main_client
def d():
    srv = main_server._build_parser() if hasattr(main_server, "_build_parser") else None
    m = ZKPMode()
    results = [(NS(cid="x"), fit(list(W.values()), {}))]
    ret = m.aggregate_fit_override(1, results, [], None, NS(zkp_backend="pedersen", sim_mode=False))
    admitted = [cp.cid for cp, _ in m.pre_aggregate(results, NS(zkp_backend="pedersen"))]
    return {"pedersen_override_return": ret, "admitted": admitted}
run("d zkp with pedersen backend (CLI default)", d)

# e. TenSEAL server without public context in non-sim
from fl.privacy.he_tenseal import HeTensealMode
from fl.core.security import make_tenseal_context, aggregate_custom
def e():
    he = HeTensealMode(); ctx = make_tenseal_context()
    class TNet:
        def state_dict(self): return Net().state_dict()
    enc = he._encrypt_params(TNet(), ctx)
    results = [(NS(cid=x), fit(enc, {})) for x in ("c1", "c2")]
    ret = he.aggregate_fit_override(1, results, [], None, cfg)
    pre = he.pre_aggregate(results, cfg)
    arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in pre]
    agg = aggregate_custom([(a, fr.num_examples) for a, (_, fr) in zip(arrays, pre)])
    return {"override_return": ret, "fedavg_input_dtypes": sorted({str(x.dtype) for x in arrays[0]}), "fedavg_output_dtypes": sorted({str(x.dtype) for x in agg}), "output_first_values": [x.ravel()[:3].tolist() for x in agg][:1]}
run("e he_tenseal server, server_context=None, non-sim", e)

# f. malformed proof payload in plaintext verification
def f():
    bad = [{"layer": "w", "shape": [1, 2]}]  # no scale/bound/proof
    return zg.verify_gnark_proofs(list(W.values()), list(W), bad)
run("f verify_gnark_proofs, proof missing fields", f)
def f2():
    bad = [{"layer": "w__chunk_0", "shape": [5], "scale": "1.0", "bound_sq": "1", "proof_b64": "AA=="}]
    return zg.verify_gnark_proofs(list(W.values()), list(W), bad)
run("f2 verify_gnark_proofs, chunk shape larger than layer", f2)

# g. DP without params file
from fl.privacy.dp import DifferentialPrivacyMode
def g():
    p = DifferentialPrivacyMode().setup_client_context(NS(dp_params_path="/nonexistent/dp.pkl", dp_epsilon=10.0, dp_delta=1e-5, dp_max_grad_norm=1.0, dp_noise_multiplier=0.484481))
    return {"epsilon": p.epsilon, "noise_multiplier": p.noise_multiplier}
run("g dp, params file missing", g)

# h. elgamal no admission → server still commits
from fl.server import FedPrivate
from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode
def h():
    mode = HeElGamalZKPMode()
    class Chain:
        events = []
        def commit_model(self, **kw): self.events.append(("ModelCommit", kw["round"], kw["num_clients"] if "num_clients" in kw else len(kw.get("client_hashes", []))))
        def anchor_proofs(self, **kw): self.events.append(("ProofAnchor", kw["round"]))
        def save(self, p): pass
    s = FedPrivate.__new__(FedPrivate)
    s.chain = Chain(); s.mode = mode; s.config = NS(chain_ledger_path=None)
    mode._last_anchor_data = None
    s._chain_commit(3, None, [(NS(cid="x"), fit([np.zeros(3, dtype=np.uint8)], {}))])
    return Chain.events
run("h FedPrivate._chain_commit after a no-update round", h)
