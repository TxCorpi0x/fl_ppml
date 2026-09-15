"""Step 7 Phase 2 evidence: the Phase 1 attacks against the update bound.

Run from the repository root (needs the built gnark service and the pinned
production keys; starts the prover and verifier itself):

    PYTHONPATH=. python audit/evidence/norm_phase2_evidence.py

Same setting as norm_evidence.py (healthcare, 2 clients, seed 42, lr 0.001,
momentum 0.9, batch 16, one local epoch), with the calibrated bound
B = PER_EPOCH_UPDATE_NORM["healthcare"]. For `zkp` and `he_elgamal_zkp`:

  honest    both clients use the client library (clip to B, prove Δ)
  A1 raw    boosted model replacement, proved without clipping (valid proofs,
            true declared bounds)
  A2 raw    the Phase 1 largest weight-admissible update, proved without clipping
  A1 lib    the boosted attacker using the client library, so its update is
            clipped to B and admitted: what a bounded update can still do

Each case runs one honest client plus the attacker through the real server
admission code on a fresh server and evaluates the aggregate on the test set.
No source files are modified.
"""

import concurrent.futures
import copy
import json
import tempfile
from collections import OrderedDict
from types import SimpleNamespace as NS

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import average_precision_score

from fl.compare import experiment
from fl.config import FLConfig
from fl.core import elgamal_gnark as eg
from fl.core import zkp_gnark as zg
from fl.core.engine import test
from fl.core.update_bound import max_update_norm
from fl.datasets import get_dataset_loader
from fl.models.registry import get_model_for_batch

SEED, CLIENTS, LR = 42, 2, 0.001


def flat(net):
    return np.concatenate([v.detach().cpu().numpy().reshape(-1).astype(np.float64) for v in net.state_dict().values()])


def set_flat(net, vec):
    out, off = copy.deepcopy(net), 0
    sd = out.state_dict()
    for k, v in sd.items():
        sd[k] = torch.tensor(vec[off : off + v.numel()], dtype=v.dtype).reshape(v.shape)
        off += v.numel()
    out.load_state_dict(sd)
    return out


def arrays(net):
    return [t.detach().cpu().numpy().copy() for t in net.state_dict().values()]


def local_train(net, loader, epochs, flip=False):
    net = copy.deepcopy(net)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss(net(x), 1 - y if flip else y).backward()
            opt.step()
    return net


def evaluate(net, testloader):
    _, acc, _, y_true, y_proba = test(net, testloader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    return acc, 100 * average_precision_score(y_true, np.asarray(y_proba)[:, 1])


def fit(params, metrics, n):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(list(params)), n, metrics)


# ── zkp ───────────────────────────────────────────────────────────────────────


def zkp_server(g, config):
    from fl.privacy.zkp import ZKPMode

    server = ZKPMode()
    server.setup_server_context(config)
    server.bind_server_model(None, g)
    return server


def zkp_library_upload(server, g, net):
    from fl.privacy.zkp import ZKPMode

    client, ctx = ZKPMode(), {"backend": "gnark"}
    client.on_fit_config(ctx, {"server_round": 1, **server.fit_config(1)})
    client.receive_parameters(copy.deepcopy(g), arrays(g), ctx, sim_mode=False)
    net = copy.deepcopy(net)
    params = client.send_parameters(net, ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def zkp_raw_upload(server, g, net):
    from fl.privacy.zkp import quantized_update

    names = list(g.state_dict().keys())
    delta = quantized_update(arrays(net), arrays(g))
    proofs, _ = zg.generate_gnark_proofs(OrderedDict(zip(names, delta)))  # no total: true bounds
    return arrays(net), {"zkp_proofs_json": json.dumps(proofs)}


def zkp_case(g, config, honest, attacker, n):
    server = zkp_server(g, config)
    results = [(NS(cid="honest"), fit(*zkp_library_upload(server, g, honest), n[0]))]
    results.append((NS(cid="attacker"), fit(*attacker(server), n[1])))
    params, _ = server.aggregate_fit_override(1, results, [], None, NS(zkp_backend="gnark", sim_mode=False, min_fit_clients=1))
    agg = set_flat(g, np.concatenate([p.reshape(-1) for p in parameters_to_ndarrays(params)])) if params else None
    return server.last_round_report, agg


# ── he_elgamal_zkp ────────────────────────────────────────────────────────────


def eg_client(keys, server, g):
    from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode

    client = HeElGamalZKPMode()
    ctx = client.setup_client_context(keys)
    client.on_fit_config(ctx, {"server_round": 1, **server.fit_config(1)})
    client.receive_parameters(copy.deepcopy(g), arrays(g), ctx, sim_mode=False)
    return client, ctx


def eg_library_upload(keys, server, g, net):
    client, ctx = eg_client(keys, server, g)
    params = client.send_parameters(copy.deepcopy(net), ctx, sim_mode=False)
    return params, client.post_fit_metrics(ctx)


def eg_raw_upload(keys, server, g, net):
    """Encrypt the unclipped model and prove every chunk with its true bound."""
    _, ctx = eg_client(keys, server, g)
    policy, glob = ctx["policy"], ctx["global"]
    schema = eg.schema_of(net.state_dict())
    q = eg.quantize(flat(net), policy.scale, "attacker")
    chunks = eg.chunks_for(schema, policy.chunk_size)
    indices = [eg.chunk_indices(schema, c) for c in chunks]
    bounds = [max(1, eg.update_energy(q[i], glob.centered_sums()[i], glob.weight)) for i in indices]

    def prove(k):
        return eg.prove_chunk(ctx["pk"], ctx["sk"], q[indices[k]], bounds[k], eg.context_value(1, chunks[k]), glob.request(indices[k], prover=True))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(prove, range(len(chunks))))
    layers = [bytearray(eg.numel(s) * eg.CIPHERTEXT_BYTES) for _, s in schema]
    proofs = []
    for k, (chunk, (ct, proof)) in enumerate(zip(chunks, outputs)):
        layers[chunk.layer][chunk.start * eg.CIPHERTEXT_BYTES : (chunk.start + chunk.size) * eg.CIPHERTEXT_BYTES] = ct
        proofs.append({"layer": chunk.layer, "chunk": chunk.index, "bound_sq": str(bounds[k]), "proof_b64": proof, "vk_sha256": eg.pinned_vk()})
    header = np.array([eg.HEADER_MAGIC, 1, 1], dtype=np.int64)
    return [header] + [np.frombuffer(bytes(b), dtype=np.uint8) for b in layers], {"zkp_proofs_json": json.dumps(proofs)}


def eg_case(g, keys, honest, attacker, n):
    from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode

    server = HeElGamalZKPMode()
    sctx = server.setup_server_context(keys)
    server.bind_server_model(sctx, g)
    results = [(NS(cid="honest"), fit(*eg_library_upload(keys, server, g, honest), n[0]))]
    results.append((NS(cid="attacker"), fit(*attacker(server), n[1])))
    params, _ = server.aggregate_fit_override(1, results, [], sctx, keys)
    agg = None
    if params is not None:
        reader = HeElGamalZKPMode()
        rctx = reader.setup_client_context(keys)
        agg = copy.deepcopy(g)
        reader.receive_parameters(agg, parameters_to_ndarrays(params), rctx, sim_mode=False)
    return server.last_round_report, agg


def main():
    config = FLConfig(dataset="healthcare", num_clients=CLIENTS, seed=SEED, batch_size=16, sim_mode=False, local_epochs=1)
    Loader = get_dataset_loader("healthcare")
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, _, testloader = Loader().load(config)
    torch.manual_seed(SEED)
    g = get_model_for_batch(next(iter(testloader)), config.num_classes)
    B = max_update_norm(config)
    n = (len(trainloaders[0]), len(trainloaders[1]))

    h0, h1 = local_train(g, trainloaders[0], 1), local_train(g, trainloaders[1], 1)
    bad = local_train(g, trainloaders[1], 5, flip=True)
    a1 = set_flat(g, flat(g) + CLIENTS * (flat(bad) - flat(g)))
    # Phase 1 A2 for he_elgamal_zkp: uniform magnitude filling the old weight bound of 100.
    a2 = set_flat(g, 0.99 * (100.0 / np.sqrt(flat(g).size)) * -np.sign(flat(g)))

    print(f"bound B = {B:.6f} (healthcare, 1 epoch); model {flat(g).size} parameters")
    for label, net in (("honest 0", h0), ("honest 1", h1), ("A1", a1), ("A2", a2)):
        print(f"  {label:8s} ‖Δ‖ = {np.linalg.norm(flat(net) - flat(g)):.4f} ({np.linalg.norm(flat(net) - flat(g)) / B:.1f}× B)")
    print("reference: FedAvg(honest 0, honest 1) unclipped acc/AUPRC = %.1f / %.1f" % evaluate(set_flat(g, (n[0] * flat(h0) + n[1] * flat(h1)) / sum(n)), testloader))
    print("           attacker's model alone         acc/AUPRC = %.1f / %.1f" % evaluate(bad, testloader))

    if not experiment._ensure_gnark_service(tempfile.mkdtemp()):
        raise SystemExit("gnark services unavailable")
    try:
        from fl.keys.he_elgamal import generate

        d = tempfile.mkdtemp()
        generate(secret_path=f"{d}/s.json", public_path=f"{d}/p.json")
        keys = NS(sim_mode=False, he_elgamal_secret_path=f"{d}/s.json", he_elgamal_public_path=f"{d}/p.json",
                  min_fit_clients=1, dataset="healthcare", local_epochs=1)
        zcfg = NS(zkp_backend="gnark", sim_mode=False, dataset="healthcare", local_epochs=1)
        cases = [
            ("zkp", "honest", lambda: zkp_case(g, zcfg, h0, lambda s: zkp_library_upload(s, g, h1), n)),
            ("zkp", "A1 raw", lambda: zkp_case(g, zcfg, h0, lambda s: zkp_raw_upload(s, g, a1), n)),
            ("zkp", "A2 raw", lambda: zkp_case(g, zcfg, h0, lambda s: zkp_raw_upload(s, g, a2), n)),
            ("zkp", "A1 lib", lambda: zkp_case(g, zcfg, h0, lambda s: zkp_library_upload(s, g, a1), n)),
            ("he_elgamal_zkp", "honest", lambda: eg_case(g, keys, h0, lambda s: eg_library_upload(keys, s, g, h1), n)),
            ("he_elgamal_zkp", "A1 raw", lambda: eg_case(g, keys, h0, lambda s: eg_raw_upload(keys, s, g, a1), n)),
            ("he_elgamal_zkp", "A2 raw", lambda: eg_case(g, keys, h0, lambda s: eg_raw_upload(keys, s, g, a2), n)),
            ("he_elgamal_zkp", "A1 lib", lambda: eg_case(g, keys, h0, lambda s: eg_library_upload(keys, s, g, a1), n)),
        ]
        print()
        for mode, case, fn in cases:
            report, agg = fn()
            rejected = {k: v[:60] for k, v in (report.get("rejected") or {}).items()}
            if agg is not None:
                moved = np.linalg.norm(flat(agg) - flat(g))
                metrics = "acc/AUPRC %.1f / %.1f, ‖agg − g‖ = %.4f" % (*evaluate(agg, testloader), moved)
            else:
                metrics = "no aggregate"
            print(f"{mode:15s} {case:7s} admitted={report.get('admitted')} rejected={rejected} → {metrics}")
    finally:
        experiment.cleanup_all_procs()


if __name__ == "__main__":
    main()
