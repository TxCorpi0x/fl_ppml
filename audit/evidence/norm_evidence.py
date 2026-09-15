"""Step 7 evidence: the ZKP norm bound applies to weights, not updates.

Run from the repository root (needs the built gnark service and the pinned
production keys; starts the prover and verifier itself):

    PYTHONPATH=. python audit/evidence/norm_evidence.py

Measures on healthcare (2 clients, seed 42, harness defaults lr 0.001,
momentum 0.9, batch 16):
  N1  honest weight and update norms, per chunk and whole-model
  N2  what the policy admits: per-chunk plaintext bound, ElGamal shares
  A1  boosted model replacement: attacker trains on flipped labels and scales
      its update by the number of clients so FedAvg lands on its model
  A2  the largest update the bound admits: every chunk pushed to the bound
      in the direction opposite to the global weights
Both attacks go through the real server admission path of `zkp` and
`he_elgamal_zkp`, and the aggregate is evaluated on the test set.

No source files are modified.
"""

import copy
import tempfile
from types import SimpleNamespace as NS

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import average_precision_score

from fl.compare import experiment
from fl.config import FLConfig
from fl.core import elgamal_gnark as eg
from fl.core import zkp_gnark as zg
from fl.core.engine import test, train
from fl.datasets import get_dataset_loader
from fl.models.registry import get_model_for_batch

SEED, CLIENTS, LR, BATCH = 42, 2, 0.001, 16


def flat(net):
    return np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in net.state_dict().values()])


def set_flat(net, vec):
    out, off = copy.deepcopy(net), 0
    sd = out.state_dict()
    for k, v in sd.items():
        n = v.numel()
        sd[k] = torch.tensor(vec[off : off + n], dtype=v.dtype).reshape(v.shape)
        off += n
    out.load_state_dict(sd)
    return out


def chunk_norms(net, n):
    """Norm of each unit the plaintext circuit bounds separately (layer or chunk)."""
    norms = []
    for v in net.state_dict().values():
        a = v.detach().cpu().numpy().reshape(-1)
        norms += [np.linalg.norm(a[i : i + n]) for i in range(0, a.size, n)]
    return np.array(norms)


def local_train(net, loader, epochs, flip=False):
    net = copy.deepcopy(net)
    if flip:
        loader = [(x, 1 - y) for x, y in loader]
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss(net(x), y).backward()
            opt.step()
    return net


def evaluate(net, testloader):
    _, acc, _, y_true, y_proba = test(net, testloader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    return acc, 100 * average_precision_score(y_true, np.asarray(y_proba)[:, 1])


def fit(params, metrics, n):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), n, metrics)


def run_zkp(global_net, uploads):
    from fl.privacy.zkp import ZKPMode

    server = ZKPMode()
    server.bind_server_model(None, global_net)
    results = []
    for cid, net, n in uploads:
        client = ZKPMode()
        try:
            params = client.send_parameters(net, {"backend": "gnark"}, sim_mode=False)
        except RuntimeError as exc:
            print(f"      zkp client {cid}: cannot prove ({str(exc)[:80]})")
            continue
        results.append((NS(cid=cid), fit(params, client.post_fit_metrics(None), n)))
    params, out = server.aggregate_fit_override(1, results, [], None, NS(zkp_backend="gnark", sim_mode=False, min_fit_clients=1))
    report = server.last_round_report
    agg = set_flat(global_net, np.concatenate([p.reshape(-1) for p in parameters_to_ndarrays(params)])) if params else None
    return report, agg


def run_elgamal(global_net, uploads, keys):
    from fl.privacy.he_elgamal_zkp import HeElGamalZKPMode

    server = HeElGamalZKPMode()
    sctx = server.setup_server_context(keys)
    server.bind_server_model(sctx, global_net)
    results = []
    for cid, net, n in uploads:
        client = HeElGamalZKPMode()
        ctx = client.setup_client_context(keys)
        client.on_fit_config(ctx, {"server_round": 1})
        try:
            params = client.send_parameters(net, ctx, sim_mode=False)
        except (RuntimeError, ValueError) as exc:
            print(f"      he_elgamal_zkp client {cid}: cannot prove ({str(exc)[:80]})")
            continue
        results.append((NS(cid=cid), fit(params, client.post_fit_metrics(ctx), n)))
    params, out = server.aggregate_fit_override(1, results, [], sctx, keys)
    anchor = getattr(server, "_last_anchor_data", None) or {}
    report = getattr(server, "last_round_report", None) or {"admitted": anchor.get("client_ids", []), "rejected": {}, **(out or {})}
    agg = None
    if params is not None:
        reader = HeElGamalZKPMode()
        rctx = reader.setup_client_context(keys)
        agg = copy.deepcopy(global_net)
        reader.receive_parameters(agg, parameters_to_ndarrays(params), rctx, sim_mode=False)
    return report, agg


def admissible_elgamal(net, total):
    policy = eg.Policy.from_env()
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    schema = eg.schema_of(state)
    for chunk in eg.chunks_for(schema, policy.chunk_size):
        v = state[schema[chunk.layer][0]].reshape(-1)[chunk.start : chunk.start + chunk.size]
        q = np.round(v * policy.scale)
        if np.any(np.abs(q) >= eg.VALUE_LIMIT) or int(np.sum(q.astype(object) ** 2)) > eg.chunk_bound_sq(policy, chunk, total):
            return False
    return True


def main():
    config = FLConfig(dataset="healthcare", num_clients=CLIENTS, seed=SEED, batch_size=BATCH, sim_mode=False)
    Loader = get_dataset_loader("healthcare")
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, _, testloader = Loader().load(config)
    torch.manual_seed(SEED)
    g = get_model_for_batch(next(iter(testloader)), config.num_classes)
    n_chunk, B = zg.norm_chunk_n(), zg.DEFAULT_MAX_NORM
    total = flat(g).size

    print(f"model: {total} parameters; norm circuit chunk n={n_chunk}; FL_ZKP_MAX_NORM={B}; "
          f"plaintext scale {zg.DEFAULT_SCALE:g}; ElGamal scale {eg.Policy.from_env().scale}, "
          f"chunk {eg.Policy.from_env().chunk_size}")
    print(f"units bounded separately by the plaintext circuit: {len(chunk_norms(g, n_chunk))}")

    print("\nN1  honest norms")
    print(f"    global (init, seed {SEED}): ||w||={np.linalg.norm(flat(g)):.3f}, max unit {chunk_norms(g, n_chunk).max():.3f}")
    honest = {}
    for epochs in (1, 5):
        for c in range(CLIENTS):
            w = local_train(g, trainloaders[c], epochs)
            d = flat(w) - flat(g)
            honest[(epochs, c)] = w
            print(f"    client {c}, {epochs} epoch(s) from init: ||w||={np.linalg.norm(flat(w)):.3f}, "
                  f"max unit ||w||={chunk_norms(w, n_chunk).max():.3f}, ||Δ||={np.linalg.norm(d):.4f}, max|Δ_i|={np.abs(d).max():.4f}")
    trained = local_train(local_train(g, trainloaders[0], 5), trainloaders[1], 5)
    for c in range(CLIENTS):
        d = flat(local_train(trained, trainloaders[c], 1)) - flat(trained)
        print(f"    client {c}, 1 epoch from a 10-epoch model: ||Δ||={np.linalg.norm(d):.4f}")

    print("\nN2  what the policy admits")
    print(f"    plaintext zkp: every unit ||w_unit|| ≤ {B} → whole model ||w|| ≤ {B}·√units = {B * np.sqrt(len(chunk_norms(g, n_chunk))):.1f}")
    print(f"    he_elgamal_zkp: shares sum to ||w|| ≤ {B}; per coordinate |w| < {eg.VALUE_LIMIT / eg.Policy.from_env().scale:.1f}")
    print(f"    largest admissible ||Δ|| = ||w_global|| + bound (Δ = −g·(1 + bound/||g||)): "
          f"elgamal {B + np.linalg.norm(flat(g)):.1f}")

    h0, h1 = honest[(1, 0)], honest[(1, 1)]
    bad = local_train(g, trainloaders[1], 5, flip=True)
    a1 = set_flat(g, flat(g) + CLIENTS * (flat(bad) - flat(g)))
    # A2: push each plaintext unit to 0.99·B opposite to g; for ElGamal scale the
    # whole vector to 0.99 of its shares (uniform over coordinates).
    a2_plain = flat(g).copy()
    off = 0
    for v in g.state_dict().values():
        for i in range(0, v.numel(), n_chunk):
            seg = a2_plain[off + i : off + min(i + n_chunk, v.numel())]
            a2_plain[off + i : off + min(i + n_chunk, v.numel())] = -0.99 * B * seg / np.linalg.norm(seg)
        off += v.numel()
    a2_eg = -flat(g) / np.abs(flat(g)).max()  # sign pattern, then fit the smallest share
    a2_eg = 0.99 * (B / np.sqrt(total)) * np.sign(a2_eg)
    a2_plain, a2_eg = set_flat(g, a2_plain), set_flat(g, a2_eg)

    for label, net in (("A1 boosted replacement", a1), ("A2 max plaintext", a2_plain), ("A2 max elgamal", a2_eg)):
        d = flat(net) - flat(g)
        print(f"\n{label}: ||w||={np.linalg.norm(flat(net)):.2f}, max unit {chunk_norms(net, n_chunk).max():.2f}, "
              f"||Δ||={np.linalg.norm(d):.2f} ({np.linalg.norm(d) / np.linalg.norm(flat(h0) - flat(g)):.0f}× honest 1-epoch), "
              f"plaintext-admissible={bool(chunk_norms(net, n_chunk).max() <= B)}, elgamal-admissible={admissible_elgamal(net, total)}")

    print("\nReference: FedAvg(honest, honest)  acc/AUPRC = %.1f / %.1f" % evaluate(set_flat(g, (flat(h0) + flat(h1)) / 2), testloader))
    print("           attacker's model alone   acc/AUPRC = %.1f / %.1f" % evaluate(bad, testloader))

    if not experiment._ensure_gnark_service(tempfile.mkdtemp()):
        raise SystemExit("gnark services unavailable")
    try:
        from fl.keys.he_elgamal import generate

        d = tempfile.mkdtemp()
        generate(secret_path=f"{d}/s.json", public_path=f"{d}/p.json")
        keys = NS(sim_mode=False, he_elgamal_secret_path=f"{d}/s.json", he_elgamal_public_path=f"{d}/p.json", min_fit_clients=1)
        n0, n1 = len(trainloaders[0].dataset), len(trainloaders[1].dataset)
        cases = [
            ("zkp", "A1", lambda: run_zkp(g, [("honest", h0, n0), ("attacker", a1, n1)])),
            ("zkp", "A2", lambda: run_zkp(g, [("honest", h0, n0), ("attacker", a2_plain, n1)])),
            ("he_elgamal_zkp", "A1", lambda: run_elgamal(g, [("honest", h0, n0), ("attacker", a1, n1)], keys)),
            ("he_elgamal_zkp", "A2", lambda: run_elgamal(g, [("honest", h0, n0), ("attacker", a2_eg, n1)], keys)),
        ]
        print()
        for mode, attack, fn in cases:
            report, agg = fn()
            admitted = report.get("admitted") if isinstance(report, dict) else report
            metrics = "%.1f / %.1f" % evaluate(agg, testloader) if agg is not None else "no aggregate"
            print(f"{mode:15s} {attack}: admitted={admitted} rejected={list((report or {}).get('rejected', {}))} → acc/AUPRC {metrics}")
    finally:
        experiment.cleanup_all_procs()


if __name__ == "__main__":
    main()
