"""Step 5 evidence: what ZKPSampledMode._select_layers actually does.

Run from the repository root:

    PYTHONPATH=. python audit/evidence/sampling_evidence.py

Answers audit/sampling.md Q1-Q4 against current code. Model sizes are
reconstructed from the stored baseline upload bytes in each dataset's
comparison_report.json (4 bytes per float32 parameter), so the fractions refer
to the models that produced the stored results.
"""

import json
import os
from collections import OrderedDict

import torch

from fl.core.zkp_gnark import DEFAULT_MAX_LAYER_N, expected_proof_layout
from fl.models.registry import get_model
from fl.privacy.zkp import ZKPSampledMode

SAMPLING_ENV = ("FL_ZKP_SAMPLE_PCT", "FL_ZKP_NUM_LAYERS", "FL_ZKP_SAMPLE_SEED", "FL_ZKP_SELECT_BY")


def with_env(**values):
    """Run _select_layers under exactly these sampling env vars."""
    saved = {k: os.environ.pop(k, None) for k in SAMPLING_ENV}
    os.environ.update({k: str(v) for k, v in values.items()})
    return saved


def restore(saved):
    for k in SAMPLING_ENV:
        os.environ.pop(k, None)
        if saved[k] is not None:
            os.environ[k] = saved[k]


def select(state, **env):
    saved = with_env(**env)
    try:
        return ZKPSampledMode()._select_layers(state)
    finally:
        restore(saved)


def baseline_params(dataset):
    report = json.load(open(f"results/{dataset}/comparison_report.json"))
    [base] = [r for r in report if r["mode"] == "baseline"]
    return int(base["benchmark"]["communication_bytes"]["upload"]["mean"]) // 4


def build_model(dataset, n_params):
    if dataset in ("mnist", "cifar"):
        in_channels, size = (1, 28) if dataset == "mnist" else (3, 32)
        model = get_model("cnn")(num_classes=10, in_channels=in_channels, input_size=size)
    else:
        # TabularNet(hidden [64, 32], 2 classes): n = 64·d + 64 + 64·32 + 32 + 32·2 + 2
        input_dim = (n_params - (64 + 64 * 32 + 32 + 32 * 2 + 2)) // 64
        model = get_model("tabular")(input_dim=input_dim, num_classes=2, hidden_dims=[64, 32])
    assert sum(t.numel() for t in model.state_dict().values()) == n_params, dataset
    return model


print("Q1  k computation: is FL_ZKP_SAMPLE_PCT reachable?")
seven = OrderedDict((f"layer{i}", torch.zeros(i + 1)) for i in range(7))
for env in ({}, {"FL_ZKP_SAMPLE_PCT": "0.5"}, {"FL_ZKP_SAMPLE_PCT": "1.0"}, {"FL_ZKP_NUM_LAYERS": "3"}):
    print(f"    env={env!s:32} -> {len(select(seven, **env))} of 7 layers selected")

print("\nQ2  select_by default and the RNG")
twenty = OrderedDict((f"layer{i}", torch.zeros(i + 1)) for i in range(20))
default_runs = {tuple(select(twenty)) for _ in range(20)}
seeded = {s: select(twenty, FL_ZKP_SAMPLE_SEED=s) for s in (1, 2, 3)}
random_runs = {tuple(select(twenty, FL_ZKP_SELECT_BY="random")) for _ in range(20)}
print(f"    default (select_by=size): {len(default_runs)} distinct selection(s) over 20 calls: {sorted(default_runs)}")
print(f"    default with seeds 1,2,3: {seeded}")
print(f"    FL_ZKP_SELECT_BY=random:  {len(random_runs)} distinct selection(s) over 20 calls")

print("\nQ3  default selection per dataset")
print(f"    {'dataset':10} {'params':>7} {'selected layer':16} {'layer share':>11} {'proofs sel/full':>15} {'proven elems share':>18}")
for dataset in ("healthcare", "creditcard", "stock", "mnist", "cifar"):
    n = baseline_params(dataset)
    model = build_model(dataset, n)
    state = model.state_dict()
    [chosen] = select(state)
    schema = [(k, tuple(v.shape)) for k, v in state.items()]
    full = expected_proof_layout(schema)
    sel = expected_proof_layout([(chosen, tuple(state[chosen].shape))])
    share = state[chosen].numel() / n
    print(f"    {dataset:10} {n:7d} {chosen:16} {share:11.1%} {len(sel):>7}/{len(full):<7} {share:18.1%}")
print(f"    (chunking: layers above {DEFAULT_MAX_LAYER_N} elements are proved in {DEFAULT_MAX_LAYER_N}-element chunks)")

print("\nQ4  predictability")
model = build_model("healthcare", baseline_params("healthcare"))
state = model.state_dict()
before = select(state)
with torch.no_grad():
    for name, tensor in state.items():
        if name not in before:
            tensor.mul_(1e3)  # poison every layer that is not selected
after = select(state)
print(f"    selected before training: {before}; after poisoning every unselected layer: {after}")
print(f"    selection depends only on layer sizes: {before == after}")
