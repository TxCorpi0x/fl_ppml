"""The CKKS/TFHE + DP composites prove updates without enforcing the update bound.

Their proofs aren't bound to the aggregated ciphertext, so a bound adds no
integrity, and DP noise would get every honest update clipped.
"""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from tests.conftest import requires_gnark, use_gnark


class _Model(torch.nn.Module):
    def __init__(self, offset=0.0):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        with torch.no_grad():
            self.model[0].weight.copy_(torch.tensor([[0.1, -0.2], [0.3, -0.4]]) + offset)
            self.model[0].bias.copy_(torch.tensor([0.05, -0.05]) + offset)


CONFIG = NS(zkp_backend="gnark", sim_mode=False, min_fit_clients=1, dataset="not-calibrated", local_epochs=1)


@pytest.mark.parametrize("mode_name", ["he_tenseal_zkp_dp", "he_concrete_tfhe_zkp_dp"])
def test_dp_composites_set_no_bound(monkeypatch, mode_name):
    from fl.privacy import get_privacy_mode

    monkeypatch.delenv("FL_ZKP_MAX_NORM", raising=False)
    zkp = get_privacy_mode(mode_name)._zkp_mode
    zkp.setup_server_context(CONFIG)  # would raise for an uncalibrated dataset if the bound were enforced
    assert zkp.fit_config(1) == {}
    ctx = {"backend": "gnark"}
    zkp.on_fit_config(ctx, {"server_round": 1})  # no bound sent, and the client doesn't refuse
    assert ctx["max_update_norm"] is None


def test_non_dp_composites_still_enforce_the_bound(monkeypatch):
    from fl.privacy import get_privacy_mode

    monkeypatch.delenv("FL_ZKP_MAX_NORM", raising=False)
    with pytest.raises(RuntimeError, match="no calibrated"):
        get_privacy_mode("he_tenseal_zkp")._zkp_mode.setup_server_context(CONFIG)


@requires_gnark
def test_unbounded_zkp_proves_and_admits_a_large_update_unclipped(monkeypatch, gnark):
    from fl.privacy.zkp import ZKPMode

    use_gnark(monkeypatch, gnark)
    monkeypatch.delenv("FL_ZKP_MAX_NORM", raising=False)
    server = ZKPMode(enforce_update_bound=False)
    server.setup_server_context(CONFIG)
    server.bind_server_model(None, _Model())

    client, ctx = ZKPMode(enforce_update_bound=False), {"backend": "gnark"}
    client.on_fit_config(ctx, {"server_round": 1, **server.fit_config(1)})
    client.receive_parameters(_Model(), list(server._global), ctx, sim_mode=False)
    params = client.send_parameters(_Model(offset=100.0), ctx, sim_mode=False)
    metrics = client.post_fit_metrics(ctx)

    assert metrics["zkp_update_clipped"] == 0 and metrics["zkp_update_norm"] > 199
    np.testing.assert_allclose(params[0], _Model(offset=100.0).model[0].weight.detach().numpy())
    fit_res = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(params), 10, metrics)
    server.aggregate_fit_override(1, [(NS(cid="c"), fit_res)], [], None, CONFIG)
    assert server.last_round_report["admitted"] == ["c"]
    assert len(json.loads(metrics["zkp_proofs_json"])) == 2
