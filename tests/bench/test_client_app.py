"""The ClientApp: rebuilding a client per message and carrying state between rounds."""

import json

import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict
from torch.utils.data import DataLoader, TensorDataset


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
