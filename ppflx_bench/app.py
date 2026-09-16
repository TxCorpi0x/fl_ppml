"""
The Flower App: the ServerApp and ClientApp this project runs.

``ppflx.server:server_app`` builds an FLConfig from the run config, loads the
server's test data and runs the FedPrivate strategy. ``ppflx.client:client_app``
rebuilds a FlowerClient for every message from the run config, the node config
(``partition-id``) and the node's saved state, calls the matching operation and
replies with Flower records.

Both are declared as the app's components in pyproject.toml and started by
ppflx_bench.launch; the strategy and the client itself live in ppflx.server and ppflx.client.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp

from ppflx.client import FlowerClient, make_client
from ppflx.config import FLConfig, apply_process_env
from ppflx.records import config_record, metric_record, pack_state, unpack_state
from ppflx.server import make_strategy


server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """Run one experiment from the run config; write benchmark.json to its results-dir."""
    from ppflx.core.benchmark import init_benchmark
    from ppflx_bench.datasets import get_dataset_loader
    from ppflx.privacy import get_privacy_mode

    config = FLConfig.from_run_config(context.run_config)
    apply_process_env(config)

    Loader = get_dataset_loader(config.dataset)
    config.num_classes = Loader.get_spec().num_classes
    # Clients partition the same data with the same seed; the largest shard's
    # batch count sizes the ZKP update-norm bound (ppflx/core/update_bound.py).
    trainloaders, _, testloader = Loader().load(config)

    mode = get_privacy_mode(config.privacy_mode)
    benchmark = (
        init_benchmark(
            config.privacy_mode,
            config.num_clients,
            config.num_rounds,
            transport="simulated" if config.sim_mode else "network",
            zkp_backend=config.zkp_backend if "zkp" in config.privacy_mode else None,
        )
        if config.benchmark
        else None
    )
    strategy = make_strategy(
        config, mode, testloader, benchmark=benchmark, client_batches=max(len(t) for t in trainloaders)
    )

    print(f"Starting ServerApp [{config.privacy_mode}]")
    strategy.run(grid)

    if benchmark:
        os.makedirs(config.results_dir, exist_ok=True)
        bench_path = os.path.join(config.results_dir, "benchmark.json")
        benchmark.save(bench_path)
        benchmark.print_summary()


client_app = ClientApp()

# Context.state keys. A SuperNode keeps a node's context for the whole run but
# handles every message in a fresh process, so whatever a mode needs in a later
# Flower round has to be stored here.
_STATE = "ppflx.client.state"
_STATE_ARRAYS = "ppflx.client.state.arrays"
_BENCHMARK = "ppflx.client.benchmark"

# Crypto-context entries that carry over between messages: the commit–challenge
# commitment (the committed update and, for ElGamal, its encryption randomness).
PERSISTED_CONTEXT_KEYS = ("commitment",)


def _load_data(config: FLConfig):
    from ppflx_bench.datasets import get_dataset_loader

    Loader = get_dataset_loader(config.dataset)
    config.num_classes = Loader.get_spec().num_classes
    trainloaders, valloaders, _ = Loader().load(config)
    return trainloaders, valloaders


def _client_for(context: Context) -> FlowerClient:
    """Rebuild this node's client from the run config, node config and saved state."""
    from ppflx.core.benchmark import BenchmarkMetrics, init_benchmark
    from ppflx.privacy import get_privacy_mode

    config = FLConfig.from_run_config(context.run_config)
    apply_process_env(config)
    partition = int(context.node_config["partition-id"])
    partitions = int(context.node_config.get("num-partitions", config.num_clients))
    if partitions != config.num_clients or not 0 <= partition < partitions:
        raise ValueError(
            f"node partition {partition} of {partitions} does not match num-clients={config.num_clients}"
        )
    # The server sends the model in every message; never start from a checkpoint.
    config.model_save = ""

    benchmark = None
    if config.benchmark:
        if _BENCHMARK in context.state:
            benchmark = BenchmarkMetrics(**json.loads(context.state[_BENCHMARK]["raw"]))
        else:
            benchmark = init_benchmark(
                config.privacy_mode,
                config.num_clients,
                config.num_rounds,
                transport="simulated" if config.sim_mode else "network",
                zkp_backend=config.zkp_backend if "zkp" in config.privacy_mode else None,
            )

    trainloaders, valloaders = _load_data(config)
    client = make_client(str(partition), trainloaders, valloaders, get_privacy_mode(config.privacy_mode), config, benchmark)
    if _STATE in context.state and isinstance(client.crypto_ctx, dict):
        client.crypto_ctx.update(unpack_state(context.state[_STATE], context.state[_STATE_ARRAYS]))
    return client


def _save(context: Context, client: FlowerClient) -> None:
    """Store carried-over state in the context; write this client's benchmark file."""
    if isinstance(client.crypto_ctx, dict):
        kept = {k: client.crypto_ctx[k] for k in PERSISTED_CONTEXT_KEYS if k in client.crypto_ctx}
        context.state[_STATE], context.state[_STATE_ARRAYS] = pack_state(kept)
    if client.benchmark is not None:
        context.state[_BENCHMARK] = ConfigRecord({"raw": json.dumps(asdict(client.benchmark))})
        os.makedirs(client.config.results_dir, exist_ok=True)
        path = os.path.join(client.config.results_dir, f"client_{client.cid}_benchmark.json")
        with open(path, "w") as f:
            json.dump(client.benchmark.summary(), f, indent=2)


@client_app.train()
def train_handler(msg: Message, context: Context) -> Message:
    client = _client_for(context)
    params, num_examples, metrics = client.fit(msg.content["arrays"].to_numpy_ndarrays(), dict(msg.content["config"]))
    _save(context, client)
    content = RecordDict(
        {
            "arrays": ArrayRecord.from_numpy_ndarrays(params),
            "fit_metrics": config_record(metrics),
            "metrics": MetricRecord({"num-examples": int(num_examples)}),
        }
    )
    return Message(content, reply_to=msg)


@client_app.evaluate()
def evaluate_handler(msg: Message, context: Context) -> Message:
    client = _client_for(context)
    loss, num_examples, metrics = client.evaluate(msg.content["arrays"].to_numpy_ndarrays(), dict(msg.content["config"]))
    _save(context, client)
    content = RecordDict({"metrics": metric_record({"loss": float(loss), "num-examples": int(num_examples), **metrics})})
    return Message(content, reply_to=msg)


@client_app.query()
def query_handler(msg: Message, context: Context) -> Message:
    """Initial parameters for modes whose round-1 download must already be encrypted."""
    client = _client_for(context)
    params = client.get_parameters({})
    _save(context, client)
    return Message(RecordDict({"arrays": ArrayRecord.from_numpy_ndarrays(params)}), reply_to=msg)
