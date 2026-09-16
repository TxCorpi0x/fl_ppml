"""The Flower App manifest, the run config it declares, and the launcher that submits it."""

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


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
