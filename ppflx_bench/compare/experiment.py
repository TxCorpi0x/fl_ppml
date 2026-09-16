"""
Experiment runner — simulation and distributed modes.

These functions are the engine behind the comparison framework.
They are invoked by ppflx_bench.compare.runner.run_comparison() and should not
normally be called directly.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

from ppflx_bench.compare.benchmark import (
    aggregate_client_benchmarks,
    merge_server_and_clients,
)
from ppflx_bench.compare.registry import ModeConfig


# ─────────────────────────────────────────────────────────────────────────────
# Global process registry — lets Ctrl+C kill all children
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_PROCS: list = []  # List[subprocess.Popen]


def _register_proc(proc: subprocess.Popen) -> subprocess.Popen:
    _ACTIVE_PROCS.append(proc)
    return proc


def _unregister_proc(proc: subprocess.Popen) -> None:
    try:
        _ACTIVE_PROCS.remove(proc)
    except ValueError:
        pass


def cleanup_all_procs() -> None:
    """Kill every tracked subprocess and its process group (best-effort)."""
    for proc in list(_ACTIVE_PROCS):
        try:
            if proc.poll() is None:  # still running
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
    _ACTIVE_PROCS.clear()


# ─────────────────────────────────────────────────────────────────────────────
# gnark ZKP service lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_GNARK_PROCS: Dict[str, subprocess.Popen] = {}

# Role → (env var, default URL). Clients prove against the prover; the server
# verifies against a separate verifier that holds only verifying keys.
_GNARK_ROLES = {
    "prover": ("FL_ZKP_PROVER_URL", "http://127.0.0.1:9000"),
    "verifier": ("FL_ZKP_VERIFIER_URL", "http://127.0.0.1:9001"),
}


def _gnark_url(role: str) -> str:
    env_var, default = _GNARK_ROLES[role]
    return os.environ.get(env_var, default).rstrip("/")


def _gnark_port(role: str) -> int:
    try:
        return int(_gnark_url(role).rsplit(":", 1)[-1])
    except ValueError:
        return int(_GNARK_ROLES[role][1].rsplit(":", 1)[-1])


def _gnark_health(role: str) -> Optional[dict]:
    """The service's /health JSON, or None if nothing healthy answers."""
    try:
        with urllib.request.urlopen(_gnark_url(role) + "/health", timeout=2) as r:
            return json.loads(r.read()) if r.status == 200 else None
    except Exception:
        return None


def _gnark_matches_pin(role: str, health: Optional[dict]) -> bool:
    """True only if a service answers in this role under the pinned manifest.

    A legacy service (per-process keys, no role) or one started from other keys
    fails this check and is replaced rather than reused.
    """
    from ppflx.core.gnark_keys import manifest_sha256

    return bool(health) and health.get("role") == role and health.get("manifest_sha256") == manifest_sha256()


def _kill_port(port: int) -> None:
    """Kill any process listening on the given TCP port (macOS/Linux, best-effort)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [p for p in result.stdout.strip().split() if p]
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        if pids:
            time.sleep(1)
    except Exception:
        pass


def _needs_gnark(mode_cfg: ModeConfig) -> bool:
    """Return True if this mode requires the gnark ZKP gRPC service."""
    if mode_cfg.internal_mode not in ("zkp", "he_zkp", "he_zkp_dp"):
        return False
    return os.environ.get("FL_ZKP_BACKEND", "gnark").lower() == "gnark"


def _ensure_gnark_service(log_dir: str) -> bool:
    """Start the gnark prover and verifier services under the pinned keys.

    Build order:
      1. If a pre-built binary ``zkp_gnark_service/gnark_service`` exists, use it.
      2. Otherwise, run ``go build -o gnark_service .`` in that directory.

    A running service is reused only if it reports the expected role and the
    pinned manifest hash; anything else on the port is killed and replaced.
    Returns True only if both roles are up; ZKP modes must not run otherwise.
    """
    from ppflx.core.gnark_keys import keys_dir, load_manifest, missing_proving_keys, pk_dir

    try:
        load_manifest()
        missing = missing_proving_keys()
    except (OSError, ValueError, KeyError) as exc:
        print(f"[gnark] [ERROR] Pinned ZKP keys unusable: {exc}")
        return False
    if missing:
        print(
            f"[gnark] [ERROR] Proving keys {missing} not in {pk_dir()}. They are not committed; "
            f"regenerate with: zkp_gnark_service/gnark_service setup --keys-dir {keys_dir()} "
            f"--pk-dir {pk_dir()} --force (this re-pins the verifying keys, commit the result)"
        )
        return False

    binary = _gnark_binary()
    if binary is None:
        return False
    return all(_ensure_gnark_role(role, binary, log_dir) for role in _GNARK_ROLES)


def _gnark_binary() -> Optional[str]:
    # Locate service directory relative to this file:
    # ppflx_bench/compare/experiment.py → ../../zkp_gnark_service/
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    svc_dir = os.path.join(repo_root, "zkp_gnark_service")
    binary = os.path.join(svc_dir, "gnark_service")

    if not os.path.exists(binary):
        needs_build = True
    else:
        # Rebuild if any Go source file is newer than the binary
        binary_mtime = os.path.getmtime(binary)
        needs_build = any(
            os.path.getmtime(os.path.join(svc_dir, f)) > binary_mtime
            for f in os.listdir(svc_dir)
            if f.endswith(".go")
        )
        if needs_build:
            print("[gnark] Source files changed — rebuilding binary...")

    if needs_build:
        if shutil.which("go") is None:
            print(
                "[gnark] [ERROR] 'go' not in PATH and no pre-built binary found — cannot start gnark service."
            )
            return None
        print(f"[gnark] Building gnark service binary...")
        result = subprocess.run(
            ["go", "build", "-o", "gnark_service", "."],
            cwd=svc_dir,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(f"[gnark] [ERROR] Build failed:\n{result.stderr}")
            return None
        print("[gnark] Build OK.")
    return binary


def _ensure_gnark_role(role: str, binary: str, log_dir: str) -> bool:
    from ppflx.core.gnark_keys import keys_dir, pk_dir

    health = _gnark_health(role)
    if _gnark_matches_pin(role, health):
        print(f"[gnark] {role} already healthy under the pinned manifest — reusing it.")
        return True

    port = _gnark_port(role)
    if health is not None:
        print(f"[gnark] Service on port {port} is not a pinned {role} ({health}) — replacing it.")
    _kill_port(port)
    time.sleep(0.5)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"gnark_{role}.log")
    cmd = [binary, "serve", "--role", role, "--keys-dir", str(keys_dir()), "--port", str(port)]
    if role == "prover":
        cmd += ["--pk-dir", str(pk_dir())]  # the verifier never gets proving keys

    with open(log_path, "w") as log:
        proc = _register_proc(
            subprocess.Popen(
                cmd,
                cwd=os.path.dirname(binary),
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        )
    _GNARK_PROCS[role] = proc
    print(f"[gnark] {role} starting on port {port} (log: {log_path})")

    # Loading the prover's proving keys takes tens of seconds at production sizes.
    for i in range(180):
        time.sleep(1)
        health = _gnark_health(role)
        if health is not None:
            if not _gnark_matches_pin(role, health):
                print(f"[gnark] [ERROR] {role} came up with an unexpected manifest: {health}")
                return False
            print(f"[gnark] [OK] {role} healthy after {i + 1}s")
            return True
        if proc.poll() is not None:
            print(f"[gnark] [ERROR] {role} exited early (rc={proc.returncode}). Check {log_path}")
            return False

    print(f"[gnark] [ERROR] Timeout waiting for gnark {role} to become healthy.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _grpc_env(base: Dict[str, str]) -> Dict[str, str]:
    """Add gRPC keepalive env vars to a copy of ``base``."""
    env = base.copy()
    env["GRPC_ARG_KEEPALIVE_TIME_MS"] = "300000"
    env["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"] = "120000"
    env["GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "1"
    env["GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"
    # FL_ENCRYPT_LAYERS is inherited from ``base`` only when the user sets it.
    # Injecting a first-layer default here silently disabled full-model
    # encryption in every harness run.
    env["FL_CLIENT_TIMEOUT"] = os.environ.get("FL_CLIENT_TIMEOUT", "7200")
    return env


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Flower runs (SuperLink + SuperNodes, or the Simulation Runtime)
# ─────────────────────────────────────────────────────────────────────────────

# Harness argument → Flower run config key (pyproject.toml [tool.flwr.app.config]).
_RUN_CONFIG_KEYS = {
    "dataset": "dataset",
    "data_path": "data-path",
    "max_epochs": "local-epochs",
    "batch_size": "batch-size",
    "lr": "learning-rate",
    "device": "device",
    "number_clients": "num-clients",
    "rounds": "num-rounds",
    "dirichlet_alpha": "dirichlet-alpha",
    "dp_epsilon": "dp-epsilon",
    "seed": "seed",
    "chain_backend": "chain-backend",
    "chain_ledger_path": "chain-ledger-path",
}


def run_config_for(display_mode: str, base_args: Dict[str, Any], result_dir: str, simulation: bool) -> Dict[str, Any]:
    """The Flower run config for one mode of a comparison run."""
    given = {k: v for k, v in base_args.items() if v is not None and v != ""}
    unknown = sorted(set(given) - set(_RUN_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"arguments without a run config key: {unknown}")
    run_config = {_RUN_CONFIG_KEYS[k]: v for k, v in given.items()}
    num_clients = int(run_config.get("num-clients", 3))
    result_dir = os.path.abspath(result_dir)
    run_config.update(
        {
            # Select the registered mode by key: flag combinations alone can't
            # name modes such as zkp_sampled.
            "mode": display_mode,
            "num-clients": num_clients,
            # Start round 1 only once every client is connected (FedPrivate waits
            # for min-avail-clients before sizing its sample). Otherwise a late
            # client misses round 1: for commit–challenge modes it then has
            # nothing to answer the challenge with, and every mode runs its first
            # round with fewer clients.
            "min-avail-clients": num_clients,
            "results-dir": result_dir,
            "model-save": os.path.join(result_dir, "server_model.pt"),
            "benchmark": True,
            "sim-mode": simulation,
        }
    )
    return run_config


def round_failures(benchmark: Optional[Dict]) -> List[str]:
    """Rounds that did not complete cleanly, for every mode.

    A ClientApp that crashes (e.g. a native abort in an HE library) leaves its
    SuperNode running, so the run itself still completes; the ServerApp only
    records the missing reply. Such a run is not a successful benchmark.
    """
    problems = []
    for outcome in (benchmark or {}).get("round_outcomes") or []:
        if outcome.get("outcome") not in ("aggregated", "committed") or outcome.get("flower_failures"):
            problems.append(
                f"round {outcome.get('round')}: outcome={outcome.get('outcome')} "
                f"flower_failures={outcome.get('flower_failures', 0)}"
            )
    return problems


def _run_timeout(mode_cfg: ModeConfig, simulation: bool) -> int:
    env_timeout = os.environ.get("FL_SERVER_TIMEOUT", "").strip()
    if env_timeout:
        return int(env_timeout)
    if simulation:
        return mode_cfg.timeout_s
    env_client_timeout = os.environ.get("FL_CLIENT_TIMEOUT", "").strip()
    if env_client_timeout:
        client_timeout = int(env_client_timeout)
    else:
        # Heavy crypto modes can run for multiple hours on CIFAR; default to 6h.
        mode_l = str(mode_cfg.internal_mode).lower()
        client_timeout = 21600 if ("zkp" in mode_l or mode_l.startswith("he")) else 7200
    # 30 min headroom for final evaluation, checkpointing and shutdown.
    return client_timeout + 1800


def _run_federation(
    mode_cfg: ModeConfig,
    display_mode: str,
    base_args: Dict[str, Any],
    result_dir: str,
    simulation: bool,
) -> Dict:
    """Run one mode on a local SuperLink, with SuperNode processes unless simulating."""
    from ppflx_bench.launch import Federation

    run_config = run_config_for(display_mode, base_args, result_dir, simulation)
    num_clients = run_config["num-clients"]
    # A run still active this long after every SuperNode exited is stuck: the
    # ServerApp would otherwise wait for replies that can no longer arrive.
    grace = int(os.environ.get("FL_SERVER_GRACE", "600"))
    start = datetime.now()
    status, details, client_failures = "not_started", "", 0
    try:
        with Federation(result_dir, num_clients, simulation=simulation, env=_grpc_env({})) as federation:
            run_id = federation.submit(run_config)
            where = "Simulation Runtime" if simulation else f"{num_clients} SuperNodes"
            print(f"[OK] Run {run_id} submitted ({where}); logs in {result_dir}")
            status, details = federation.wait(run_id, timeout=_run_timeout(mode_cfg, simulation), grace=grace)
            client_failures = federation.client_failures()
            federation.save_app_logs(run_id)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        details = str(exc)
        print(f"[ERROR] {display_mode}: {exc}")
    duration = (datetime.now() - start).total_seconds()
    if details:
        print(f"  status={status}: {details}")

    # ── merge benchmarks ──────────────────────────────────────────────────
    bm_path = os.path.join(result_dir, "benchmark.json")
    server_bm = _load_json(bm_path)
    client_bms = [
        bm
        for i in range(num_clients)
        if (bm := _load_json(os.path.join(result_dir, f"client_{i}_benchmark.json"))) is not None
    ]
    client_agg = aggregate_client_benchmarks(client_bms)
    if client_bms:
        print(f"[OK] Aggregated {len(client_bms)} client benchmarks")

    benchmark = merge_server_and_clients(server_bm, client_agg)

    # Fix the mode name (the server writes the internal mode name, e.g. "zkp"
    # for zkp_sampled) and write the merged benchmark back to benchmark.json so
    # the on-disk file reflects client-side metrics (e.g. proof_generation).
    if benchmark is not None:
        benchmark["mode"] = display_mode
        with open(bm_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"[OK] Merged benchmark written → {bm_path}")

    completed = status == "finished:completed"
    problems = round_failures(benchmark)
    for problem in problems:
        print(f"  [FAIL] {problem}")
    success = completed and client_failures == 0 and benchmark is not None and not problems

    print(f"{display_mode.upper()} {'[OK] SUCCESS' if success else '[FAIL] FAILED'} ({duration:.1f}s)")
    if client_failures:
        print(f"  ({client_failures}/{num_clients} clients failed)")

    return {
        "mode": display_mode,
        "success": success,
        "status": status,
        "exit_code": 0 if completed else 1,
        "duration": duration,
        "result_dir": result_dir,
        "benchmark": benchmark,
        "client_failures": client_failures,
    }


def run_simulation(
    mode_cfg: ModeConfig,
    display_mode: str,
    base_args: Dict[str, Any],
    result_dir: str,
) -> Dict:
    """Run one mode on Flower's Simulation Runtime (HE modes transport plaintext)."""
    return _run_federation(mode_cfg, display_mode, base_args, result_dir, simulation=True)


def run_distributed(
    mode_cfg: ModeConfig,
    display_mode: str,
    base_args: Dict[str, Any],
    result_dir: str,
) -> Dict:
    """Run one mode with a SuperLink and one SuperNode process per client."""
    return _run_federation(mode_cfg, display_mode, base_args, result_dir, simulation=False)


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(
    mode_cfg: ModeConfig,
    display_mode: str,
    base_args: Dict[str, Any],
    output_dir: str,
    use_simulation: bool = True,
) -> Dict:
    """Dispatch to simulation or distributed runner for a single mode."""
    print(f"\n{'=' * 60}")
    print(
        f"  {display_mode.upper()} "
        f"({'SIMULATION' if use_simulation else 'NON-SIMULATION'})"
    )
    print(f"{'=' * 60}\n")

    result_dir = os.path.join(output_dir, display_mode)
    os.makedirs(result_dir, exist_ok=True)

    # Start the gnark ZKP service if this mode needs it; never run a ZKP mode without it.
    if _needs_gnark(mode_cfg) and not _ensure_gnark_service(result_dir):
        return {
            "mode": display_mode,
            "success": False,
            "exit_code": None,
            "result_dir": result_dir,
            "benchmark": None,
            "error": "gnark proof service unavailable; ZKP mode not run",
        }

    if use_simulation:
        return run_simulation(mode_cfg, display_mode, base_args, result_dir)
    else:
        return run_distributed(mode_cfg, display_mode, base_args, result_dir)
