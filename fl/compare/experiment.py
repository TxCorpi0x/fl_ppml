"""
Experiment runner — simulation and distributed modes.

These functions are the engine behind the comparison framework.
They are invoked by fl.compare.runner.run_comparison() and should not
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
from typing import Any, Dict, Optional

from fl.compare.benchmark import (
    aggregate_client_benchmarks,
    merge_server_and_clients,
)
from fl.compare.registry import ModeConfig


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
    from fl.core.gnark_keys import manifest_sha256

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


def _wait_for_server(server_proc, server_start_ts: float, server_timeout: float, grace: float) -> Optional[str]:
    """Wait for the server after all clients exited; terminate it if it outlives the grace period or the timeout.

    Returns None if the server exited on its own, otherwise why it was stopped.
    A terminated server has a non-zero return code, so the run is marked failed.
    """
    clients_done = time.monotonic()
    reason = None
    while server_proc.poll() is None:
        now = time.monotonic()
        if server_timeout > 0 and now - server_start_ts > server_timeout:
            reason = f"server timeout after {now - server_start_ts:.0f}s"
        elif now - clients_done > grace:
            reason = f"server still running {now - clients_done:.0f}s after every client exited"
        if reason:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                else:
                    server_proc.terminate()
                server_proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                server_proc.kill()
                server_proc.wait(timeout=10)
            return reason
        time.sleep(0.5)
    return None


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
    from fl.core.gnark_keys import keys_dir, load_manifest, missing_proving_keys, pk_dir

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
    # fl/compare/experiment.py → ../../zkp_gnark_service/
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
    from fl.core.gnark_keys import keys_dir, pk_dir

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
    """Add gRPC keepalive / message-size env vars to a copy of ``base``."""
    env = base.copy()
    env["FL_GRPC_MAX_MESSAGE_LENGTH"] = os.environ.get(
        "FL_GRPC_MAX_MESSAGE_LENGTH", str(2_147_483_647)
    )
    env["GRPC_ARG_KEEPALIVE_TIME_MS"] = "300000"
    env["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"] = "120000"
    env["GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "1"
    env["GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"
    # FL_ENCRYPT_LAYERS is inherited from ``base`` only when the user sets it.
    # Injecting a first-layer default here silently disabled full-model
    # encryption in every harness run (audit/binding.md B-1).
    env["FL_CLIENT_TIMEOUT"] = os.environ.get("FL_CLIENT_TIMEOUT", "7200")
    return env


def _build_mode_flags(mode_cfg: ModeConfig) -> list:
    """Return CLI flag list for the given mode."""
    flags: list = []
    m = mode_cfg.internal_mode
    b = mode_cfg.he_backend

    if m == "he":
        flags.append("--he")
        if b:
            flags.extend(["--he_backend", b])
        if b != "concrete_tfhe":
            flags.extend(["--path_keys", "keys/he_tenseal/secret_key.pkl"])
            flags.extend(["--path_public_key", "keys/he_tenseal/public_key.pkl"])
    elif m == "he_zkp":
        # Hybrid: HE encryption + ZKP integrity proofs
        flags.append("--he")
        if b:
            flags.extend(["--he_backend", b])
        if b != "concrete_tfhe":
            flags.extend(["--path_keys", "keys/he_tenseal/secret_key.pkl"])
            flags.extend(["--path_public_key", "keys/he_tenseal/public_key.pkl"])
        flags.append("--zkp")
        zkp_backend = os.environ.get("FL_ZKP_BACKEND", "gnark").lower()
        flags.extend(["--zkp_backend", zkp_backend])
        if zkp_backend != "gnark":
            flags.extend(["--zkp_params", "keys/zkp/zkp_params.pkl"])
    elif m == "he_zkp_dp":
        # Triple: HE encryption + ZKP integrity proofs + DP-SGD noise
        flags.append("--he")
        if b:
            flags.extend(["--he_backend", b])
        if b != "concrete_tfhe":
            flags.extend(["--path_keys", "keys/he_tenseal/secret_key.pkl"])
            flags.extend(["--path_public_key", "keys/he_tenseal/public_key.pkl"])
        flags.append("--zkp")
        zkp_backend = os.environ.get("FL_ZKP_BACKEND", "gnark").lower()
        flags.extend(["--zkp_backend", zkp_backend])
        if zkp_backend != "gnark":
            flags.extend(["--zkp_params", "keys/zkp/zkp_params.pkl"])
        flags.append("--dp")
        flags.extend(["--dp_params", "keys/dp/dp_params.pkl"])
    elif m == "zkp":
        flags.append("--zkp")
        zkp_backend = os.environ.get("FL_ZKP_BACKEND", "gnark").lower()
        flags.extend(["--zkp_backend", zkp_backend])
        if zkp_backend != "gnark":
            flags.extend(["--zkp_params", "keys/zkp/zkp_params.pkl"])
    elif m == "dp":
        flags.append("--dp")
        flags.extend(["--dp_params", "keys/dp/dp_params.pkl"])

    return flags


def _load_benchmark(result_dir: str, display_mode: str) -> Optional[Dict]:
    """Load the benchmark JSON written by simulation.py / main_server.py."""
    # fl/runner.run_mode() saves as benchmark_{mode_name}.json
    candidates = [
        os.path.join(result_dir, f"benchmark_{display_mode}.json"),
        os.path.join(result_dir, "benchmark.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data or {}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Simulation mode (in-process Flower)
# ─────────────────────────────────────────────────────────────────────────────


def run_simulation(
    mode_cfg: ModeConfig,
    display_mode: str,
    base_args: Dict[str, Any],
    result_dir: str,
) -> Dict:
    """Run a single experiment via simulation.py subprocess."""
    cmd = [sys.executable, "simulation.py", "simulation"]

    for key, value in base_args.items():
        if value is not None and value != "":
            cmd.extend([f"--{key}", str(value)])

    cmd.extend(_build_mode_flags(mode_cfg))
    # Select the registered mode by key: flag combinations alone can't name
    # modes such as zkp_sampled (audit/numbers.md item 1).
    cmd.extend(["--privacy_mode", display_mode])
    cmd += [
        "--benchmark",
        "--save_results",
        f"{result_dir}/",
        "--model_save",
        f"{result_dir}/model.pt",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["FL_SIMULATION"] = "1"

    timeout_s = mode_cfg.timeout_s
    start = datetime.now()

    stdout_log = open(f"{result_dir}/stdout.log", "w")
    stderr_log = open(f"{result_dir}/stderr.log", "w")
    # Write the full command line into the log so callers can grep for flags
    # like --seed without needing to capture the parent process stdout.
    stdout_log.write(f"# Command: {' '.join(cmd)}\n")
    stdout_log.flush()
    try:
        proc = _register_proc(
            subprocess.Popen(
                cmd,
                stdout=stdout_log,
                stderr=stderr_log,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        returncode = proc.returncode
    finally:
        _unregister_proc(proc)
        stdout_log.close()
        stderr_log.close()

    duration = (datetime.now() - start).total_seconds()

    benchmark = _load_benchmark(result_dir, display_mode)
    if benchmark is None and returncode == 0:
        print(
            f"[WARN]  Warning: {display_mode} benchmark file not found after successful run"
        )

    success = returncode == 0 and benchmark is not None
    print(
        f"{display_mode.upper()} {'[OK] SUCCESS' if success else '[FAIL] FAILED'} ({duration:.1f}s)"
    )
    return {
        "mode": display_mode,
        "duration": duration,
        "exit_code": returncode,
        "result_dir": result_dir,
        "benchmark": benchmark,
        "success": success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distributed mode (real gRPC server + client processes)
# ─────────────────────────────────────────────────────────────────────────────

_PORT_MAP = {
    "baseline": 8081,
    "he": 8082,
    "zkp": 8083,
    "dp": 8084,
    "he_zkp": 8085,
    "he_zkp_dp": 8086,
}


def run_distributed(
    mode_cfg: ModeConfig,
    display_mode: str,
    base_args: Dict[str, Any],
    result_dir: str,
) -> Dict:
    """Run a single experiment with real gRPC server + N client processes."""
    num_clients = base_args.get("number_clients", 3)
    m = mode_cfg.internal_mode

    # ── common args (shared by server and clients) ─────────────────────────
    common_args: list = []
    for key, flag in {
        "dataset": "--dataset",
        "data_path": "--data_path",
        "max_epochs": "--max_epochs",
        "batch_size": "--batch_size",
        "device": "--device",
        "number_clients": "--number_clients",  # clients need this to partition data correctly
        "rounds": "--rounds",  # keep client benchmark metadata aligned with server run
        "dirichlet_alpha": "--dirichlet_alpha",  # non-IID partitioning
        "dp_epsilon": "--dp_epsilon",  # DP privacy budget override
        "seed": "--seed",  # reproducibility seed
    }.items():
        v = base_args.get(key)
        if v is not None and v != "":
            common_args.extend([flag, str(v)])

    server_extra: list = []
    for key, flag in {
        "rounds": "--rounds",
    }.items():
        v = base_args.get(key)
        if v is not None and v != "":
            server_extra.extend([flag, str(v)])
    # Start round 1 only once every client is connected (FedPrivate.configure_fit
    # waits for min_avail_clients before sizing its sample). Otherwise a late client
    # misses round 1: for commit–challenge modes it then has nothing to answer
    # the challenge with, and every mode runs its first round with fewer clients.
    server_extra.extend(["--min_avail_clients", str(num_clients)])

    # Chain ledger args are server-only (clients don't write to the ledger)
    chain_backend = base_args.get("chain_backend", "")
    chain_ledger_path = base_args.get("chain_ledger_path", "")
    if chain_backend:
        server_extra.extend(["--chain_backend", chain_backend])
    if chain_ledger_path and chain_backend != "none":
        server_extra.extend(["--chain_ledger_path", chain_ledger_path])

    # Select the registered mode by key: flag combinations alone can't name
    # modes such as zkp_sampled (audit/numbers.md item 1).
    mode_flags = _build_mode_flags(mode_cfg) + ["--privacy_mode", display_mode]

    # ── server ────────────────────────────────────────────────────────────
    server_cmd = (
        [sys.executable, "main_server.py", "server"]
        + common_args
        + server_extra
        + mode_flags
        + [
            "--benchmark",
            "--save_results",
            result_dir,
            "--model_save",
            f"{result_dir}/server_model.pt",
        ]
    )
    print(f"Starting server: {' '.join(server_cmd)}\n")

    port = _PORT_MAP.get(m, 8081)
    server_addr = f"127.0.0.1:{port}"
    env_server = _grpc_env(os.environ.copy())
    env_server["FL_SERVER_ADDRESS"] = server_addr

    server_log_path = f"{result_dir}/server.log"
    with open(server_log_path, "w") as server_log:
        server_start_ts = time.monotonic()
        server_proc = _register_proc(
            subprocess.Popen(
                server_cmd,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env=env_server,
            )
        )

    time.sleep(5)

    if server_proc.poll() is not None:
        print(f"[ERROR] Server failed to start! Check {server_log_path}")
        return {
            "mode": display_mode,
            "success": False,
            "exit_code": server_proc.returncode,
            "result_dir": result_dir,
            "benchmark": None,
        }

    print(f"[OK] Server started (PID: {server_proc.pid})")

    # ── clients ───────────────────────────────────────────────────────────
    client_benchmark_args = [
        "--benchmark",
        "--save_results",
        result_dir,
        "--model_save",
        f"{result_dir}/model.pt",
    ]
    client_procs = []

    for cid in range(num_clients):
        client_cmd = (
            [sys.executable, "main_client.py", "client"]
            + common_args
            + mode_flags
            + client_benchmark_args
            + ["--id_client", str(cid)]
        )
        print(f"Starting client {cid}...")
        log_path = f"{result_dir}/client_{cid}.log"
        log_file = open(log_path, "w")
        env_client = _grpc_env(os.environ.copy())
        env_client["FL_SERVER_ADDRESS"] = server_addr
        proc = _register_proc(
            subprocess.Popen(
                client_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env=env_client,
            )
        )
        client_procs.append((proc, log_file, cid))

    print(f"[OK] All {num_clients} clients started, waiting for completion...")

    env_client_timeout = os.environ.get("FL_CLIENT_TIMEOUT", "").strip()
    if env_client_timeout:
        client_timeout = int(env_client_timeout)
    else:
        # Heavy crypto modes can run for multiple hours on CIFAR; default to 6h
        # when no explicit timeout is provided.
        mode_l = str(m).lower()
        client_timeout = (
            21600
            if ("zkp" in mode_l or "he_" in mode_l or mode_l.startswith("he"))
            else 7200
        )

    env_server_timeout = os.environ.get("FL_SERVER_TIMEOUT", "").strip()
    if env_server_timeout:
        server_timeout = int(env_server_timeout)
    else:
        # Server should have at least as much wall-clock budget as clients.
        # Add 30min headroom for final eval/checkpoint/merge and shutdown.
        server_timeout = max(client_timeout, client_timeout + 1800)

    # ── wait for clients ──────────────────────────────────────────────────
    # IMPORTANT: do not wait sequentially with the same timeout per proc, or
    # client_0 is always penalized in long runs (it is waited on first and can
    # hit timeout while other clients get extra wall-clock time).
    client_failures = 0
    pending = {
        proc: {
            "cid": cid,
            "log_file": log_file,
            "start_ts": time.monotonic(),
        }
        for proc, log_file, cid in client_procs
    }

    while pending:
        progressed = False
        for proc in list(pending.keys()):
            meta = pending[proc]
            cid = meta["cid"]
            log_file = meta["log_file"]

            ec = proc.poll()
            if ec is not None:
                progressed = True
                if ec != 0:
                    print(f"[WARN]  Client {cid} failed (exit {ec})")
                    client_failures += 1
                else:
                    print(f"[OK] Client {cid} completed")
                _unregister_proc(proc)
                try:
                    log_file.close()
                except Exception:
                    pass
                pending.pop(proc, None)
                continue

            if client_timeout > 0:
                elapsed = time.monotonic() - float(meta["start_ts"])
                if elapsed > client_timeout:
                    progressed = True
                    print(f"⏱️  Client {cid} timed out after {elapsed:.0f}s!")
                    try:
                        if hasattr(os, "killpg"):
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        else:
                            proc.terminate()
                        proc.wait(timeout=10)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    client_failures += 1
                    _unregister_proc(proc)
                    try:
                        log_file.close()
                    except Exception:
                        pass
                    pending.pop(proc, None)

        if not progressed:
            time.sleep(1.0)

    # ── wait for server ───────────────────────────────────────────────────
    # Every client has exited by now. A server that doesn't follow within the
    # grace period is stuck (audit/failmodes.md F-2: it waited in round-1
    # evaluation for clients that had all failed, for over an hour).
    grace = 60 if client_failures else int(os.environ.get("FL_SERVER_GRACE", "600"))
    print("\nWaiting for server to complete...")
    try:
        stop_reason = _wait_for_server(server_proc, server_start_ts, server_timeout, grace)
    finally:
        _unregister_proc(server_proc)
    if stop_reason:
        print(f"⏱️  {stop_reason}; server terminated")
    else:
        print("[OK] Server completed")

    # ── merge benchmarks ──────────────────────────────────────────────────
    server_bm = None
    bm_path = os.path.join(result_dir, "benchmark.json")
    if os.path.exists(bm_path):
        with open(bm_path) as f:
            server_bm = json.load(f)

    client_bms = []
    for i in range(num_clients):
        p = os.path.join(result_dir, f"client_{i}_benchmark.json")
        if os.path.exists(p):
            with open(p) as f:
                client_bms.append(json.load(f))

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

    success = (
        server_proc.returncode == 0 and client_failures == 0 and benchmark is not None
    )

    print(f"{display_mode.upper()} {'[OK] SUCCESS' if success else '[FAIL] FAILED'}")
    if client_failures:
        print(f"  ({client_failures}/{num_clients} clients failed)")

    return {
        "mode": display_mode,
        "success": success,
        "exit_code": server_proc.returncode,
        "result_dir": result_dir,
        "benchmark": benchmark,
        "client_failures": client_failures,
    }


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
