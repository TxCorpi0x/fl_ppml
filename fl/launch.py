"""
Run one experiment on Flower: a SuperLink, one SuperNode per client, and the
Flower App declared in pyproject.toml.

The SuperLink starts the ServerApp (``fl.app:server_app``); each SuperNode
starts a ClientApp process (``fl.app:client_app``) for every message. All of
them run from the repository root with this checkout on PYTHONPATH, so relative
key and dataset paths resolve as they do for compare.py. ``flwr run`` submits
the run; its FAB carries only pyproject.toml and the README.

With ``simulation=True`` the SuperLink runs Flower's Simulation Runtime instead
of connecting SuperNodes, and HE modes transport plaintext (sim-mode).

ZKP modes need the gnark proof service (compare.py starts it). By hand::

    python -m fl.launch --mode baseline --dataset healthcare --num-clients 3 --num-rounds 3
    python -m fl.launch --mode dp --simulation --results-dir results/dp_sim/
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent
CONNECTION = "fl-ppml"


@dataclass(frozen=True)
class Ports:
    fleet: int = 19092
    control: int = 19093
    runtime: int = 19091
    supernode: int = 19100  # SuperNode i serves its runtime API on supernode + i


# ── Run config ───────────────────────────────────────────────────────────────


def run_config_defaults() -> Dict[str, object]:
    """The run config keys and defaults from pyproject.toml [tool.flwr.app.config]."""
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["flwr"]["app"]["config"]


def make_run_config(overrides: Dict[str, object]) -> Dict[str, object]:
    """Defaults with ``overrides`` applied, each cast to its default's type."""
    config = dict(run_config_defaults())
    unknown = sorted(set(overrides) - set(config))
    if unknown:
        raise ValueError(f"unknown run config keys: {unknown}")
    for key, value in overrides.items():
        kind = type(config[key])
        if kind is bool and not isinstance(value, bool):
            raise TypeError(f"run config {key!r} must be a boolean, got {value!r}")
        config[key] = kind(value)
    return config


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def write_run_config(path: Path, run_config: Dict[str, object]) -> Path:
    path.write_text("".join(f"{json.dumps(k)} = {_toml_value(v)}\n" for k, v in run_config.items()))
    return path


# ── Processes ────────────────────────────────────────────────────────────────


def flwr_env(flwr_home: Path, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(extra or {})
    env["PATH"] = os.pathsep.join([str(BIN), env.get("PATH", "")])
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(REPO), env.get("PYTHONPATH", "")] if p)
    env["FLWR_HOME"] = str(flwr_home)
    # Benchmark runs send no usage telemetry and make no update checks.
    env["FLWR_TELEMETRY_ENABLED"] = "0"
    # Apps run in this environment. Without this, the SuperLink creates a fresh
    # runtime environment per run and `uv sync`s pyproject.toml into it.
    env["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
    env["FLWR_DISABLE_UPDATE_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _require_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"port {port} is in use ({exc}); stop the process holding it") from None


def _wait_for_port(port: int, proc: subprocess.Popen, name: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited during startup (rc={proc.returncode})")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise RuntimeError(f"{name} not listening on port {port} after {timeout:.0f}s")


def _terminate(proc: subprocess.Popen) -> None:
    """Stop a process started in its own session, with everything it spawned."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def _json_output(out: subprocess.CompletedProcess) -> dict:
    start = out.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"flwr printed no JSON (rc={out.returncode}): {(out.stderr or out.stdout).strip()}")
    return json.JSONDecoder().raw_decode(out.stdout[start:])[0]


def wait_for_run(
    status: Callable[[], Tuple[str, str]],
    clients_alive: Callable[[], bool],
    stop: Callable[[], None],
    *,
    timeout: float,
    grace: float,
    poll: float = 2.0,
) -> Tuple[str, str]:
    """Poll a run until it finishes; stop it on timeout or when it outlives its clients.

    Returns ``(status, details)``. A run stopped here reports
    ``finished:stopped`` and the reason, so it is never counted as completed.
    """
    start = time.monotonic()
    clients_gone: Optional[float] = None
    while True:
        current, details = status()
        if current.startswith("finished"):
            return current, details
        now = time.monotonic()
        if clients_alive():
            clients_gone = None
        elif clients_gone is None:
            clients_gone = now
        reason = None
        if timeout > 0 and now - start > timeout:
            reason = f"run timeout after {now - start:.0f}s"
        elif clients_gone is not None and now - clients_gone > grace:
            reason = f"run still active {now - clients_gone:.0f}s after every client exited"
        if reason:
            stop()
            return "finished:stopped", reason
        time.sleep(poll)


class Federation:
    """A SuperLink and SuperNodes for one run: started on enter, stopped on exit.

    Logs go to ``work_dir``: server.log (SuperLink), serverapp.log (ServerApp,
    fetched after the run) and client_<i>.log (SuperNode i and its ClientApps).
    """

    def __init__(
        self,
        work_dir,
        num_clients: int,
        *,
        simulation: bool = False,
        ports: Ports = Ports(),
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.num_clients = num_clients
        self.simulation = simulation
        self.ports = ports
        self.flwr_home = self.work_dir / ".flwr"
        self.env = flwr_env(self.flwr_home, env)
        self.superlink: Optional[subprocess.Popen] = None
        self.supernodes: List[subprocess.Popen] = []
        self._logs: list = []

    def __enter__(self) -> "Federation":
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.flwr_home.mkdir(parents=True, exist_ok=True)
        (self.flwr_home / "config.toml").write_text(
            f'[superlink]\ndefault = "{CONNECTION}"\n\n'
            f'[superlink.{CONNECTION}]\naddress = "127.0.0.1:{self.ports.control}"\ninsecure = true\n'
        )
        try:
            self._start_superlink()
            if not self.simulation:
                self._start_supernodes()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _spawn(self, cmd: list, log_name: str) -> subprocess.Popen:
        cmd = [str(c) for c in cmd]
        log = open(self.work_dir / log_name, "w")
        self._logs.append(log)
        log.write(f"# Command: {' '.join(cmd)}\n")
        log.flush()
        return subprocess.Popen(cmd, cwd=REPO, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _start_superlink(self) -> None:
        p = self.ports
        for port in (p.fleet, p.control, p.runtime):
            _require_free(port)
        cmd = [
            BIN / "flower-superlink", "--insecure", "--isolation", "subprocess",
            "--disable-runtime-dependency-installation",
            "--fleet-api-address", f"127.0.0.1:{p.fleet}",
            "--control-api-address", f"127.0.0.1:{p.control}",
            "--host", "127.0.0.1", "--port", p.runtime,
        ]
        if self.simulation:
            cmd.append("--simulation")
        self.superlink = self._spawn(cmd, "server.log")
        _wait_for_port(p.control, self.superlink, "SuperLink")

    def _start_supernodes(self) -> None:
        for i in range(self.num_clients):
            port = self.ports.supernode + i
            _require_free(port)
            cmd = [
                BIN / "flower-supernode", "--insecure",
                "--superlink", f"127.0.0.1:{self.ports.fleet}",
                "--node-config", f"partition-id={i} num-partitions={self.num_clients}",
                "--host", "127.0.0.1", "--port", port,
            ]
            self.supernodes.append(self._spawn(cmd, f"client_{i}.log"))

    def _flwr(self, *args, timeout: float = 300) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(BIN / "flwr"), *map(str, args)], cwd=REPO, env=self.env, capture_output=True, text=True, timeout=timeout
        )

    def submit(self, overrides: Dict[str, object]) -> int:
        path = write_run_config(self.work_dir / "run_config.toml", make_run_config(overrides))
        args = ["run", REPO, CONNECTION, "--run-config", path, "--format", "json"]
        if self.simulation:
            args += ["--federation-config", f"num-supernodes={self.num_clients}"]
        out = self._flwr(*args)
        payload = _json_output(out)
        if not payload.get("success") or payload.get("run-id") is None:
            raise RuntimeError(f"flwr run failed: {out.stdout.strip()} {out.stderr.strip()}")
        return int(payload["run-id"])

    def status(self, run_id: int, attempts: int = 3) -> Tuple[str, str]:
        for attempt in range(attempts):
            try:
                runs = _json_output(self._flwr("ls", CONNECTION, "--run-id", run_id, "--format", "json", timeout=60)).get("runs") or []
                if runs:
                    return runs[0]["status"], runs[0].get("status-details", "")
                error = f"run {run_id} not listed"
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"cannot read the status of run {run_id}: {error}")

    def save_app_logs(self, run_id: int) -> None:
        """Write the ServerApp's output to serverapp.log (Flower keeps it in the SuperLink, not in server.log)."""
        try:
            out = self._flwr("log", run_id, CONNECTION, "--show", timeout=120)
            (self.work_dir / "serverapp.log").write_text(out.stdout + out.stderr)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[flwr] could not fetch logs of run {run_id}: {exc}")

    def stop_run(self, run_id: int) -> None:
        self._flwr("stop", run_id, CONNECTION, timeout=60)

    def clients_alive(self) -> bool:
        return self.simulation or any(p.poll() is None for p in self.supernodes)

    def client_failures(self) -> int:
        """SuperNodes that exited on their own; call before close()."""
        return sum(p.poll() is not None for p in self.supernodes)

    def wait(self, run_id: int, *, timeout: float = 0, grace: float = 600) -> Tuple[str, str]:
        return wait_for_run(
            lambda: self.status(run_id), self.clients_alive, lambda: self.stop_run(run_id), timeout=timeout, grace=grace
        )

    def close(self) -> None:
        for proc in [*self.supernodes, self.superlink]:
            if proc is not None:
                _terminate(proc)
        for log in self._logs:
            log.close()
        self._logs.clear()


def run(
    overrides: Dict[str, object],
    *,
    num_clients: int,
    work_dir,
    simulation: bool = False,
    timeout: float = 0.0,
    grace: float = 600.0,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Start a federation, run the app with ``overrides`` and wait for it to finish."""
    with Federation(work_dir, num_clients, simulation=simulation, env=env) as federation:
        run_id = federation.submit(overrides)
        print(f"[flwr] run {run_id} submitted ({'simulation' if simulation else f'{num_clients} SuperNodes'})")
        status, details = federation.wait(run_id, timeout=timeout, grace=grace)
        failures = federation.client_failures()
        federation.save_app_logs(run_id)
    print(f"[flwr] run {run_id}: {status}{f' ({details})' if details else ''}")
    return {"run_id": run_id, "status": status, "details": details, "client_failures": failures}


def _parse_bool(text: str) -> bool:
    if text.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")
    return text.lower() == "true"


def main(argv=None) -> int:
    defaults = run_config_defaults()
    parser = argparse.ArgumentParser(description="Run one experiment on a local SuperLink and SuperNodes.")
    for key, value in defaults.items():
        kind = _parse_bool if isinstance(value, bool) else type(value)
        parser.add_argument(f"--{key}", dest=key, type=kind, default=None, help=f"default: {value!r}")
    parser.add_argument("--simulation", action="store_true", help="Flower Simulation Runtime; HE modes transport plaintext")
    parser.add_argument("--timeout", type=float, default=0.0, help="stop the run after this many seconds (0: no limit)")
    parser.add_argument("--work-dir", default=None, help="logs and Flower state (default: --results-dir)")
    args = parser.parse_args(argv)

    overrides = {key: getattr(args, key) for key in defaults if getattr(args, key) is not None}
    if args.simulation:
        overrides["sim-mode"] = True
    config = make_run_config(overrides)
    overrides.setdefault("min-avail-clients", config["num-clients"])
    outcome = run(
        overrides,
        num_clients=int(config["num-clients"]),
        work_dir=args.work_dir or config["results-dir"],
        simulation=args.simulation,
        timeout=args.timeout,
    )
    return 0 if outcome["status"] == "finished:completed" and not outcome["client_failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
