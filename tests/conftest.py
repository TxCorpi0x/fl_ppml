"""Shared fixtures."""

import os
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

GNARK_BINARY = Path(os.environ.get("FL_GNARK_BINARY", Path(__file__).resolve().parents[1] / "zkp_gnark_service" / "gnark_service"))
TEST_NORM_N, TEST_ELGAMAL_N = 8, 4

requires_gnark = pytest.mark.skipif(
    not GNARK_BINARY.exists(), reason="gnark_service binary not built"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gnark_setup(keys_dir: Path, pk_dir: Path, norm_n=TEST_NORM_N, elgamal_n=TEST_ELGAMAL_N) -> None:
    subprocess.run(
        [str(GNARK_BINARY), "setup", "--keys-dir", str(keys_dir), "--pk-dir", str(pk_dir),
         "--norm-n", str(norm_n), "--elgamal-n", str(elgamal_n)],
        check=True,
        capture_output=True,
    )


def start_gnark(role: str, keys_dir: Path, pk_dir: Path = None):
    """Start one gnark service role on a free port; returns (process, url)."""
    port = _free_port()
    cmd = [str(GNARK_BINARY), "serve", "--role", role, "--keys-dir", str(keys_dir), "--port", str(port)]
    if pk_dir is not None:
        cmd += ["--pk-dir", str(pk_dir)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        if proc.poll() is not None:
            pytest.fail(f"gnark {role} exited: {proc.stderr.read()}")
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                return proc, url
        except requests.RequestException:
            time.sleep(0.1)
    proc.terminate()
    pytest.fail(f"gnark {role} did not become healthy")


@pytest.fixture(scope="session", autouse=True)
def gnark_test_keys(tmp_path_factory):
    """Pin small test keys for the whole session, never the committed production keys."""
    if not GNARK_BINARY.exists():
        yield None
        return
    root = tmp_path_factory.mktemp("gnark_keys")
    keys_dir, pk_dir = root / "keys", root / "pk"
    gnark_setup(keys_dir, pk_dir)
    saved = {k: os.environ.get(k) for k in ("FL_ZKP_KEYS_DIR", "FL_ZKP_PK_DIR")}
    os.environ["FL_ZKP_KEYS_DIR"], os.environ["FL_ZKP_PK_DIR"] = str(keys_dir), str(pk_dir)
    try:
        yield SimpleNamespace(keys_dir=keys_dir, pk_dir=pk_dir)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="session")
def gnark(gnark_test_keys):
    """Separate prover and verifier processes sharing the session's pinned test keys."""
    if gnark_test_keys is None:
        pytest.skip("gnark_service binary not built")
    prover, prover_url = start_gnark("prover", gnark_test_keys.keys_dir, gnark_test_keys.pk_dir)
    verifier, verifier_url = start_gnark("verifier", gnark_test_keys.keys_dir)
    try:
        yield SimpleNamespace(
            prover=prover_url,
            verifier=verifier_url,
            keys_dir=gnark_test_keys.keys_dir,
            pk_dir=gnark_test_keys.pk_dir,
        )
    finally:
        for proc in (prover, verifier):
            proc.terminate()
            proc.wait(timeout=10)


def use_gnark(monkeypatch, gnark) -> None:
    """Point the Python ZKP clients at the session's prover and verifier."""
    import ppflx.core.zkp_gnark as zkp_gnark

    monkeypatch.setattr(zkp_gnark, "DEFAULT_PROVER_URL", gnark.prover)
    monkeypatch.setattr(zkp_gnark, "DEFAULT_VERIFIER_URL", gnark.verifier)


def break_gnark(monkeypatch, url: str = "http://127.0.0.1:1") -> None:
    """Point both service roles at an unreachable address."""
    import ppflx.core.zkp_gnark as zkp_gnark

    monkeypatch.setattr(zkp_gnark, "DEFAULT_PROVER_URL", url)
    monkeypatch.setattr(zkp_gnark, "DEFAULT_VERIFIER_URL", url)
