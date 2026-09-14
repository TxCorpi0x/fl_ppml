"""Shared fixtures."""

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

GNARK_BINARY = Path(__file__).resolve().parents[1] / "zkp_gnark_service" / "gnark_service"

requires_gnark = pytest.mark.skipif(
    not GNARK_BINARY.exists(), reason="gnark_service binary not built"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def gnark_service_url():
    """Start the real gnark service on a free port for the test session."""
    if not GNARK_BINARY.exists():
        pytest.skip("gnark_service binary not built")
    port = _free_port()
    proc = subprocess.Popen(
        [str(GNARK_BINARY)],
        env={**os.environ, "ZKP_SERVICE_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            pytest.fail("gnark service did not become healthy")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)
