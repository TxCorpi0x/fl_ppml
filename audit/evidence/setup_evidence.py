"""Step 6 evidence: where Groth16 keys come from and whether they survive.

Run from the repository root (needs the built gnark service binary):

    PYTHONPATH=. python audit/evidence/setup_evidence.py

Starts real service instances on free ports and checks:
  Q1  a proof made by one service instance, verified by a second, separate
      instance of the same binary (what a split prover/verifier deployment
      would do)
  Q2  a proof verified after the same service is restarted (is the verifying
      key stable across restarts?)
  Q3  the same checks for he_elgamal_zkp proofs
  Q4  how long the first proof for a circuit size takes compared with the
      second (setup runs inside the first request)

No source files are modified.
"""

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

import fl.core.zkp_gnark as zg
from fl.core import elgamal_gnark as eg

BINARY = Path(__file__).resolve().parents[2] / "zkp_gnark_service" / "gnark_service"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(port):
    proc = subprocess.Popen([str(BINARY)], env={**os.environ, "ZKP_SERVICE_PORT": str(port)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                return proc, url
        except requests.RequestException:
            time.sleep(0.1)
    raise RuntimeError("service did not start")


def stop(proc):
    proc.terminate()
    proc.wait(timeout=10)


def light_ok(proofs, url):
    return zg.verify_gnark_proofs_light(proofs, service_url=url)[0]


def main():
    weights = {"w": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)}
    port_a, port_b = free_port(), free_port()
    a, url_a = start(port_a)
    b, url_b = start(port_b)
    try:
        t0 = time.perf_counter()
        proofs, _ = zg.generate_gnark_proofs(weights, service_url=url_a)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        zg.generate_gnark_proofs(weights, service_url=url_a)
        second = time.perf_counter() - t0

        print("Q1  norm circuit: prove on A, verify on A      ->", light_ok(proofs, url_a))
        print("    norm circuit: prove on A, verify on B      ->", light_ok(proofs, url_b))
        print("    plaintext /verify (hash recomputed) on B    ->",
              zg.verify_gnark_proofs(list(weights.values()), list(weights), proofs, service_url=url_b)[0])

        from fl.keys.he_elgamal import generate, load_client

        with tempfile.TemporaryDirectory() as d:
            generate(secret_path=f"{d}/s.json", public_path=f"{d}/p.json")
            pk = load_client(f"{d}/s.json")["pk"]
        zg.DEFAULT_SERVICE_URL = url_a
        q = np.array([3, -4], dtype=np.int64)
        ct, proof = eg.prove_chunk(pk, q, 100, 7)
        print("Q3  elgamal circuit: prove on A, verify on A   ->", eg.verify_chunk(pk, ct, 100, 7, proof))
        zg.DEFAULT_SERVICE_URL = url_b
        print("    elgamal circuit: prove on A, verify on B   ->", eg.verify_chunk(pk, ct, 100, 7, proof))

        stop(a)
        a, url_a = start(port_a)
        print("Q2  norm circuit: prove on A, restart A, verify ->", light_ok(proofs, url_a))
        zg.DEFAULT_SERVICE_URL = url_a
        print("    elgamal: prove on A, restart A, verify      ->", eg.verify_chunk(pk, ct, 100, 7, proof))

        print(f"Q4  first proof for n=4 (includes setup): {first:.3f} s; second: {second:.3f} s")
    finally:
        stop(a)
        stop(b)


if __name__ == "__main__":
    main()
