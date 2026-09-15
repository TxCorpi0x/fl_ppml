"""Step 6 evidence: where Groth16 keys come from and whether they survive.

Run from the repository root (needs the built gnark service binary):

    PYTHONPATH=. python audit/evidence/setup_evidence.py

Phase 1 (commit 0977aff) started bare service instances, each running its own
lazy setup. Since Phase 2 the service refuses to start without pinned keys, so
this script runs `gnark_service setup` once into a temporary directory (small
circuits: norm n=8, ElGamal n=4) and starts real processes on free ports:

  Q1  a proof from the prover, verified by two separate verifier processes
  Q2  the same proof verified after a verifier restart
  Q3  the same checks for he_elgamal_zkp proofs
  Q4  a verifier started from a second, independent setup refuses the proof
  Q5  role separation: the verifier has no proving endpoints and needs no
      proving keys; the prover serves no verification endpoints
  Q6  first vs second proof time (setup no longer runs inside a request)

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

BINARY = Path(__file__).resolve().parents[2] / "zkp_gnark_service" / "gnark_service"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup(root):
    subprocess.run([str(BINARY), "setup", "--keys-dir", f"{root}/keys", "--pk-dir", f"{root}/pk",
                    "--norm-n", "8", "--elgamal-n", "4"], check=True, capture_output=True)


def start(role, root, with_pk=True):
    port = free_port()
    cmd = [str(BINARY), "serve", "--role", role, "--keys-dir", f"{root}/keys", "--port", str(port)]
    if with_pk:
        cmd += ["--pk-dir", f"{root}/pk"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                return proc, url
        except requests.RequestException:
            time.sleep(0.1)
    raise RuntimeError(f"{role} did not start")


def stop(*procs):
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=10)


def outcome(fn):
    try:
        return fn()
    except Exception as exc:
        return f"refused ({type(exc).__name__}: {str(exc)[:70]})"


def main():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
        setup(root)
        setup(other)
        os.environ["FL_ZKP_KEYS_DIR"], os.environ["FL_ZKP_PK_DIR"] = f"{root}/keys", f"{root}/pk"

        import fl.core.zkp_gnark as zg
        from fl.core import elgamal_gnark as eg
        from fl.keys.he_elgamal import generate, load_client

        prover, prover_url = start("prover", root)
        v1, v1_url = start("verifier", root, with_pk=False)
        v2, v2_url = start("verifier", root, with_pk=False)
        foreign, foreign_url = start("verifier", other, with_pk=False)
        procs = [prover, v1, v2, foreign]
        try:
            weights = {"w": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)}
            t0 = time.perf_counter()
            proofs, _ = zg.generate_gnark_proofs(weights, service_url=prover_url)
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            zg.generate_gnark_proofs(weights, service_url=prover_url)
            second = time.perf_counter() - t0
            light = lambda url: zg.verify_gnark_proofs_light(proofs, service_url=url)[0]

            print("Q1  norm: prove on P, verify on V1           ->", light(v1_url))
            print("    norm: prove on P, verify on V2           ->", light(v2_url))
            print("    plaintext /verify (hash recomputed) on V2 ->",
                  zg.verify_gnark_proofs(list(weights.values()), list(weights), proofs, service_url=v2_url)[0])

            with tempfile.TemporaryDirectory() as d:
                generate(secret_path=f"{d}/s.json", public_path=f"{d}/p.json")
                pk = load_client(f"{d}/s.json")["pk"]
            zg.DEFAULT_PROVER_URL = prover_url
            ct, proof = eg.prove_chunk(pk, np.array([3, -4], dtype=np.int64), 100, 7)
            for label, url in (("V1", v1_url), ("V2", v2_url)):
                zg.DEFAULT_VERIFIER_URL = url
                print(f"Q3  elgamal: prove on P, verify on {label}        ->", eg.verify_chunk(pk, ct, 100, 7, proof))

            stop(v1)
            procs.remove(v1)
            v1, v1_url = start("verifier", root, with_pk=False)
            procs.append(v1)
            zg.DEFAULT_VERIFIER_URL = v1_url
            print("Q2  norm: prove on P, restart V1, verify     ->", light(v1_url))
            print("    elgamal: prove on P, restart V1, verify  ->", eg.verify_chunk(pk, ct, 100, 7, proof))

            zg.DEFAULT_VERIFIER_URL = foreign_url
            print("Q4  norm: verify on verifier with other keys ->", outcome(lambda: light(foreign_url)))
            print("    elgamal: same                            ->",
                  outcome(lambda: eg.verify_chunk(pk, ct, 100, 7, proof)))

            print("Q5  POST verifier/prove                      -> HTTP",
                  requests.post(f"{v1_url}/prove", json={}, timeout=5).status_code)
            print("    POST verifier/elgamal/prove              -> HTTP",
                  requests.post(f"{v1_url}/elgamal/prove", json={}, timeout=5).status_code)
            print("    POST prover/verify_light                 -> HTTP",
                  requests.post(f"{prover_url}/verify_light", json={}, timeout=5).status_code)
            print("    verifier started without --pk-dir         -> healthy:",
                  requests.get(f"{v1_url}/health", timeout=5).json()["role"] == "verifier")

            print(f"Q6  first proof n=8 (padded): {first:.3f} s; second: {second:.3f} s")
        finally:
            stop(*procs)


if __name__ == "__main__":
    main()
