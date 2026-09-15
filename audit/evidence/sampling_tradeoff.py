"""Step 5 Phase 3: sampling rate vs prover cost vs detection probability.

Run from the repository root (needs the built gnark service binary):

    PYTHONPATH=. python audit/evidence/sampling_tradeoff.py

1. Measures he_elgamal_zkp_sampled proving time (/elgamal/prove_with) at
   several chunk sizes on a dedicated service instance, with nothing else
   running, and fits t(k) = a + b·k seconds per chunk of k coordinates.
2. For each dataset's real coordinate count and each sampling rate: sampled
   coordinates, predicted client proving time per round, its fraction of full
   proving, and detection probability 1 − C(n−m, s)/C(n, s) for m bad
   coordinates, plus the smallest m detected with probability 0.95 / 0.99.
3. Writes audit/tables/sampling_tradeoff.{csv,md} and
   audit/figures/sampling_tradeoff.png.

Timing is a cost measurement on this machine, not a security result.
"""

import csv
import math
import os
import platform
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

import fl.core.zkp_gnark as zkp_gnark
from fl.core import elgamal_gnark as eg
from fl.core.sampling import detection_probability, sample_size

REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / "zkp_gnark_service" / "gnark_service"
CHUNK = 128
MEASURE_SIZES = (16, 64, 128)
REPEATS = 3
RATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
BAD_COUNTS = (1, 5, 10, 25, 50, 100)
# Coordinate counts of the models behind the stored results (audit/sampling.md Q3).
DATASETS = {"healthcare": 2914, "creditcard": 4130, "mnist": 44426, "cifar": 62006}


def start_service():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    from fl.core.gnark_keys import keys_dir, pk_dir

    # Since Step 6 the service needs pinned keys; every chunk is padded to the
    # manifest's fixed size, so prove time no longer shrinks with k.
    proc = subprocess.Popen([str(BINARY), "serve", "--role", "prover", "--keys-dir", str(keys_dir()),
                             "--pk-dir", str(pk_dir()), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                return proc, url
        except requests.RequestException:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("gnark service did not start")


def measure_chunk_times(pk):
    rng = np.random.default_rng(0)
    times = {}
    for k in MEASURE_SIZES:
        q = rng.integers(-500, 500, size=k).astype(np.int64)
        _, rand = eg.encrypt_values(pk, q)
        bound = 10**12
        eg.prove_with(pk, q, rand, bound, 1)  # warm-up: circuit compile + setup
        samples = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            eg.prove_with(pk, q, rand, bound, 1)
            samples.append(time.perf_counter() - t0)
        times[k] = float(np.median(samples))
    sizes = np.array(list(times), dtype=float)
    b, a = np.polyfit(sizes, np.array(list(times.values())), 1)
    return times, max(a, 0.0), b


def proving_seconds(s, a, b):
    full, rem = divmod(s, CHUNK)
    return full * (a + b * CHUNK) + ((a + b * rem) if rem else 0.0)


def smallest_m(n, s, target):
    lo, hi = 1, n
    if detection_probability(n, s, n) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if detection_probability(n, s, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    proc, url = start_service()
    zkp_gnark.DEFAULT_PROVER_URL = url
    try:
        from fl.keys.he_elgamal import generate, load_client

        with tempfile.TemporaryDirectory() as d:
            generate(secret_path=f"{d}/s.json", public_path=f"{d}/p.json")
            pk = load_client(f"{d}/s.json")["pk"]
            times, a, b = measure_chunk_times(pk)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    print(f"machine: {platform.processor() or platform.machine()}, {os.cpu_count()} cpus")
    print("measured median prove_with seconds per chunk:", {k: round(v, 3) for k, v in times.items()})
    print(f"fit: t(k) = {a:.3f} + {b * 1000:.3f} ms × k")

    rows = []
    for dataset, n in DATASETS.items():
        full_time = proving_seconds(n, a, b)
        for rate in RATES:
            s = sample_size(n, rate)
            t = proving_seconds(s, a, b)
            row = {"dataset": dataset, "n": n, "rate": rate, "sampled": s,
                   "prove_s_per_client_round": round(t, 1), "fraction_of_full_cost": round(t / full_time, 4),
                   "m_for_p95": smallest_m(n, s, 0.95), "m_for_p99": smallest_m(n, s, 0.99)}
            for m in BAD_COUNTS:
                row[f"p_detect_m{m}"] = round(detection_probability(n, s, m), 4)
            rows.append(row)

    tables, figures = REPO / "audit" / "tables", REPO / "audit" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with open(tables / "sampling_tradeoff.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    detect_cols = [f"p_detect_m{m}" for m in BAD_COUNTS]
    lines = [
        "# Sampling trade-off (he_elgamal_zkp_sampled)",
        "",
        f"Chunk size {CHUNK}. Proving time is predicted from median /elgamal/prove_with timings "
        f"{ {k: round(v, 3) for k, v in times.items()} } s at chunk sizes {list(MEASURE_SIZES)}, "
        f"fit t(k) = {a:.3f} s + {b * 1000:.3f} ms·k, one caller, {os.cpu_count()} CPUs. "
        "Detection probability is exact (hypergeometric) for m bad coordinates committed before the seed.",
        "",
        "| dataset | rate | sampled | prove s / client / round | share of full | " + " | ".join(f"P(detect) m={m}" for m in BAD_COUNTS) + " | smallest m, P≥0.95 | smallest m, P≥0.99 |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(BAD_COUNTS) + "---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['rate']} | {r['sampled']} | {r['prove_s_per_client_round']} | {r['fraction_of_full_cost']:.1%} | "
            + " | ".join(f"{r[c]:.3f}" for c in detect_cols)
            + f" | {r['m_for_p95']} | {r['m_for_p99']} |"
        )
    (tables / "sampling_tradeoff.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in (1, 5, 10, 25, 50):
        left.plot(RATES, [detection_probability(2914, sample_size(2914, r), m) for r in RATES], marker="o", label=f"m = {m}")
    left.set_xscale("log")
    left.set_xlabel("sampling rate (log)")
    left.set_ylabel("detection probability")
    left.set_title("Detection vs rate (healthcare, n = 2914)")
    left.axhline(0.99, color="grey", linestyle=":", linewidth=1)
    left.legend(fontsize=8)
    for dataset, n in DATASETS.items():
        pts = [r for r in rows if r["dataset"] == dataset]
        right.plot([r["prove_s_per_client_round"] for r in pts], [r["p_detect_m10"] for r in pts], marker="o", label=dataset)
    right.set_xscale("log")
    right.set_xlabel("predicted proving seconds per client per round (log)")
    right.set_ylabel("P(detect), m = 10 bad coordinates")
    right.set_title("Cost vs detection (points = rates)")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "sampling_tradeoff.png", dpi=150)
    print(f"wrote {tables / 'sampling_tradeoff.csv'}, {tables / 'sampling_tradeoff.md'}, {figures / 'sampling_tradeoff.png'}")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
