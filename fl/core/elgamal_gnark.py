"""
Client for the gnark service's verifiable ElGamal endpoints.

Client and server transport for ciphertext-bound update proofs with the update
bound (docs/ZKP.md, sections 6.3 and 4). Each model coordinate is quantized to an
integer q = round(w · scale) with |q| < 2^17 and encrypted with exponential
ElGamal on BabyJubJub. One Groth16 proof per chunk has as public inputs the
chunk's ciphertexts, the public key, the chunk's declared bound, a context
value (round, layer, chunk), and the global model at the same slots: the
server's previous aggregate ciphertexts and their total weight W, or the public
initial model. It proves

    Σ (W·q_i − S_i)² ≤ bound_chunk

where S is the global aggregate's centered plaintext, which the prover shows
decrypts the aggregate under the shared client key. The server accepts a
client only if the declared chunk bounds sum to at most its own total, so the
whole update satisfies ‖q − S/W‖ ≤ B·scale + √n/2.

Everything the verifier compares against — schema, chunking, total bound,
context, global model — is derived from ``Policy``, the server's bound and the
server's own global model, never from client metadata.
"""

from __future__ import annotations

import base64
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests

import fl.core.zkp_gnark as zkp_gnark

VALUE_LIMIT = 1 << 17  # |q| < 2^17, matches elgamalValueBits in elgamal.go
WEIGHT_LIMIT = 1 << 14  # aggregate weight W < 2^14, matches elgamalWeightBits
CIPHERTEXT_BYTES = 64  # compressed C1 ‖ C2
SCALAR_BYTES = 32  # big-endian encryption randomness per coordinate
HEADER_MAGIC = 0x45474C31  # "EGL1"
DEFAULT_SCALE = 10_000
_MAX_INDEX = 1 << 16
_MAX_ROUND = 1 << 32


class ElGamalServiceError(RuntimeError):
    """Service unreachable or failed internally: infrastructure, not a client fault."""


class ElGamalRejected(RuntimeError):
    """Service refused the request contents: malformed input or unsatisfied statement."""


_VERIFIER_PATHS = frozenset({"/elgamal/verify", "/elgamal/aggregate"})


def pinned_vk() -> str:
    """SHA-256 of the pinned ElGamal-circuit verifying key."""
    from fl.core.gnark_keys import ELGAMAL_CIRCUIT, pinned_vk_sha256

    return pinned_vk_sha256(ELGAMAL_CIRCUIT)


def _check_prover_key(data: dict, path: str) -> None:
    if data.get("vk_sha256") != pinned_vk():
        raise ElGamalServiceError(
            f"{path}: prover uses verifying key {data.get('vk_sha256')!r}, pinned key is {pinned_vk()!r}"
        )


def _post(path: str, payload: dict, read_timeout: float) -> dict:
    base = zkp_gnark.DEFAULT_VERIFIER_URL if path in _VERIFIER_PATHS else zkp_gnark.DEFAULT_PROVER_URL
    url = f"{base}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=(10.0, read_timeout))
    except requests.RequestException as exc:
        raise ElGamalServiceError(f"{url}: {exc}") from exc
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if 400 <= resp.status_code < 500:
        raise ElGamalRejected(f"{path}: {data.get('error') or resp.status_code}")
    if resp.status_code >= 300:
        raise ElGamalServiceError(f"{path}: HTTP {resp.status_code} {data.get('error', '')}")
    return data


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _int64_b64(values) -> str:
    return _b64(np.asarray(values, dtype="<i8").tobytes())


@dataclass(frozen=True)
class Policy:
    """Protocol parameters. The server's copy is authoritative."""

    scale: int
    chunk_size: int

    @classmethod
    def from_env(cls) -> "Policy":
        from fl.core.gnark_keys import ELGAMAL_CIRCUIT, circuit_size

        return cls(
            scale=int(os.environ.get("FL_ELGAMAL_SCALE", str(DEFAULT_SCALE))),
            # Fixed by the pinned keys: one circuit size, shorter chunks padded.
            chunk_size=circuit_size(ELGAMAL_CIRCUIT),
        )


@dataclass(frozen=True)
class Chunk:
    layer: int
    index: int
    start: int
    size: int


Schema = List[Tuple[str, Tuple[int, ...]]]


def schema_of(state_dict) -> Schema:
    return [(name, tuple(int(d) for d in tensor.shape)) for name, tensor in state_dict.items()]


def numel(shape: Sequence[int]) -> int:
    return int(np.prod(shape)) if len(shape) else 1


def chunks_for(schema: Schema, chunk_size: int) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if len(schema) >= _MAX_INDEX:
        raise ValueError(f"model has {len(schema)} tensors; at most {_MAX_INDEX - 1} supported")
    chunks = []
    for layer, (_, shape) in enumerate(schema):
        n = numel(shape)
        for index, start in enumerate(range(0, n, chunk_size)):
            if index >= _MAX_INDEX:
                raise ValueError(f"layer {layer} needs more than {_MAX_INDEX} chunks")
            chunks.append(Chunk(layer, index, start, min(chunk_size, n - start)))
    return chunks


def chunk_indices(schema: Schema, chunk: Chunk) -> np.ndarray:
    """Flat model indices a per-layer chunk covers."""
    offset = sum(numel(shape) for _, shape in schema[: chunk.layer])
    return np.arange(offset + chunk.start, offset + chunk.start + chunk.size, dtype=np.int64)


def context_value(server_round: int, chunk: Chunk) -> int:
    if not 0 <= server_round < _MAX_ROUND:
        raise ValueError(f"round {server_round} outside [0, 2^32)")
    return (server_round << 32) | (chunk.layer << 16) | chunk.index


def quantize(values: np.ndarray, scale: int, name: str) -> np.ndarray:
    q = np.round(np.asarray(values, dtype=np.float64) * scale)
    if not np.all(np.isfinite(q)) or np.any(np.abs(q) >= VALUE_LIMIT):
        worst = float(np.nanmax(np.abs(values))) if np.size(values) else 0.0
        raise ValueError(
            f"layer '{name}': |w|·scale must be < {VALUE_LIMIT} (max |w| = {worst:.4g}, "
            f"scale = {scale}); refusing to encrypt an unrepresentable update"
        )
    return q.astype("<i8")


# ── The global model an update is measured against ────────────────────────────


@dataclass
class GlobalModel:
    """Flat global model: the server's aggregate, or the public initial model.

    ``ct``/``weight`` is what the server returned last round; only clients,
    who decrypt it, also have ``sums`` (S = Σ n_k·q_k per coordinate). Before any
    aggregation the global model is the initial model's quantized ``plain``
    values with weight 1.
    """

    weight: int
    ct: Optional[bytes] = None
    sums: Optional[np.ndarray] = None
    plain: Optional[np.ndarray] = None

    @classmethod
    def initial(cls, arrays, scale: int) -> "GlobalModel":
        flat = [quantize(np.asarray(a).reshape(-1), scale, f"initial model tensor {i}") for i, a in enumerate(arrays)]
        return cls(weight=1, plain=np.concatenate(flat))

    @classmethod
    def aggregate(cls, weight: int, layers, sums=None) -> "GlobalModel":
        if not 1 <= int(weight) < WEIGHT_LIMIT:
            raise ValueError(f"aggregate weight {weight} outside [1, {WEIGHT_LIMIT})")
        return cls(
            weight=int(weight),
            ct=b"".join(np.asarray(layer, dtype=np.uint8).tobytes() for layer in layers),
            sums=None if sums is None else np.concatenate([np.asarray(s, dtype=np.int64).reshape(-1) for s in sums]),
        )

    @property
    def size(self) -> int:
        return int(self.plain.size) if self.plain is not None else len(self.ct) // CIPHERTEXT_BYTES

    def centered_sums(self) -> np.ndarray:
        if self.plain is not None:
            return self.plain
        if self.sums is None:
            raise RuntimeError("global sums unknown: only a client that decrypted the aggregate has them")
        return self.sums

    def request(self, indices, *, prover: bool) -> dict:
        idx = np.asarray(indices, dtype=np.int64)
        if self.plain is not None:
            return {"global_plain_b64": _int64_b64(self.plain[idx])}
        cts = np.frombuffer(self.ct, dtype=np.uint8).reshape(-1, CIPHERTEXT_BYTES)[idx]
        out = {"global_ct_b64": _b64(cts.tobytes()), "global_weight": self.weight}
        if prover:
            out["global_sums_b64"] = _int64_b64(self.centered_sums()[idx])
        return out


def total_bound_sq(max_update_norm: float, scale: int, weight: int, n: int) -> int:
    """Server's bound on Σ(W·q − S)² over n proven coordinates.

    W²·⌈B·scale + √n/2⌉²: an update with ‖Δ‖ ≤ B always fits, because rounding
    q to an integer moves each coordinate by at most ½ (triangle inequality).
    """
    units = math.ceil(max_update_norm * scale + math.sqrt(n) / 2)
    return (int(weight) * units) ** 2


def update_energy(q, sums, weight: int) -> int:
    d = int(weight) * np.asarray(q, dtype=object) - np.asarray(sums, dtype=object)
    return int(np.dot(d, d))


def quantize_update(glob: GlobalModel, local_flat, max_update_norm: float, scale: int) -> Tuple[np.ndarray, float, bool]:
    """Quantize a clipped local model so its update fits the bound.

    Returns (q, ‖Δ‖ before clipping in weight units, whether Δ was clipped).
    q is rounded around the global model, not around 0, so rounding error
    stays within the √n/2 slack of ``total_bound_sq``.
    """
    target = glob.centered_sums().astype(np.float64) / glob.weight
    delta = np.asarray(local_flat, dtype=np.float64) * scale - target
    norm = float(np.linalg.norm(delta)) / scale
    factor = min(1.0, 0.999 * max_update_norm / norm) if norm > 0 else 1.0
    q = np.round(target + factor * delta)
    if not np.all(np.isfinite(q)) or np.any(np.abs(q) >= VALUE_LIMIT):
        raise ValueError(f"update not representable: |q| must be < {VALUE_LIMIT} at scale {scale}")
    return q.astype("<i8"), norm, factor < 1.0


# ── Service calls ─────────────────────────────────────────────────────────────


def prove_chunk(pk_hex: str, sk: str, q: np.ndarray, bound_sq: int, context: int, glob: dict) -> Tuple[bytes, str]:
    data = _post(
        "/elgamal/prove",
        {"pk": pk_hex, "sk": str(sk), "values_b64": _int64_b64(q), "bound_sq": str(bound_sq), "context": str(context), **glob},
        zkp_gnark.DEFAULT_PROVE_TIMEOUT,
    )
    ct = base64.b64decode(data["ct_b64"])
    if len(ct) != len(q) * CIPHERTEXT_BYTES or not data.get("proof_b64"):
        raise ElGamalServiceError("/elgamal/prove returned a malformed response")
    _check_prover_key(data, "/elgamal/prove")
    return ct, data["proof_b64"]


def encrypt_values(pk_hex: str, q: np.ndarray) -> Tuple[bytes, bytes]:
    """Encrypt quantized values, returning (ciphertexts, randomness).

    The randomness never leaves the client; it's needed to prove statements
    about exactly these ciphertexts after the server's challenge.
    """
    data = _post("/elgamal/encrypt", {"pk": pk_hex, "values_b64": _int64_b64(q)}, zkp_gnark.DEFAULT_PROVE_TIMEOUT)
    ct, rand = base64.b64decode(data["ct_b64"]), base64.b64decode(data["rand_b64"])
    if len(ct) != len(q) * CIPHERTEXT_BYTES or len(rand) != len(q) * SCALAR_BYTES:
        raise ElGamalServiceError("/elgamal/encrypt returned a malformed response")
    return ct, rand


def prove_with(pk_hex: str, sk: str, q: np.ndarray, rand: bytes, bound_sq: int, context: int, glob: dict) -> Tuple[bytes, str]:
    """Prove the chunk statement for the ciphertexts determined by (q, rand)."""
    data = _post(
        "/elgamal/prove_with",
        {
            "pk": pk_hex,
            "sk": str(sk),
            "values_b64": _int64_b64(q),
            "rand_b64": _b64(rand),
            "bound_sq": str(bound_sq),
            "context": str(context),
            **glob,
        },
        zkp_gnark.DEFAULT_PROVE_TIMEOUT,
    )
    ct = base64.b64decode(data["ct_b64"])
    if len(ct) != len(q) * CIPHERTEXT_BYTES or not data.get("proof_b64"):
        raise ElGamalServiceError("/elgamal/prove_with returned a malformed response")
    _check_prover_key(data, "/elgamal/prove_with")
    return ct, data["proof_b64"]


def verify_chunk(pk_hex: str, ct: bytes, bound_sq: int, context: int, proof_b64: str, glob: dict) -> bool:
    """True iff the proof verifies for exactly these ciphertexts, global model and public inputs.

    Malformed client input counts as a failed verification; infrastructure
    failures raise ElGamalServiceError.
    """
    try:
        data = _post(
            "/elgamal/verify",
            {
                "pk": pk_hex,
                "ct_b64": _b64(ct),
                "bound_sq": str(bound_sq),
                "context": str(context),
                "proof_b64": proof_b64,
                "vk_sha256": pinned_vk(),  # the server's pin, never the client's claim
                **glob,
            },
            zkp_gnark.DEFAULT_VERIFY_TIMEOUT,
        )
    except ElGamalRejected:
        return False
    return data.get("verified") is True


def aggregate(cts: List[bytes], weights: List[int]) -> bytes:
    data = _post(
        "/elgamal/aggregate",
        {"cts_b64": [_b64(ct) for ct in cts], "weights": [int(w) for w in weights]},
        zkp_gnark.DEFAULT_VERIFY_TIMEOUT,
    )
    return base64.b64decode(data["ct_b64"])


def decrypt(sk: str, ct: bytes, total_weight: int) -> np.ndarray:
    """Return Σ weight_k · q_k per coordinate."""
    max_abs = total_weight * VALUE_LIMIT
    data = _post(
        "/elgamal/decrypt",
        {"sk": sk, "ct_b64": _b64(ct), "offset_total": str(max_abs), "max_abs": max_abs},
        zkp_gnark.DEFAULT_VERIFY_TIMEOUT,
    )
    return np.asarray(data["values"], dtype=np.int64)


def parse_declared_bounds(proofs, keys) -> Optional[dict]:
    """key → declared bound_sq for each proof, or None if any is malformed or non-positive."""
    out = {}
    try:
        for key, proof in zip(keys, proofs):
            bound = int(str(proof["bound_sq"]))
            if not 0 < bound < zkp_gnark.BN254_R:
                return None
            out[key] = bound
    except (KeyError, TypeError, ValueError):
        return None
    return out
