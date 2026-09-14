"""
Client for the gnark service's verifiable ElGamal endpoints.

Implements the transport side of audit/binding.md Design B + A. Each model
coordinate is quantized to an integer q = round(w · scale) with
|q| < 2^17, encrypted with exponential ElGamal on BabyJubJub, and covered by a
Groth16 proof per chunk whose public inputs are the chunk's ciphertexts, the
public key, the chunk's norm bound and a context value (round, layer, chunk).

Everything the verifier compares against — schema, chunking, bounds, context —
is derived from ``Policy`` and the model schema, never from client metadata.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import requests

import fl.core.zkp_gnark as zkp_gnark

VALUE_LIMIT = 1 << 17  # |q| < 2^17, matches elgamalValueBits in elgamal.go
CIPHERTEXT_BYTES = 64  # compressed C1 ‖ C2
HEADER_MAGIC = 0x45474C31  # "EGL1"
_MAX_INDEX = 1 << 16
_MAX_ROUND = 1 << 32


class ElGamalServiceError(RuntimeError):
    """Service unreachable or failed internally: infrastructure, not a client fault."""


class ElGamalRejected(RuntimeError):
    """Service refused the request contents: malformed input or unsatisfied statement."""


def _post(path: str, payload: dict, read_timeout: float) -> dict:
    url = f"{zkp_gnark.DEFAULT_SERVICE_URL}{path}"
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


@dataclass(frozen=True)
class Policy:
    """Protocol parameters. The server's copy is authoritative."""

    scale: int
    chunk_size: int
    max_norm: float

    @classmethod
    def from_env(cls) -> "Policy":
        return cls(
            scale=int(os.environ.get("FL_ELGAMAL_SCALE", "1000")),
            chunk_size=int(os.environ.get("FL_ELGAMAL_CHUNK", "128")),
            max_norm=float(os.environ.get("FL_ZKP_MAX_NORM", "100.0")),
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


def chunk_bound_sq(policy: Policy, chunk: Chunk, total_numel: int) -> int:
    """Per-chunk share of the model-wide bound Σq² ≤ (max_norm·scale)².

    Shares are proportional to chunk size, so passing every chunk implies the
    model-wide bound. This is stricter than the global bound: an honest update
    whose energy is concentrated in a few chunks can fail.
    """
    total_sq = round(policy.max_norm * policy.scale) ** 2
    bound = total_sq * chunk.size // total_numel
    if bound < 1:
        raise ValueError("norm bound too small for this chunking")
    return bound


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


def prove_chunk(pk_hex: str, q: np.ndarray, bound_sq: int, context: int) -> Tuple[bytes, str]:
    data = _post(
        "/elgamal/prove",
        {
            "pk": pk_hex,
            "values_b64": base64.b64encode(np.asarray(q, dtype="<i8").tobytes()).decode("ascii"),
            "bound_sq": str(bound_sq),
            "context": str(context),
        },
        zkp_gnark.DEFAULT_PROVE_TIMEOUT,
    )
    ct = base64.b64decode(data["ct_b64"])
    if len(ct) != len(q) * CIPHERTEXT_BYTES or not data.get("proof_b64"):
        raise ElGamalServiceError("/elgamal/prove returned a malformed response")
    return ct, data["proof_b64"]


def verify_chunk(pk_hex: str, ct: bytes, bound_sq: int, context: int, proof_b64: str) -> bool:
    """True iff the proof verifies for exactly these ciphertexts and public inputs.

    Malformed client input counts as a failed verification; infrastructure
    failures raise ElGamalServiceError.
    """
    try:
        data = _post(
            "/elgamal/verify",
            {
                "pk": pk_hex,
                "ct_b64": base64.b64encode(ct).decode("ascii"),
                "bound_sq": str(bound_sq),
                "context": str(context),
                "proof_b64": proof_b64,
            },
            zkp_gnark.DEFAULT_VERIFY_TIMEOUT,
        )
    except ElGamalRejected:
        return False
    return data.get("verified") is True


def aggregate(cts: List[bytes], weights: List[int]) -> bytes:
    data = _post(
        "/elgamal/aggregate",
        {
            "cts_b64": [base64.b64encode(ct).decode("ascii") for ct in cts],
            "weights": [int(w) for w in weights],
        },
        zkp_gnark.DEFAULT_VERIFY_TIMEOUT,
    )
    return base64.b64decode(data["ct_b64"])


def decrypt(sk: str, ct: bytes, total_weight: int) -> np.ndarray:
    """Return Σ weight_k · q_k per coordinate."""
    max_abs = total_weight * VALUE_LIMIT
    data = _post(
        "/elgamal/decrypt",
        {
            "sk": sk,
            "ct_b64": base64.b64encode(ct).decode("ascii"),
            "offset_total": str(max_abs),
            "max_abs": max_abs,
        },
        zkp_gnark.DEFAULT_VERIFY_TIMEOUT,
    )
    return np.asarray(data["values"], dtype=np.int64)
