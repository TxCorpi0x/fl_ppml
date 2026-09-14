import base64
import concurrent.futures
import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)
from .zkp_utils import to_numpy, post_json

DEFAULT_SERVICE_URL = os.environ.get("FL_ZKP_SERVICE_URL", "http://127.0.0.1:9000")
DEFAULT_SCALE = float(os.environ.get("FL_ZKP_SCALE", "1000000"))
DEFAULT_MAX_NORM = float(os.environ.get("FL_ZKP_MAX_NORM", "100.0"))
# Increased from 120 to 600: first chunk of a new size triggers gnark circuit
# compilation + groth16 setup which can take several minutes for large layers.
DEFAULT_TIMEOUT = float(os.environ.get("FL_ZKP_TIMEOUT", "600"))
# Stage-specific read timeouts. ``generate_gnark_proofs`` can take much longer
# than verification for first-time circuit sizes on constrained machines.
DEFAULT_PROVE_TIMEOUT = float(
    os.environ.get("FL_ZKP_PROVE_TIMEOUT", os.environ.get("FL_ZKP_TIMEOUT", "1800"))
)
DEFAULT_VERIFY_TIMEOUT = float(
    os.environ.get("FL_ZKP_VERIFY_TIMEOUT", os.environ.get("FL_ZKP_TIMEOUT", "900"))
)
DEFAULT_VERIFY_LIGHT_TIMEOUT = float(
    os.environ.get(
        "FL_ZKP_VERIFY_LIGHT_TIMEOUT", os.environ.get("FL_ZKP_TIMEOUT", "900")
    )
)
DEFAULT_PARALLELISM = int(os.environ.get("FL_ZKP_PARALLELISM", "1"))
DEFAULT_AGGRESSIVE_GC = os.environ.get("FL_ZKP_AGGRESSIVE_GC", "1")
DEFAULT_GC_SLEEP_MS = float(os.environ.get("FL_ZKP_GC_SLEEP_MS", "0"))
# Max elements per ZKP layer chunk.  Layers larger than this are split into
# multiple chunks so the gnark service never compiles a circuit beyond ~664K
# R1CS constraints (~2000 params × 332 constraints/param), which fits in RAM.
DEFAULT_MAX_LAYER_N = int(os.environ.get("FL_ZKP_MAX_LAYER_N", "2000"))


def _current_rss_mb() -> float:
    """Return current process RSS in MB (Linux /proc-based, best effort)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # VmRSS value is in kB
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _safe_collect() -> None:
    """Trigger Python GC when enabled via env (default on for memory safety)."""
    if DEFAULT_AGGRESSIVE_GC != "0":
        gc.collect()


def _default_rss_limit_mb() -> float:
    """Choose a conservative RSS guard from total RAM when env is unset.

    Priority:
      1) FL_ZKP_MAX_RSS_MB env override (if set and >0)
      2) 70% of MemTotal from /proc/meminfo
      3) fixed fallback 4096 MB
    """
    env_v = os.environ.get("FL_ZKP_MAX_RSS_MB", "").strip()
    if env_v:
        try:
            value = float(env_v)
            if value > 0:
                return value
        except Exception:
            pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = float(line.split()[1])
                    # 70% leaves headroom for Flower, PyTorch, gnark, and OS.
                    auto_limit = (total_kb / 1024.0) * 0.70
                    # Keep a sane lower bound even on small machines.
                    return max(2048.0, auto_limit)
    except Exception:
        pass

    return 4096.0


def _estimate_state_dict_bytes(state_dict: Dict[str, np.ndarray]) -> int:
    """Estimate raw parameter bytes in the provided state_dict."""
    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += int(tensor.numel()) * int(tensor.element_size())
        elif isinstance(tensor, np.ndarray):
            total += int(tensor.nbytes)
        else:
            arr = np.asarray(tensor)
            total += int(arr.nbytes)
    return total


def _expand_to_chunks(
    name: str,
    tensor,
    max_n: int,
) -> List[Tuple[str, np.ndarray]]:
    """Flatten *tensor* and return one or more (chunk_name, chunk_array) pairs.

    If the tensor has <= max_n elements it is returned as-is (preserving its
    original shape).  Otherwise the flat array is split into consecutive chunks
    of exactly max_n elements (the last chunk may be smaller) and each chunk is
    named ``"<name>__chunk_<i>"``.
    """
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        arr = np.array(tensor)

    n = arr.size
    if n <= max_n:
        return [(name, arr)]

    flat = arr.flatten()
    chunks = []
    for i, start in enumerate(range(0, n, max_n)):
        chunk = flat[start : start + max_n]
        chunks.append((f"{name}__chunk_{i}", chunk))
    logger.debug(
        "ZKP chunking: layer '%s' (n=%d) → %d chunks of ≤%d",
        name,
        n,
        len(chunks),
        max_n,
    )
    return chunks


def _parse_layer_filter(env_value: str, fallback: List[str]) -> List[str]:
    if not env_value or env_value.upper() == "ALL":
        return fallback
    return [item.strip() for item in env_value.split(",") if item.strip()]


def _http_timeout(read_timeout_s: float):
    """Return (connect_timeout, read_timeout) tuple for requests."""
    # Keep connect timeout short; only read timeout needs to be long for gnark work.
    return (10.0, float(read_timeout_s))


def _quantize_weights(weights: np.ndarray, scale: float) -> np.ndarray:
    return np.round(weights.astype(np.float64, copy=False) * scale).astype(np.int64)


def _encode_weights(weights: np.ndarray) -> Tuple[str, List[int]]:
    encoded = base64.b64encode(weights.tobytes(order="C")).decode("ascii")
    return encoded, list(weights.shape)


def _build_payload(
    weights: np.ndarray, scale: float, bound_sq: int, layer_name: str
) -> Dict:
    quantized = _quantize_weights(weights, scale)
    weights_b64, shape = _encode_weights(quantized)
    # Validate that the encoded byte length matches expected element size (int64)
    try:
        decoded = base64.b64decode(weights_b64)
    except Exception as exc:
        raise ValueError(
            f"invalid base64 encoding for weights of layer {layer_name}: {exc}"
        ) from exc
    expected_bytes = int(np.prod(shape)) * 8
    actual_bytes = len(decoded)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"weights byte length mismatch for layer {layer_name}: expected {expected_bytes}, got {actual_bytes}"
        )
    return {
        "layer_name": layer_name,
        "weights_b64": weights_b64,
        "shape": shape,
        "scale": str(scale),
        "bound_sq": str(bound_sq),
    }


def generate_gnark_proofs(
    state_dict: Dict[str, np.ndarray],
    layers: Optional[List[str]] = None,
    service_url: Optional[str] = None,
    scale: Optional[float] = None,
    max_norm: Optional[float] = None,
    timeout: Optional[float] = None,
) -> Tuple[List[Dict], int]:
    """
    Generate per-layer proofs using the gnark service.

    Returns:
        proofs: list of proof payloads (JSON-serializable)
        proof_bytes: total encoded proof size in bytes
    """
    service_url = service_url or DEFAULT_SERVICE_URL
    scale = scale if scale is not None else DEFAULT_SCALE
    max_norm = max_norm if max_norm is not None else DEFAULT_MAX_NORM
    timeout = timeout if timeout is not None else DEFAULT_PROVE_TIMEOUT

    layer_names = list(state_dict.keys())
    protected_layers = _parse_layer_filter(
        os.environ.get("FL_ZKP_LAYERS", "ALL"), layer_names
    )
    if layers is not None:
        protected_layers = [name for name in protected_layers if name in layers]

    bound_sq = int((max_norm * scale) ** 2)

    proofs: List[Dict] = []
    total_bytes = 0

    # Prepare a worker that posts a single layer proof and returns payload + size
    def _prove_one(item):
        name, tensor = item
        payload = None
        data = None
        response = None
        weights = None
        # Handle both torch tensors and numpy arrays
        try:
            weights = to_numpy(tensor)

            payload = _build_payload(weights, scale, bound_sq, name)
            data = post_json(f"{service_url}/prove", payload, _http_timeout(timeout))
            proof_b64 = data.get("proof_b64")
            hash_hex = data.get("hash_hex")
            # The circuit size is fixed by the weights we sent, so the shape
            # comes from our own request. The service does not echo it; if a
            # service version does, it must agree.
            shape = payload["shape"]
            echoed_shape = data.get("shape")
            if echoed_shape is not None and list(echoed_shape) != list(shape):
                raise RuntimeError(
                    f"gnark /prove returned shape {echoed_shape} for layer '{name}', "
                    f"but {shape} was sent"
                )
            # Validate required response fields
            if not proof_b64:
                logger.error(
                    "gnark /prove returned empty proof for layer %s: %s",
                    name,
                    data,
                )
                server_err = data.get("error") if isinstance(data, dict) else None
                raise RuntimeError(
                    f"Missing proof for layer '{name}': {server_err or data}"
                )
            if not hash_hex:
                raise RuntimeError(
                    f"Missing 'hash_hex' in proof response for layer '{name}': {data}"
                )
            if not shape:
                raise RuntimeError(f"Refusing to prove empty tensor for layer '{name}'")

            proof_payload = {
                "layer": name,
                "shape": shape,
                "scale": payload["scale"],
                "bound_sq": payload["bound_sq"],
                "proof_b64": proof_b64,
                "hash_hex": hash_hex,
            }
            return proof_payload, len(proof_b64)
        except RuntimeError:
            # re-raise our own informative runtime errors
            raise
        except Exception as exc:
            raise RuntimeError(
                f"gnark proof generation failed for layer '{name}': {exc}"
            ) from exc
        finally:
            # Explicitly drop large per-layer buffers (quantized arrays/base64 payloads)
            # before the next layer to reduce transient RSS spikes on image models.
            del payload, data, response, weights
            _safe_collect()

    # Build items to prove — large layers are split into chunks so the gnark
    # service never has to compile a circuit with an unbounded number of R1CS
    # constraints (e.g. fc1.weight with 48 000 params → ~16M constraints → OOM).
    max_n = DEFAULT_MAX_LAYER_N
    items: List[Tuple[str, np.ndarray]] = []
    for name in protected_layers:
        if name not in state_dict:
            continue
        items.extend(_expand_to_chunks(name, state_dict[name], max_n))
    if not items:
        return proofs, total_bytes

    # Parallelism configuration
    max_workers = max(1, DEFAULT_PARALLELISM)
    rss_limit_mb = _default_rss_limit_mb()

    # Optional RSS-aware worker throttling for memory-sensitive runs.
    if max_workers > 1:
        current_rss = _current_rss_mb()
        state_bytes = _estimate_state_dict_bytes(state_dict)
        # Heuristic transient multiplier: quantization + encoding + request JSON copies.
        est_peak_per_worker_mb = max(
            64.0, (state_bytes / max(1, len(items))) * 16 / (1024.0 * 1024.0)
        )
        headroom_mb = max(0.0, rss_limit_mb - current_rss)
        safe_workers = int(
            max(1, min(max_workers, headroom_mb // est_peak_per_worker_mb))
        )
        if safe_workers < max_workers:
            logger.warning(
                "ZKP memory guard: reducing FL_ZKP_PARALLELISM from %s to %s "
                "(rss=%.1fMB, limit=%.1fMB, est_peak/worker=%.1fMB)",
                max_workers,
                safe_workers,
                current_rss,
                rss_limit_mb,
                est_peak_per_worker_mb,
            )
            max_workers = safe_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_prove_one, it): it[0] for it in items}
        for fut in concurrent.futures.as_completed(futures):
            layer_name = futures[fut]
            try:
                proof_payload, size = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"gnark proof generation failed for layer '{layer_name}': {exc}"
                )
            proofs.append(proof_payload)
            total_bytes += size

            # Per-layer memory checkpoint: collect and optionally abort before kernel OOM.
            _safe_collect()
            rss_now = _current_rss_mb()
            if rss_now > rss_limit_mb:
                raise MemoryError(
                    "ZKP memory guard triggered: RSS "
                    f"{rss_now:.1f}MB exceeds limit={rss_limit_mb:.1f}MB. "
                    "Reduce protected layers (e.g. FL_ZKP_NUM_LAYERS=1), "
                    "set FL_ZKP_PARALLELISM=1, or use zkp_sampled for image datasets."
                )

            # Optional tiny pause to let allocator/GC settle in constrained environments.
            if DEFAULT_GC_SLEEP_MS > 0:
                try:
                    time.sleep(DEFAULT_GC_SLEEP_MS / 1000.0)
                except Exception:
                    pass

    return proofs, total_bytes


BN254_R = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001


class GnarkServiceError(RuntimeError):
    """The proof service is unreachable or failed: an infrastructure fault, not a client fault."""


def _post_service(url: str, payload: Dict, read_timeout: float) -> Dict:
    """POST a verification request.

    Connection errors, timeouts, 5xx and non-JSON bodies raise
    GnarkServiceError. A 4xx means the service refused client-supplied
    contents, so it is returned as an unverified result.
    """
    try:
        resp = requests.post(url, json=payload, timeout=_http_timeout(read_timeout))
    except requests.RequestException as exc:
        raise GnarkServiceError(f"{url}: {exc}") from exc
    if resp.status_code >= 500:
        raise GnarkServiceError(f"{url}: HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise GnarkServiceError(f"{url}: non-JSON response") from exc
    if not isinstance(data, dict):
        raise GnarkServiceError(f"{url}: unexpected response type {type(data).__name__}")
    if 400 <= resp.status_code < 500:
        return {"verified": False, "error": data.get("error", f"HTTP {resp.status_code}")}
    return data


def expected_proof_layout(
    schema: List[Tuple[str, Tuple[int, ...]]], max_n: Optional[int] = None
) -> Dict[str, List[int]]:
    """Proof name → shape that a complete upload must cover.

    Mirrors the chunking in generate_gnark_proofs, computed from the server's
    own model schema rather than client metadata.
    """
    max_n = max_n or DEFAULT_MAX_LAYER_N
    layout: Dict[str, List[int]] = {}
    for name, shape in schema:
        n = int(np.prod(shape)) if len(shape) else 1
        if n <= max_n:
            layout[name] = [int(d) for d in shape]
        else:
            for i, start in enumerate(range(0, n, max_n)):
                layout[f"{name}__chunk_{i}"] = [min(max_n, n - start)]
    return layout


def policy_bound_sq(scale: Optional[float] = None, max_norm: Optional[float] = None) -> int:
    """The server's bound on Σq², using the same formula as generate_gnark_proofs."""
    scale = DEFAULT_SCALE if scale is None else scale
    max_norm = DEFAULT_MAX_NORM if max_norm is None else max_norm
    return int((max_norm * scale) ** 2)


def check_proof_policy(
    proofs,
    schema: List[Tuple[str, Tuple[int, ...]]],
    *,
    require_hash: bool,
    scale: Optional[float] = None,
    max_norm: Optional[float] = None,
) -> Optional[str]:
    """Return a rejection reason if a proof set doesn't match server policy, else None.

    Checks exact coverage of the server's model, per-proof shape, the scale and
    bound the server enforces, and (for light verification) a canonical hash.
    """
    if not isinstance(proofs, list) or not proofs or not all(isinstance(p, dict) for p in proofs):
        return "proofs must be a non-empty list of objects"
    layout = expected_proof_layout(schema)
    names = [p.get("layer") for p in proofs]
    if len(set(names)) != len(names):
        return "duplicate proofs for the same layer or chunk"
    if set(names) != set(layout):
        missing = sorted(set(layout) - set(names))
        unexpected = sorted(str(n) for n in set(names) - set(layout))
        return f"proof coverage mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}"

    scale = DEFAULT_SCALE if scale is None else scale
    bound = policy_bound_sq(scale, max_norm)
    for p in proofs:
        name = p["layer"]
        try:
            shape = [int(d) for d in p.get("shape") or []]
            proof_scale = float(p["scale"])
            proof_bound = int(str(p["bound_sq"]))
        except (KeyError, TypeError, ValueError):
            return f"proof '{name}' is malformed"
        if shape != layout[name]:
            return f"proof '{name}' has shape {shape}, server expects {layout[name]}"
        if proof_scale != float(scale) or proof_bound != bound:
            return f"proof '{name}' uses scale {proof_scale} / bound {proof_bound}, not the server policy"
        if not isinstance(p.get("proof_b64"), str) or not p["proof_b64"]:
            return f"proof '{name}' has no proof bytes"
        if require_hash:
            try:
                h = int(str(p.get("hash_hex")), 16)
            except ValueError:
                return f"proof '{name}' has a malformed hash"
            if not 0 <= h < BN254_R:
                return f"proof '{name}' hash is not a canonical field element"
    return None


def verify_gnark_proofs(
    parameters: List[np.ndarray],
    layer_names: List[str],
    proofs: List[Dict],
    service_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Verify per-layer proofs against the parameters the verifier holds.

    The MiMC hash public input is recomputed from ``parameters``, so a proof
    only verifies for the exact values it was generated over. Malformed proof
    entries and incomplete chunk coverage count as failures; only an
    unreachable or failing service raises (GnarkServiceError).

    Returns:
        ok: True if every layer is fully covered and all proofs verify
        failures: layer or chunk names that failed
    """
    service_url = service_url or DEFAULT_SERVICE_URL
    timeout = timeout if timeout is not None else DEFAULT_VERIFY_TIMEOUT

    proof_map = {
        item["layer"]: item
        for item in proofs or []
        if isinstance(item, dict) and isinstance(item.get("layer"), str)
    }
    failures: List[str] = []

    def _verify_one(key: str, values: np.ndarray, proof: Dict) -> bool:
        try:
            shape = [int(d) for d in proof["shape"]]
            payload = _build_payload(
                np.asarray(values).reshape(shape),
                float(proof["scale"]),
                int(str(proof["bound_sq"])),
                key,
            )
            payload["proof_b64"] = str(proof["proof_b64"])
        except (KeyError, TypeError, ValueError):
            logger.warning("verify: malformed proof for %s", key)
            return False
        if not payload["proof_b64"]:
            return False
        data = _post_service(f"{service_url}/verify", payload, timeout)
        return data.get("verified") is True

    for name, weights in zip(layer_names, parameters):
        arr = np.asarray(weights)
        chunk_keys = sorted(
            (k for k in proof_map if k.startswith(f"{name}__chunk_") and k.rsplit("_", 1)[-1].isdigit()),
            key=lambda k: int(k.rsplit("_", 1)[-1]),
        )
        if chunk_keys:
            flat = arr.reshape(-1)
            offset = 0
            items = []
            for ck in chunk_keys:
                try:
                    size = int(np.prod([int(d) for d in proof_map[ck]["shape"]]))
                except (KeyError, TypeError, ValueError):
                    size = -1
                if size <= 0 or offset + size > flat.size:
                    failures.append(ck)
                    items = None
                    break
                items.append((ck, flat[offset : offset + size], proof_map[ck]))
                offset += size
            if items is None:
                continue
            if offset != flat.size:
                failures.append(f"{name} (chunks cover {offset} of {flat.size} values)")
                continue
        elif name in proof_map:
            items = [(name, arr, proof_map[name])]
        else:
            failures.append(name)
            continue

        for key, values, proof in items:
            if not _verify_one(key, values, proof):
                failures.append(key)

    return len(failures) == 0, failures


def verify_gnark_proofs_light(
    proofs: List[Dict],
    service_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Verify per-layer Groth16 proofs using only their public inputs.

    No weights are needed: each proof carries ``hash_hex`` (MiMC of the
    proved vector), ``bound_sq`` and ``shape``. This proves a statement about a
    client-chosen vector; nothing ties that vector to anything the server
    aggregates (audit/binding.md). Malformed or non-canonical entries count as
    failures; only an unreachable or failing service raises
    (GnarkServiceError). Coverage and policy are checked separately by
    check_proof_policy.

    Returns:
        (ok, failures) — ok is True iff every proof verified.
    """
    service_url = service_url or DEFAULT_SERVICE_URL
    timeout = timeout if timeout is not None else DEFAULT_VERIFY_LIGHT_TIMEOUT

    failures: List[str] = []

    for proof in proofs or []:
        name = proof.get("layer", "<unknown>") if isinstance(proof, dict) else "<malformed>"
        try:
            shape = [int(d) for d in proof["shape"]]
            bound_sq = int(str(proof["bound_sq"]))
            hash_int = int(str(proof["hash_hex"]), 16)
            proof_b64 = str(proof["proof_b64"])
        except (KeyError, TypeError, ValueError, AttributeError):
            failures.append(str(name))
            continue
        if not (shape and proof_b64) or not 0 < bound_sq < BN254_R or not 0 <= hash_int < BN254_R:
            failures.append(str(name))
            continue

        payload = {
            "layer_name": str(name),
            "shape": shape,
            "bound_sq": str(bound_sq),
            "hash_hex": str(proof["hash_hex"]),
            "proof_b64": proof_b64,
        }
        data = _post_service(f"{service_url}/verify_light", payload, timeout)
        if data.get("verified") is not True:
            logger.warning("verify_light: proof invalid for layer %s: %s", name, data.get("error", ""))
            failures.append(str(name))

    return len(failures) == 0, failures
