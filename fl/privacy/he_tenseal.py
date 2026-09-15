"""
TenSEAL CKKS Homomorphic Encryption mode.

Client-side:
  Each client holds a TenSEAL context with the secret key.
  Parameters are CKKS-encrypted before upload.
  Parameters received from the server are decrypted locally.

Server-side:
  The server holds the public TenSEAL context (no secret key).
  Aggregation is performed under encryption (weighted average of CKKSTensors).
  The resulting encrypted aggregate is sent back to clients for local decryption.

Simulation mode (FL_SIMULATION=1):
  Encryption cost is measured via BenchmarkTimer but plain numpy arrays are
  transported.  This is safe for in-process Flower simulation which cannot
  serialise arbitrary Python objects across simulated network boundaries.
"""

from __future__ import annotations

import gc
import io
import os
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

# ── Chunking constants ─────────────────────────────────────────────────────
# gRPC on macOS truncates single recvmsg calls above ~256 MB.  Large CKKS
# tensors are split into ≤ _CKKS_CHUNK_SIZE chunks, each prefixed with an
# 8-byte CCHK header, and reassembled on the receiving side.
# 48 MB is chosen conservatively: Flower's gRPC max_message_length defaults
# to 536_870_912 (512 MB) but the OS recv buffer is typically ~256 MB on
# macOS, and creditcard CKKS layers can exceed 450 MB uncompressed.
_CKKS_CHUNK_SIZE: int = 48 * 1024 * 1024  # 48 MB per chunk
_CCHK_MAGIC: bytes = b"CCHK"  # 4-byte magic identifier

import numpy as np

from fl.privacy.base import PrivacyMode, _plain_params
from fl.privacy.registry import register_mode


# ── Chunk helpers (module-level so they're usable everywhere) ─────────────


def _pack_raw(raw: bytes) -> List[np.ndarray]:
    """Return *raw* as a list of uint8 arrays safe for gRPC transport.

    Small payloads (< _CKKS_CHUNK_SIZE) are zlib-compressed and returned as
    a single uint8 array.  Large payloads are split into ≤ _CKKS_CHUNK_SIZE
    chunks WITHOUT zlib (encrypted data is incompressible) and each chunk is
    prefixed with an 8-byte ``CCHK total_chunks chunk_idx`` header array so
    the receiver knows how to reassemble them.
    """
    if len(raw) < _CKKS_CHUNK_SIZE:
        compressed = zlib.compress(raw)
        return [np.frombuffer(compressed, dtype=np.uint8).copy()]
    # Large: split into chunks, skip zlib (no benefit for random CKKS bytes)
    chunk_bytes = [
        raw[i : i + _CKKS_CHUNK_SIZE] for i in range(0, len(raw), _CKKS_CHUNK_SIZE)
    ]
    total = len(chunk_bytes)
    result: List[np.ndarray] = []
    for idx, chunk in enumerate(chunk_bytes):
        header = np.frombuffer(
            _CCHK_MAGIC + struct.pack(">HH", total, idx), dtype=np.uint8
        ).copy()
        result.append(header)
        result.append(np.frombuffer(chunk, dtype=np.uint8).copy())
    return result


def _unpack_arrays(flat_params: List[np.ndarray]) -> List[np.ndarray]:
    """Inverse of layer-wise ``_pack_raw``: reassemble chunked sequences.

    Scans *flat_params* (a Flower parameter list) for CCHK headers and
    concatenates the associated data chunks.  Non-chunked entries pass through
    unchanged.
    """
    result: List[np.ndarray] = []
    i = 0
    while i < len(flat_params):
        arr = flat_params[i]
        b = (
            arr.tobytes()
            if (isinstance(arr, np.ndarray) and arr.dtype == np.uint8)
            else b""
        )
        if len(b) == 8 and b[:4] == _CCHK_MAGIC:
            total, idx = struct.unpack(">HH", b[4:])
            if idx == 0 and i + 1 < len(flat_params):
                chunk_data: List[bytes] = [flat_params[i + 1].tobytes()]
                j = i + 2
                while len(chunk_data) < total and j + 1 < len(flat_params):
                    hdr_b = (
                        flat_params[j].tobytes()
                        if (
                            isinstance(flat_params[j], np.ndarray)
                            and flat_params[j].dtype == np.uint8
                            and len(flat_params[j]) == 8
                        )
                        else b""
                    )
                    if hdr_b[:4] == _CCHK_MAGIC:
                        _, cidx = struct.unpack(">HH", hdr_b[4:])
                        if cidx == len(chunk_data):
                            chunk_data.append(flat_params[j + 1].tobytes())
                            j += 2
                            continue
                    break  # unexpected — stop here
                # Strict validation: all chunks must be present.
                # A partial join produces truncated CKKS bytes that look like
                # valid protobuf to TenSEAL but fail internal zlib decompression
                # (stream decompression failed).  Raise early with diagnostics.
                if len(chunk_data) != total:
                    raise RuntimeError(
                        f"[HE-TenSEAL] CCHK chunk reassembly incomplete: "
                        f"received {len(chunk_data)}/{total} chunks for a "
                        f"{total * _CKKS_CHUNK_SIZE // (1024*1024)} MB tensor. "
                        f"Likely gRPC recvmsg truncation (EMSGSIZE). "
                        f"Try reducing FL_ENCRYPT_LAYERS or lowering "
                        f"_CKKS_CHUNK_SIZE further."
                    )
                full = b"".join(chunk_data)
                result.append(np.frombuffer(full, dtype=np.uint8).copy())
                i = j
                continue
        result.append(arr)
        i += 1
    return result


@register_mode("he_tenseal")
class HeTensealMode(PrivacyMode):
    """TenSEAL CKKS Homomorphic Encryption."""

    @property
    def name(self) -> str:
        return "he_tenseal"

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_client_context(self, config):
        """Load or create TenSEAL context with the secret key."""
        from fl.core.security import make_tenseal_context
        from fl.keys.he_tenseal import load_client, write_context

        secret_path = config.he_tenseal_secret_path
        if os.path.exists(secret_path):
            ctx = load_client(secret_path)
            print(f"[HE-TenSEAL] Client context loaded from {secret_path}")
        else:
            ctx = make_tenseal_context()
            write_context(secret_path, ctx.serialize(save_secret_key=True))
            print(f"[HE-TenSEAL] New client context created → {secret_path}")

        return ctx

    def setup_server_context(self, config):
        """Load public TenSEAL context (no secret key) for server-side aggregation."""
        from fl.keys.he_tenseal import load_server

        public_path = config.he_tenseal_public_path
        if not os.path.exists(public_path):
            if config.sim_mode:
                print("[HE-TenSEAL] No server public key found; simulation mode transports plaintext.")
                return None
            # Without a context the server would FedAvg raw ciphertext bytes
            # (audit/failmodes.md E-1).
            raise FileNotFoundError(
                f"TenSEAL public key not found: {public_path}\n"
                "Run: python -m fl.keys generate he_tenseal"
            )

        ctx = load_server(public_path)
        print(f"[HE-TenSEAL] Server context loaded from {public_path}")
        return ctx

    # ── Server: initial parameter distribution ────────────────────────────

    def use_client_for_initial_params(self, config) -> bool:
        """
        Server holds no secret key, so it cannot encrypt initial weights.
        Returning True causes Flower to poll a client's get_parameters()
        which already CKKS-encrypts — eliminating the round-1 plaintext leak.
        """
        return not config.sim_mode

    # ── Client: parameter upload ──────────────────────────────────────────

    def get_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> List[np.ndarray]:
        """Encrypt (or simulate encryption of) parameters for upload."""
        from fl.core.security import crypte
        from fl.core.benchmark import BenchmarkTimer, estimate_params_size

        if sim_mode:
            if benchmark:
                with BenchmarkTimer(benchmark, "encryption"):
                    _ = crypte(net.state_dict(), context, encrypt_layers)
            params = _plain_params(net)
        else:
            if benchmark:
                with BenchmarkTimer(benchmark, "encryption"):
                    params = self._encrypt_params(net, context, encrypt_layers)
            else:
                params = self._encrypt_params(net, context, encrypt_layers)

        if benchmark:
            benchmark.add_upload_size(estimate_params_size(params))
        return params

    def send_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ):
        return self.get_parameters(
            net,
            context,
            sim_mode=sim_mode,
            benchmark=benchmark,
            encrypt_layers=encrypt_layers,
        )

    def receive_parameters(
        self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ):
        """Decrypt server parameters (or apply plain params in simulation)."""
        from fl.core.params import set_parameters
        from fl.core.benchmark import BenchmarkTimer, estimate_params_size

        if benchmark:
            benchmark.add_download_size(estimate_params_size(params))

        if sim_mode:
            if benchmark:
                with BenchmarkTimer(benchmark, "decryption"):
                    # Simulate decryption cost without actually doing it
                    try:
                        sample = next(iter(net.state_dict().values())).detach()
                        from fl.core.security import crypte

                        _ = crypte({"sample": sample}, context, None)
                        _ = context.secret_key()
                    except Exception:
                        pass
            set_parameters(
                net, [p.astype(np.float32, copy=False) for p in params], None, None
            )
        else:
            # Reassemble any CCHK-chunked arrays before decryption. A failure
            # raises: training on stale local weights while reporting success
            # would hide it (audit/failmodes.md E-2).
            params = _unpack_arrays(list(params))
            if benchmark:
                with BenchmarkTimer(benchmark, "decryption"):
                    set_parameters(net, params, context, None, he_backend="tenseal")
            else:
                set_parameters(net, params, context, None, he_backend="tenseal")

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        metrics = {}
        if benchmark:
            if benchmark.encryption_time:
                metrics["encryption_time"] = benchmark.encryption_time[-1]
            if benchmark.decryption_time:
                metrics["decryption_time"] = benchmark.decryption_time[-1]
        return metrics

    # ── Server: aggregation under encryption ──────────────────────────────

    def aggregate_fit_override(
        self, server_round, results, failures, server_context, config, benchmark=None
    ) -> Optional[Tuple]:
        """Aggregate CKKS-encrypted parameters on the server (non-simulation only)."""
        import tenseal as ts
        from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
        from fl.core.benchmark import BenchmarkTimer
        from fl.core.security import (
            aggregate_custom,
            _CVEC_MAGIC,
            _parse_cvec,
            _make_cvec,
        )

        sim_mode = config.sim_mode

        if sim_mode:
            return None  # simulation transports plaintext; standard FedAvg
        if server_context is None:
            raise RuntimeError(
                "[HE-TenSEAL] no server context in non-simulation mode; refusing to FedAvg ciphertext bytes"
            )

        print(
            f"[HE-TenSEAL] Round {server_round}: aggregating encrypted parameters from {len(results)} clients…"
        )

        enc_results = []
        total_enc_size = 0
        layer_shapes: Optional[list] = None  # shape metadata from first client

        for client_proxy, fit_res in results:
            # Reassemble any chunked arrays before CKKS loading
            received = _unpack_arrays(parameters_to_ndarrays(fit_res.parameters))
            tensors = []
            shapes = []  # per-layer: tuple or None
            for arr in received:
                if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
                    enc_bytes = arr.tobytes()
                    total_enc_size += len(enc_bytes)
                    # Attempt zlib decompress (small arrays are compressed)
                    try:
                        raw = zlib.decompress(enc_bytes)
                    except Exception:
                        raw = enc_bytes  # large CKKS chunks arrive uncompressed
                    # CVEC-prefixed ckks_vector (new compact format)
                    if raw[:4] == _CVEC_MAGIC:
                        shape, ckks_bytes = _parse_cvec(raw)
                        tensors.append(ts.ckks_vector_from(server_context, ckks_bytes))
                        shapes.append(shape)
                    # Plain numpy .npy fallback
                    elif raw[:6] == b"\x93NUMPY":
                        plain = np.load(io.BytesIO(raw), allow_pickle=False)
                        tensors.append(plain)
                        shapes.append(None)
                    else:
                        # Legacy ckks_tensor path; raise on corruption
                        try:
                            tensors.append(ts.ckks_tensor_from(server_context, raw))
                            shapes.append(None)
                        except Exception as ckks_err:
                            raise RuntimeError(
                                f"[HE-TenSEAL] CKKS deserialisation failed and payload "
                                f"is not a numpy .npy file (first 8 bytes: "
                                f"{raw[:8].hex()}). Likely gRPC chunk corruption. "
                                f"Original error: {ckks_err}"
                            ) from ckks_err
                else:
                    tensors.append(arr)
                    shapes.append(None)
            enc_results.append((tensors, fit_res.num_examples))
            if layer_shapes is None:
                layer_shapes = shapes

        print(
            f"[HE-TenSEAL] Received {total_enc_size / (1024 * 1024):.2f} MB encrypted"
        )

        num_examples_total = float(sum(n for _, n in enc_results))
        norm_weights = [float(n) / num_examples_total for _, n in enc_results]
        num_layers = len(enc_results[0][0])
        aggregated = []

        with BenchmarkTimer(benchmark, "server_aggregate"):
            for layer_idx in range(num_layers):
                acc = None
                for (tensors, _), alpha in zip(enc_results, norm_weights):
                    term = tensors[layer_idx] * alpha
                    acc = term if acc is None else acc + term
                    del term
                # Serialize back; use _pack_raw so large CKKS layers are chunked
                if hasattr(acc, "serialize"):
                    raw_agg = acc.serialize()
                    # Re-attach CVEC header so clients can reshape correctly
                    if layer_shapes and layer_shapes[layer_idx] is not None:
                        raw_agg = _make_cvec(layer_shapes[layer_idx], raw_agg)
                    print(
                        f"[HE-TenSEAL] layer {layer_idx}: agg size={len(raw_agg)} bytes"
                        f" chunks={max(1, (len(raw_agg) + _CKKS_CHUNK_SIZE - 1) // _CKKS_CHUNK_SIZE)}"
                    )
                else:
                    buf = io.BytesIO()
                    np.save(buf, np.asarray(acc, dtype=np.float32))
                    raw_agg = buf.getvalue()
                aggregated.extend(_pack_raw(raw_agg))
                del acc
                if (layer_idx + 1) % 5 == 0:
                    gc.collect()

            parameters_aggregated = ndarrays_to_parameters(aggregated)

        del enc_results
        gc.collect()

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def pre_aggregate(self, results, config):
        """In simulation, decompress any CTE2-encoded uint8 payloads → float32."""
        return _decompress_cte2_results(results)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _encrypt_params(self, net, context, encrypt_layers=None) -> List[np.ndarray]:
        """Return CKKS-encrypted parameter arrays as uint8 numpy buffers.

        ``crypte()`` returns a **list** of Layer/CryptedLayer objects in the
        same order as ``state_dict()``.  Each CryptedLayer.weight_array is a
        ``ts.CKKSTensor``; call its ``.serialize()`` to get raw bytes, then
        zlib-compress for transport.

        Only layers listed in *encrypt_layers* (or an explicitly set
        ``FL_ENCRYPT_LAYERS`` env var) are CKKS-encrypted; remaining layers are
        sent as plain float32 arrays. With neither set, every layer is
        encrypted (audit/binding.md B-1: the previous default silently
        encrypted only ``model.0.*``, and nothing at all on CNNs).
        """
        from fl.core.security import crypte, _make_cvec

        if encrypt_layers is None:
            env_val = os.environ.get("FL_ENCRYPT_LAYERS", "ALL").strip()
            if env_val and env_val.upper() != "ALL":
                encrypt_layers = [s.strip() for s in env_val.split(",") if s.strip()]
                missing = sorted(set(encrypt_layers) - set(net.state_dict()))
                if missing:
                    raise ValueError(
                        f"FL_ENCRYPT_LAYERS names layers not in the model: {missing}"
                    )

        layers = crypte(net.state_dict(), context, encrypt_layers)
        result = []
        for layer in layers:
            wa = layer.weight_array
            if hasattr(wa, "serialize"):
                # CryptedLayer: ts.CKKSVector (or legacy CKKSTensor) → bytes
                raw = wa.serialize()
                # Prefix with CVEC header so receivers know the original shape
                # and can call ckks_vector_from() for efficient deserialisation.
                shape = getattr(layer, "_original_shape", None)
                if shape is not None:
                    raw = _make_cvec(shape, raw)
            else:
                # Plain Layer: ndarray → bytes via numpy
                buf = io.BytesIO()
                np.save(buf, wa)
                raw = buf.getvalue()
            result.extend(_pack_raw(raw))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Shared: decompress Concrete-emulation / TenSEAL simulation uint8 blobs
# ─────────────────────────────────────────────────────────────────────────────


def _decompress_cte2_results(results):
    """
    Convert any uint8 CTE2-envelope or numpy .npy blobs back to float32
    arrays so FedAvg weighted average works correctly.

    Expects Flower's ``[(ClientProxy, FitRes), ...]`` format and returns
    the same structure with decompressed parameters written back into each
    ``FitRes``.
    """
    import struct
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    decompressed = []
    for client, fit_res in results:
        params = parameters_to_ndarrays(fit_res.parameters)
        fixed = []
        for p in params:
            if isinstance(p, np.ndarray) and p.dtype == np.uint8:
                raw = p.tobytes()
                try:
                    if raw.startswith(b"CTE2") and len(raw) >= 8:
                        payload = zlib.decompress(raw[8:])
                        if not payload.startswith(b"CARR"):
                            raise ValueError("invalid CARR")
                        ndim = payload[5]
                        offset = 6
                        shape = struct.unpack(
                            ">" + "I" * ndim, payload[offset : offset + 4 * ndim]
                        )
                        offset += 4 * ndim
                        arr = np.frombuffer(payload[offset:], dtype=np.float32).reshape(
                            shape
                        )
                        fixed.append(arr.astype(np.float32, copy=False))
                        continue
                    if raw.startswith(b"\x93NUMPY"):
                        arr = np.load(io.BytesIO(raw), allow_pickle=False)
                        fixed.append(arr.astype(np.float32, copy=False))
                        continue
                    # Legacy zlib+npy
                    arr = np.load(io.BytesIO(zlib.decompress(raw)), allow_pickle=False)
                    fixed.append(arr.astype(np.float32, copy=False))
                    continue
                except Exception as exc:
                    # FedAvg over raw bytes is never meaningful (audit/failmodes.md E-3).
                    raise ValueError(
                        f"client {getattr(client, 'cid', '?')}: uint8 parameter is not a decodable "
                        "plaintext envelope; refusing to average ciphertext or corrupt bytes"
                    ) from exc
            fixed.append(p)
        fit_res.parameters = ndarrays_to_parameters(fixed)
        decompressed.append((client, fit_res))
    return decompressed
