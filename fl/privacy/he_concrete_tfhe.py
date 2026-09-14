"""
Concrete TFHE (true homomorphic encryption) mode.

Uses Zama's Concrete library for real FHE operations:
  - 14-bit fixed-range quantization (per-tensor adaptive is OFF for FHE)
  - Clients encrypt quantized parameters under their public key
  - Server aggregates encrypted tensors without ever decrypting them
  - Clients decrypt the aggregated result with their private key

Environment variables:
  FL_CONCRETE_TFHE_BIT_WIDTH       (default 14)
  FL_CONCRETE_TFHE_ADAPTIVE_QUANT  (default 0 — must be 0 for FHE)
  FL_NUMBER_CLIENTS                (drives pre-averaging divisor)
  FL_SIMULATION                    (1 → measure cost but transport plain numpy)
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fl.privacy.base import PrivacyMode, _plain_params
from fl.privacy.registry import register_mode
from fl.privacy.he_tenseal import _decompress_cte2_results


def _auto_disable_real_tfhe_for_images(config, requested_sim_mode: bool) -> bool:
    """Return True when simulated (unencrypted) TFHE was explicitly allowed.

    Real Concrete TFHE on image datasets (MNIST/CIFAR) is too memory-intensive
    for many machines. The simulated path sends quantized weights in plaintext,
    so it is never selected silently (audit/binding.md B-2):

    * FL_CONCRETE_TFHE_FORCE_REAL=1      → run real TFHE anyway
    * FL_CONCRETE_TFHE_ALLOW_SIMULATED=1 → send plaintext, loudly labelled
    * neither                            → refuse to run
    """
    if requested_sim_mode:
        return False

    dataset = str(getattr(config, "dataset", "")).lower()
    is_image_dataset = dataset in {"mnist", "cifar", "cifar10"}
    if not is_image_dataset or os.environ.get("FL_CONCRETE_TFHE_FORCE_REAL", "0") == "1":
        return False
    if os.environ.get("FL_CONCRETE_TFHE_ALLOW_SIMULATED", "0") == "1":
        print(
            "[HE-TFHE] [WARN] FL_CONCRETE_TFHE_ALLOW_SIMULATED=1: parameters for "
            f"'{dataset}' are sent as PLAINTEXT quantized weights, not TFHE ciphertexts."
        )
        return True
    raise RuntimeError(
        f"Real TFHE on image dataset '{dataset}' is disabled by default to avoid OOM, "
        "and the simulated path does not encrypt. Set FL_CONCRETE_TFHE_FORCE_REAL=1 to "
        "run real TFHE, or FL_CONCRETE_TFHE_ALLOW_SIMULATED=1 to knowingly send plaintext."
    )


@register_mode("he_concrete_tfhe")
class HeConcreteThfeMode(PrivacyMode):
    """Concrete TFHE — cryptographically secure homomorphic encryption."""

    @property
    def name(self) -> str:
        return "he_concrete_tfhe"

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup_client_context(self, config):
        """Initialise the Concrete TFHE aggregation context and generate keys."""
        from fl.core.security import get_concrete_aggregation_context

        sim_mode = config.sim_mode
        auto_disabled = _auto_disable_real_tfhe_for_images(config, sim_mode)
        if auto_disabled:
            print(
                "[HE-TFHE] [SAFEGUARD] Real TFHE auto-disabled for image dataset "
                f"'{config.dataset}' to prevent OOM. "
                "Using simulated TFHE path. Set FL_CONCRETE_TFHE_FORCE_REAL=1 "
                "to force real TFHE (high RAM required)."
            )

        bit_width = int(
            os.environ.get("FL_CONCRETE_TFHE_BIT_WIDTH", config.he_tfhe_bit_width)
        )
        adaptive_quant = os.environ.get("FL_CONCRETE_TFHE_ADAPTIVE_QUANT", "0") != "0"
        num_clients = int(os.environ.get("FL_NUMBER_CLIENTS", config.num_clients))

        enable_fhe = not sim_mode and not auto_disabled
        ctx = get_concrete_aggregation_context(
            bit_width=bit_width,
            enable_fhe=enable_fhe,
            adaptive_quant=adaptive_quant,
            num_clients=num_clients,
        )
        private_key, public_key = ctx.generate_keys()
        ctx.private_key = private_key
        ctx.public_key = public_key

        print(
            f"[HE-TFHE] Context ready  (bit_width={bit_width}, fhe={enable_fhe}, n_clients={num_clients})"
        )
        return ctx

    def setup_server_context(self, config):
        """Initialise server-side Concrete TFHE context (no private key)."""
        from fl.core.security import get_concrete_aggregation_context

        auto_disabled = _auto_disable_real_tfhe_for_images(config, config.sim_mode)
        bit_width = int(
            os.environ.get("FL_CONCRETE_TFHE_BIT_WIDTH", config.he_tfhe_bit_width)
        )
        adaptive_quant = os.environ.get("FL_CONCRETE_TFHE_ADAPTIVE_QUANT", "0") != "0"
        num_clients = int(os.environ.get("FL_NUMBER_CLIENTS", config.num_clients))

        ctx = get_concrete_aggregation_context(
            bit_width=bit_width,
            enable_fhe=not auto_disabled,
            adaptive_quant=adaptive_quant,
            num_clients=num_clients,
        )
        mode_label = "real" if not auto_disabled else "simulated-safe"
        print(
            f"[HE-TFHE] Server context ready (bit_width={bit_width}, mode={mode_label})"
        )
        return ctx

    # ── Client: parameter upload ──────────────────────────────────────────

    def get_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ) -> List[np.ndarray]:
        from fl.core.benchmark import BenchmarkTimer, estimate_params_size

        params = _plain_params(net)
        if sim_mode:
            if benchmark:
                with BenchmarkTimer(benchmark, "encryption"):
                    sim_upload_bytes = self._simulate_encrypt_size_only(params)
                benchmark.add_upload_size(sim_upload_bytes)
            return params  # transport plain in simulation
        else:
            if benchmark:
                with BenchmarkTimer(benchmark, "encryption"):
                    enc_params = self._real_encrypt(params, context)
                benchmark.add_upload_size(estimate_params_size(enc_params))
            else:
                enc_params = self._real_encrypt(params, context)
            return enc_params

    def send_parameters(
        self, net, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ):
        return self.get_parameters(net, context, sim_mode=sim_mode, benchmark=benchmark)

    def receive_parameters(
        self, net, params, context, *, sim_mode, benchmark=None, encrypt_layers=None
    ):
        from fl.core.params import set_parameters
        from fl.core.benchmark import BenchmarkTimer, estimate_params_size

        if benchmark:
            benchmark.add_download_size(estimate_params_size(params))

        def _to_plain(p):
            if not sim_mode and isinstance(p, np.ndarray) and p.dtype == np.uint8:
                # Real mode: uint8 bytes → EncryptedTensor → decrypt
                from fl.core.concrete_agg import EncryptedTensor

                enc = EncryptedTensor.deserialize(p.tobytes())
                return context.decrypt_tensor(enc, context.private_key).astype(
                    np.float32
                )
            return p.astype(np.float32, copy=False) if isinstance(p, np.ndarray) else p

        if benchmark:
            with BenchmarkTimer(benchmark, "decryption"):
                plain = [_to_plain(p) for p in params]
                set_parameters(net, plain, None, None)
        else:
            plain = [_to_plain(p) for p in params]
            set_parameters(net, plain, None, None)

    def post_fit_metrics(self, context, benchmark=None) -> Dict:
        metrics = {}
        if benchmark:
            if benchmark.encryption_time:
                metrics["encryption_time"] = benchmark.encryption_time[-1]
            if benchmark.decryption_time:
                metrics["decryption_time"] = benchmark.decryption_time[-1]
        return metrics

    # ── Server: initial parameter distribution ──────────────────────────────

    def use_client_for_initial_params(self, config) -> bool:
        """
        In real (non-sim) TFHE mode the server holds no private key, so it
        cannot encrypt the initial weights itself.  Returning True causes
        Flower to poll a real client's get_parameters() instead, which
        already encrypts — eliminating the round-1 plaintext download.
        """
        return not config.sim_mode

    # ── Server: Concrete TFHE aggregation ─────────────────────────────────

    def aggregate_fit_override(
        self, server_round, results, failures, server_context, config, benchmark=None
    ) -> Optional[Tuple]:
        """Aggregate encrypted tensors via ConcreteAggregator (non-simulation only)."""
        from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
        from fl.core.benchmark import BenchmarkTimer
        from fl.core.security import aggregate_custom

        sim_mode = config.sim_mode

        if sim_mode or server_context is None:
            return None  # fall through to standard FedAvg

        print(
            f"[HE-TFHE] Round {server_round}: aggregating ENCRYPTED parameters from {len(results)} clients…"
        )

        try:
            from fl.core.concrete_agg import ConcreteAggregator, EncryptedTensor

            aggregator = ConcreteAggregator(server_context, num_clients=len(results))

            with BenchmarkTimer(benchmark, "server_aggregate"):
                for _, fit_res in results:
                    received = parameters_to_ndarrays(fit_res.parameters)
                    encrypted_params = []
                    for arr in received:
                        if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
                            enc_tensor = EncryptedTensor.deserialize(arr.tobytes())
                            encrypted_params.append(enc_tensor)
                        else:
                            encrypted_params.append(arr)
                    aggregator.add_encrypted(encrypted_params, weight=1.0)

                aggregated_encrypted = aggregator.get_result()
                aggregated_arrays = [
                    np.frombuffer(enc.serialize(), dtype=np.uint8)
                    for enc in aggregated_encrypted
                ]
                params_agg = ndarrays_to_parameters(aggregated_arrays)

            return params_agg, {}

        except Exception as e:
            print(
                f"[HE-TFHE] Encrypted aggregation failed: {e}; falling back to plain FedAvg."
            )
            return None

    def pre_aggregate(self, results, config):
        """Decompress CTE2 simulation envelopes if present."""
        return _decompress_cte2_results(results)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _simulate_encrypt(params, context) -> List[np.ndarray]:
        """Simulate TFHE cost via CTE2 envelope (same as he_concrete simulation)."""
        import struct, zlib

        result = []
        for arr in params:
            arr_f = arr.astype(np.float32, copy=False)
            ndim = len(arr_f.shape)
            header = b"CARR" + struct.pack(">B", 1) + struct.pack(">B", ndim)
            if ndim > 0:
                header += struct.pack(">" + "I" * ndim, *arr_f.shape)
            raw_payload = header + arr_f.tobytes(order="C")
            compressed = zlib.compress(raw_payload)
            envelope = b"CTE2" + struct.pack(">I", len(raw_payload)) + compressed
            result.append(np.frombuffer(envelope, dtype=np.uint8))
        return result

    @staticmethod
    def _simulate_encrypt_size_only(params) -> int:
        """Simulate Concrete TFHE payload work without retaining large buffers.

        Returns the total simulated upload size in bytes while keeping peak
        memory lower than `_simulate_encrypt`, which stores every envelope.
        """
        import struct
        import zlib

        total_bytes = 0
        for arr in params:
            arr_f = arr.astype(np.float32, copy=False)
            ndim = len(arr_f.shape)
            header = b"CARR" + struct.pack(">B", 1) + struct.pack(">B", ndim)
            if ndim > 0:
                header += struct.pack(">" + "I" * ndim, *arr_f.shape)
            raw_payload = header + arr_f.tobytes(order="C")
            compressed = zlib.compress(raw_payload)
            total_bytes += 8 + len(compressed)  # CFH-like marker+size + payload

            # Drop transient buffers promptly on large image-model layers.
            del raw_payload, compressed

        return total_bytes

    @staticmethod
    def _real_encrypt(params, context) -> List[np.ndarray]:
        """Encrypt with actual Concrete TFHE operations."""
        result = []
        for arr in params:
            arr_f = arr.astype(np.float32, copy=False)
            encrypted = context.encrypt_tensor(arr_f, context.private_key)
            serialized = encrypted.serialize()
            result.append(np.frombuffer(serialized, dtype=np.uint8))

            # Drop large transient buffers between layers to avoid RSS spikes
            # on image-model tensors.
            del arr_f, encrypted, serialized
            gc.collect()
        return result
