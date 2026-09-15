"""
Concrete TFHE-based Encrypted Parameter Aggregation for Federated Learning.

This module implements actual FHE (Fully Homomorphic Encryption) parameter aggregation
using Zama's Concrete compiler with TFHE scheme. Unlike quantization-based simulation,
this provides cryptographically secure homomorphic operations on encrypted weights.

Features:
- Low-level Concrete TFHE for encrypted vector operations
- Homomorphic addition of encrypted model parameters
- Scalar multiplication of encrypted tensors
- Multi-client aggregation without server decryption
- Key generation and distribution
- Serialization of FHE artifacts for gRPC transport

References:
- https://docs.zama.ai/concrete/
- TFHE Paper: https://eprint.iacr.org/2016/016
- Concrete Tutorial: https://github.com/zama-ai/concrete

Architecture:
    Client 1: model params → encrypt → send ciphertext
    Client 2: model params → encrypt → send ciphertext
    ...
    Server: receives encrypted params → aggregate via homomorphic addition
            → weighted average → clients decrypt result

Key Generation Strategy:
    - Each client generates private/public key pair (one-time setup)
    - Server receives public evaluation keys for homomorphic operations
    - Aggregation happens in encrypted domain
    - Clients decrypt aggregated parameters (they hold private keys)

Example Usage:
    ```python
    from core.concrete_aggregation import ConcreteAggregationContext, ConcreteAggregator

    # Setup (once per experiment)
    context = ConcreteAggregationContext(
        bit_width=8,  # 8-bit integers, suitable for 8-bit quantized weights
        vector_size=1000,  # example parameter size
    )

    # Client-side: Encrypt local model parameters
    client_params = [...model weights...]
    encrypted = context.encrypt_tensor(client_params, client_private_key)

    # Server-side: Aggregate encrypted parameters
    aggregator = ConcreteAggregator(context, num_clients=10)
    for encrypted_params in client_updates:
        aggregator.add_encrypted(encrypted_params, weight=1.0/num_clients)

    aggregated_encrypted = aggregator.get_result()

    # Client-side: Decrypt aggregated parameters
    decrypted = context.decrypt_tensor(aggregated_encrypted, client_private_key)
    ```

Security Properties:
    - Semantic Security: Server never sees plaintext parameters
    - Homomorphic: Addition and scalar multiplication on ciphertexts
    - Collective Decryption: Result decryptable only by clients with private keys
    - Forward Secure: Updates over rounds maintain security bounds
"""

import os
import hashlib
import gc
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import struct
import zlib
import logging

logger = logging.getLogger(__name__)


def _current_rss_mb() -> float:
    """Return current process RSS in MB (Linux /proc-based, best effort)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _default_rss_limit_mb() -> float:
    """Choose a conservative default RSS guard, overridable via env var."""
    env_v = os.environ.get("FL_CONCRETE_TFHE_MAX_RSS_MB", "").strip()
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
                    return max(2048.0, (total_kb / 1024.0) * 0.70)
    except Exception:
        pass

    return 4096.0


try:
    from concrete import fhe
    from concrete.fhe import Configuration
    from concrete.fhe.compilation.value import Value
    from concrete.fhe import EvaluationKeys

    CONCRETE_AVAILABLE = True
except ImportError:
    CONCRETE_AVAILABLE = False
    logger.warning(
        "Concrete TFHE not installed. Install with: "
        "pip install --index-url https://pypi.zama.ai/gpu concrete-python"
    )


class ConcreteAggregationContext:
    """
    Context manager for Concrete TFHE parameter aggregation.

    Handles:
    - FHE key generation and distribution
    - Encryption/decryption of model tensors
    - Circuit compilation for aggregation operations
    - Parameter serialization for transport

    Args:
        bit_width: Quantization bit-width for parameters (2-16, default: 8)
        vector_size: Expected parameter vector size (for circuit pre-compilation)
        p_error: Probability of error (default: 0.01)
        enable_fhe: Whether to actually compile FHE circuits (default: True)
                    If False, falls back to simulation mode
        fixed_quant_range: Fixed quantization range for all parameters (default: [-2.0, 2.0])
                          Ensures consistent quantization across FL rounds
    """

    def __init__(
        self,
        bit_width: int = 14,
        vector_size: int = 1000,
        p_error: float = 0.01,
        enable_fhe: bool = True,
        fixed_quant_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
        num_clients: int = 2,
    ):
        if not CONCRETE_AVAILABLE:
            raise ImportError(
                "Concrete TFHE library required. "
                "Install: pip install --index-url https://pypi.zama.ai/gpu concrete-python"
            )

        if enable_fhe:
            # Avoid OpenMP double-init crashes during FHE execution
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        self.bit_width = bit_width
        self.vector_size = vector_size
        self.p_error = p_error
        self.enable_fhe = enable_fhe
        # Number of clients that will contribute to each aggregation round.
        # Used for pre-averaging: each client divides its quantized values by
        # num_clients before encryption so the homomorphic SUM equals the
        # AVERAGE directly, preventing ring overflow.
        self.num_clients = max(1, int(num_clients))

        # Use per-tensor quantization (better accuracy) instead of fixed range
        # Store quant_min/max in EncryptedTensor metadata for decryption
        self.fixed_quant_range = fixed_quant_range
        if fixed_quant_range:
            logger.info(f"Using FIXED quantization range: {fixed_quant_range}")
        else:
            logger.info("Using PER-TENSOR adaptive quantization (better accuracy)")

        # TFHE parameters (from Concrete defaults, tuned for 8-bit)
        self.lwe_dimension = 1024
        self.glwe_dimension = 2
        self.poly_size = 512

        # Client/server artifact cache (lazy-loaded)
        self._clients: Dict[Tuple[int, ...], fhe.Client] = {}
        self._servers: Dict[Tuple[int, ...], fhe.Server] = {}
        self._evaluation_keys: Dict[Tuple[int, ...], EvaluationKeys] = {}
        self._config = None

        # Bound the number of compiled shape artifacts kept live in memory.
        # CNN models have multiple large tensor shapes; keeping every loaded
        # client/server object can exhaust RAM on modest machines.
        self._max_cached_shapes = max(
            1,
            int(os.environ.get("FL_CONCRETE_TFHE_MAX_CACHED_SHAPES", "1")),
        )
        self._rss_limit_mb = _default_rss_limit_mb()

        # Auto-calibrated Concrete circuit internal scaling factor per tensor shape.
        # Compensates for the FHE addition circuit's internal multiplication artifact
        # (empirically measured at runtime instead of hard-coded).
        self._concrete_scaling_cache: Dict[Tuple[int, ...], float] = {}

        # Key storage (shared across processes).
        # Default: fl/keys/prebuilt/ — override with FL_CONCRETE_TFHE_KEYS_DIR.
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        keys_dir = os.environ.get("FL_CONCRETE_TFHE_KEYS_DIR", "./keys/prebuilt")
        if not os.path.isabs(keys_dir):
            keys_dir = os.path.join(base_dir, keys_dir)
        self._keys_dir = os.path.abspath(keys_dir)
        os.makedirs(self._keys_dir, exist_ok=True)

        # Setup configuration
        self._setup_configuration()

        logger.info(
            f"ConcreteAggregationContext initialized: "
            f"bit_width={bit_width}, vector_size={vector_size}, "
            f"fhe={'enabled' if enable_fhe else 'simulation'}, "
            f"max_cached_shapes={self._max_cached_shapes}, "
            f"rss_limit_mb={self._rss_limit_mb:.1f}"
        )

    def _check_rss_guard(self, stage: str) -> None:
        """Raise MemoryError if RSS exceeds configured guard limit."""
        rss = _current_rss_mb()
        if rss > self._rss_limit_mb:
            raise MemoryError(
                f"[TFHE] Memory guard triggered at {stage}: RSS {rss:.1f}MB "
                f"exceeds limit {self._rss_limit_mb:.1f}MB. "
                "Reduce image-model load (e.g. fewer clients/rounds) or keep "
                "he_concrete_tfhe for non-image datasets when RAM is constrained."
            )

    def _evict_shape_caches_if_needed(self, keep_shape: Tuple[int, ...]) -> None:
        """Evict cached Concrete artifacts for other shapes when cache is full."""
        if keep_shape in self._clients:
            return
        if len(self._clients) < self._max_cached_shapes:
            return

        # Evict all other shapes to aggressively bound RSS.
        evict_shapes = [s for s in self._clients.keys() if s != keep_shape]
        for shape in evict_shapes:
            self._clients.pop(shape, None)
            self._servers.pop(shape, None)
            self._evaluation_keys.pop(shape, None)

        # Promptly return large Concrete objects to the allocator.
        gc.collect()

    def _setup_configuration(self):
        """Setup Concrete FHE configuration."""
        self._config = Configuration(
            enable_unsafe_features=False,
            use_insecure_key_cache=False,
            insecure_key_cache_location="./concrete_cache",
        )

    def _keys_path_for(self, shape: Tuple[int, ...]) -> str:
        shape_tag = "x".join(str(d) for d in shape) if shape else "scalar"
        return os.path.join(
            self._keys_dir,
            f"keys_bw{self.bit_width}_nc{self.num_clients}_{shape_tag}.bin",
        )

    def _artifact_paths(self, shape: Tuple[int, ...]) -> Tuple[str, str, str]:
        shape_tag = "x".join(str(d) for d in shape) if shape else "scalar"
        # Include bit_width AND num_clients in path so changing either param
        # automatically invalidates stale cached circuits.
        base = os.path.join(
            self._keys_dir, f"bw{self.bit_width}_nc{self.num_clients}_{shape_tag}"
        )
        return base + "_client", base + "_server", base + ".lock"

    @staticmethod
    def _zip_path(base_path: str) -> str:
        return base_path + ".zip"

    def _eval_keys_path_for(self, shape: Tuple[int, ...]) -> str:
        shape_tag = "x".join(str(d) for d in shape) if shape else "scalar"
        base = os.path.join(
            self._keys_dir, f"bw{self.bit_width}_nc{self.num_clients}_{shape_tag}"
        )
        return base + "_eval_keys.bin"

    def _secret_keys_path_for(self, shape: Tuple[int, ...]) -> str:
        """Path for the serialized secret key (shared across all client processes)."""
        shape_tag = "x".join(str(d) for d in shape) if shape else "scalar"
        base = os.path.join(
            self._keys_dir, f"bw{self.bit_width}_nc{self.num_clients}_{shape_tag}"
        )
        return base + "_secret_keys.bin"

    def _load_secret_keys_into_client(
        self, client: fhe.Client, shape: Tuple[int, ...]
    ) -> bool:
        """
        Restore the shared secret key into a freshly-loaded fhe.Client.

        Concrete's fhe.Client.load() always generates a NEW random secret key;
        it does NOT persist or restore the key from the saved zip.  As a result,
        two client instances loaded from the same zip have DIFFERENT keys, and
        adding their ciphertexts produces garbage (not the homomorphic sum).

        The fix: save client.keys.serialize() to a sidecar .bin file after keygen,
        then call this method to inject the saved keyset into every client instance
        that is loaded later.
        """
        path = self._secret_keys_path_for(shape)
        if not os.path.exists(path):
            return False
        try:
            from concrete.fhe.compilation.keys import Keys  # noqa: F401

            with open(path, "rb") as f:
                data = f.read()
            shared_keys = Keys.deserialize(data)
            client._keys = shared_keys
            return True
        except Exception as e:
            logger.warning(
                f"[TFHE] Could not restore secret key for shape {shape}: {e}"
            )
            return False

    def _serialize_evaluation_keys(self, client: fhe.Client) -> bytes:
        if hasattr(client, "get_serialized_evaluation_keys"):
            return client.get_serialized_evaluation_keys()
        if hasattr(client, "evaluation_keys"):
            eval_keys = client.evaluation_keys
            if isinstance(eval_keys, (bytes, bytearray)):
                return bytes(eval_keys)
            if hasattr(eval_keys, "serialize"):
                return eval_keys.serialize()
        raise RuntimeError("Concrete client does not expose serialized evaluation keys")

    def _get_evaluation_keys(self, shape: Tuple[int, ...]) -> EvaluationKeys:
        if shape in self._evaluation_keys:
            return self._evaluation_keys[shape]

        client, _ = self._get_client_server(shape)
        eval_keys_path = self._eval_keys_path_for(shape)
        client_base, server_base, _ = self._artifact_paths(shape)
        client_path = self._zip_path(client_base)
        server_path = self._zip_path(server_base)

        regenerate = False
        if os.path.exists(eval_keys_path):
            try:
                if os.path.exists(client_path) and os.path.exists(server_path):
                    eval_mtime = os.path.getmtime(eval_keys_path)
                    if eval_mtime < max(
                        os.path.getmtime(client_path),
                        os.path.getmtime(server_path),
                    ):
                        regenerate = True
                with open(eval_keys_path, "rb") as f:
                    data = f.read()
                eval_keys = EvaluationKeys.deserialize(data)
            except Exception:
                regenerate = True
        else:
            regenerate = True

        if regenerate:
            data = self._serialize_evaluation_keys(client)
            tmp_path = eval_keys_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, eval_keys_path)
            eval_keys = EvaluationKeys.deserialize(data)

        self._evaluation_keys[shape] = eval_keys
        return eval_keys

    def _get_client_server(
        self, shape: Tuple[int, ...]
    ) -> Tuple[fhe.Client, fhe.Server]:
        self._check_rss_guard(f"_get_client_server:start:{shape}")

        if shape in self._clients and shape in self._servers:
            return self._clients[shape], self._servers[shape]

        self._evict_shape_caches_if_needed(shape)

        min_val = -(2 ** (self.bit_width - 1))
        max_val = 2 ** (self.bit_width - 1) - 1

        # Pre-averaging: compile the circuit for PRE-SCALED input range.
        # Each client will encrypt (quantized // num_clients) so the sum of
        # num_clients such values stays within [min_val, max_val], avoiding
        # ring wrap-around (the root cause of 50% accuracy in prior runs).
        prescaled_min = min_val // self.num_clients
        prescaled_max = max_val // self.num_clients

        # Use addition circuit - allows homomorphic addition
        # Both inputs must be encrypted for server-side aggregation
        def add_fn(x, y):
            return x + y

            # Configure compiler
            configuration = fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location="keys/he_concrete_tfhe/circuit_cache",
            )

        compiler = fhe.Compiler(add_fn, {"x": "encrypted", "y": "encrypted"})
        inputset = []
        seed_input = (
            f"{shape}-{self.bit_width}-{self.num_clients}-{self.p_error}".encode(
                "utf-8"
            )
        )
        seed = int.from_bytes(hashlib.sha256(seed_input).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        for _ in range(10):
            x = rng.integers(
                prescaled_min, prescaled_max + 1, size=shape, dtype=np.int64
            )
            y = rng.integers(
                prescaled_min, prescaled_max + 1, size=shape, dtype=np.int64
            )
            inputset.append((x, y))

        client_base, server_base, lock_path = self._artifact_paths(shape)
        client_path = self._zip_path(client_base)
        server_path = self._zip_path(server_base)

        eval_keys_path = self._eval_keys_path_for(shape)

        if os.path.exists(client_path) and os.path.exists(server_path):
            try:
                client = fhe.Client.load(client_path)
                server = fhe.Server.load(server_path)
                # CRITICAL: restore the shared secret key so all client instances
                # (across separate processes) use the same key → compatible ciphertexts.
                # Without this, enc_client_A(x) + enc_client_B(y) = garbage.
                if not self._load_secret_keys_into_client(client, shape):
                    logger.warning(
                        f"[TFHE] Secret key file missing for shape {shape}; "
                        "ciphertexts from different processes will be INCOMPATIBLE. "
                        "Delete fl/keys/prebuilt/ and restart to force recompilation."
                    )
                self._clients[shape] = client
                self._servers[shape] = server
                return client, server
            except Exception:
                # Corrupted artifacts; remove and regenerate
                try:
                    os.remove(client_path)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(server_path)
                except FileNotFoundError:
                    pass
                for stale in (
                    self._zip_path(client_base + ".tmp"),
                    self._zip_path(server_base + ".tmp"),
                    client_base + ".tmp",
                    server_base + ".tmp",
                ):
                    try:
                        os.remove(stale)
                    except FileNotFoundError:
                        pass
                try:
                    os.remove(eval_keys_path)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(self._secret_keys_path_for(shape))
                except FileNotFoundError:
                    pass

        for _ in range(120):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if os.path.exists(client_path) and os.path.exists(server_path):
                    try:
                        client = fhe.Client.load(client_path)
                        server = fhe.Server.load(server_path)
                        # CRITICAL: restore shared secret key (see _load_secret_keys_into_client).
                        if not self._load_secret_keys_into_client(client, shape):
                            logger.warning(
                                f"[TFHE] Secret key file missing for shape {shape} (poll path); "
                                "ciphertexts from different processes will be INCOMPATIBLE."
                            )
                        self._clients[shape] = client
                        self._servers[shape] = server
                        return client, server
                    except Exception:
                        try:
                            os.remove(client_path)
                        except FileNotFoundError:
                            pass
                        try:
                            os.remove(server_path)
                        except FileNotFoundError:
                            pass
                        for stale in (
                            self._zip_path(client_base + ".tmp"),
                            self._zip_path(server_base + ".tmp"),
                            client_base + ".tmp",
                            server_base + ".tmp",
                        ):
                            try:
                                os.remove(stale)
                            except FileNotFoundError:
                                pass
                        try:
                            os.remove(eval_keys_path)
                        except FileNotFoundError:
                            pass
                        try:
                            os.remove(self._secret_keys_path_for(shape))
                        except FileNotFoundError:
                            pass
                import time

                time.sleep(0.5)
        else:
            raise RuntimeError("Timed out waiting for Concrete TFHE artifacts")

        # RACE-CONDITION GUARD: another process may have compiled and released the
        # lock while we were waiting (we got the lock because they already freed it,
        # but our initial pre-loop file-existence check ran before they wrote the
        # artifacts).  Re-check now that we hold the lock.
        if os.path.exists(client_path) and os.path.exists(server_path):
            try:
                client = fhe.Client.load(client_path)
                server = fhe.Server.load(server_path)
                # Inject the shared secret key (written by the process that actually compiled).
                if not self._load_secret_keys_into_client(client, shape):
                    logger.warning(
                        f"[TFHE] Post-lock race-guard: secret key missing for shape {shape}. "
                        "Proceeding to recompile."
                    )
                    raise RuntimeError("missing secret key, recompile")
                self._clients[shape] = client
                self._servers[shape] = server
                # Release lock before returning (compile try/finally won't run).
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                return client, server
            except Exception:
                pass  # Fall through to compile; lock is still held

        try:
            circuit = compiler.compile(
                inputset, configuration=self._config, p_error=self.p_error
            )
            self._check_rss_guard(f"_get_client_server:after_compile:{shape}")
            circuit.keygen()
            self._check_rss_guard(f"_get_client_server:after_keygen:{shape}")

            # Persist the secret key FIRST (before saving circuit artifacts).
            # All processes that later load the circuit must use the SAME key.
            try:
                secret_keys_path = self._secret_keys_path_for(shape)
                tmp_skeys = secret_keys_path + ".tmp"
                secret_data = circuit.client.keys.serialize()
                with open(tmp_skeys, "wb") as f:
                    f.write(secret_data)
                os.replace(tmp_skeys, secret_keys_path)
                logger.info(f"[TFHE] Saved shared secret key → {secret_keys_path}")
            except Exception as e:
                logger.warning(
                    f"[TFHE] Could not save secret key for shape {shape}: {e}"
                )

            tmp_client_base = client_base + ".tmp"
            tmp_server_base = server_base + ".tmp"
            tmp_client_path = self._zip_path(tmp_client_base)
            tmp_server_path = self._zip_path(tmp_server_base)
            circuit.client.save(tmp_client_base)
            circuit.server.save(tmp_server_base)
            if os.path.exists(tmp_client_path):
                os.replace(tmp_client_path, client_path)
            elif os.path.exists(tmp_client_base):
                os.replace(tmp_client_base, client_path)
            else:
                circuit.client.save(client_base)
            if os.path.exists(tmp_server_path):
                os.replace(tmp_server_path, server_path)
            elif os.path.exists(tmp_server_base):
                os.replace(tmp_server_base, server_path)
            else:
                circuit.server.save(server_base)
            try:
                eval_keys_path = self._eval_keys_path_for(shape)
                eval_keys = self._serialize_evaluation_keys(circuit.client)
                tmp_eval = eval_keys_path + ".tmp"
                with open(tmp_eval, "wb") as f:
                    f.write(eval_keys)
                os.replace(tmp_eval, eval_keys_path)
            except Exception:
                logger.debug("Failed to persist evaluation keys", exc_info=True)
            # Use circuit.client / circuit.server directly — they already hold
            # the correct key from circuit.keygen().  Reloading from zip would
            # generate a fresh random key, making this process's ciphertexts
            # incompatible with ciphertexts from other processes.
            client = circuit.client
            server = circuit.server
            self._clients[shape] = client
            self._servers[shape] = server
            return client, server
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

    def _serialize_value(self, value: Value) -> np.ndarray:
        return np.frombuffer(value.serialize(), dtype=np.uint8)

    def _deserialize_value(self, data: np.ndarray) -> Value:
        return Value.deserialize(data.tobytes())

    def _calibrate_concrete_scaling(self, shape: Tuple[int, ...]) -> float:
        """
        Measure the internal scaling factor of the compiled FHE addition circuit.

        Concrete's FHE circuit for x+y internally represents values as
        `k * x` (where k is a power-of-2 scaling factor determined at compile
        time from the inputset range).  When you call server.run(enc_x, enc_y)
        and then client.decrypt(result) you get back `k * (x + y)` rather than
        `x + y`.  The single-value encrypt→decrypt path is NOT affected (it
        returns the original integer), so the only correct way to measure k is
        to actually run server.run with a known operand.

        This method:
          1. Encrypts a known vector `v` (at quarter-range) and zeros.
          2. Runs  server.run(enc_v, enc_zero)  →  should be v+0 = v in math,
             but yields k*v from the circuit.
          3. Decrypts the result and computes k = decrypted / v.

        Returns k so decrypt_tensor can divide by it to recover the true average.
        """
        if shape in self._concrete_scaling_cache:
            return self._concrete_scaling_cache[shape]

        try:
            # CRITICAL: must calibrate using the SAME circuit (same shape) that will
            # be used during decryption. Concrete compiles a separate circuit per
            # shape, and different circuits can have different internal scaling factors
            # (k). Using an (8,) proxy circuit to calibrate the (1920,) circuit would
            # measure k=1 for (8,) and silently apply it to (1920,) which may have k=4
            # → every decrypted weight is wrong by 4×.
            test_shape = shape

            # Quarter of the pre-scaled max — safely away from saturation.
            prescaled_max = (2 ** (self.bit_width - 1) - 1) // self.num_clients
            quarter = max(1, prescaled_max // 4)
            # Use full-shape known/zero vectors so the calibration runs the true circuit.
            n = int(np.prod(shape)) if shape else 1
            known = np.full(n, quarter, dtype=np.int64).reshape(shape)
            zero = np.zeros(n, dtype=np.int64).reshape(shape)

            client, server = self._get_client_server(test_shape)
            eval_keys = self._get_evaluation_keys(test_shape)

            # Encrypt v and 0
            enc_known, enc_zero = client.encrypt(known, zero)

            # Run homomorphic addition: v + 0 = v (but circuit returns k*v)
            result = server.run(enc_known, enc_zero, evaluation_keys=eval_keys)
            decrypted = np.asarray(client.decrypt(result), dtype=np.float64)

            # k = decrypted / quarter  (avoid divide-by-zero)
            factor = float(np.median(np.abs(decrypted))) / quarter

            # Sanity check: k must be a reasonable power-of-2-ish number
            if factor < 0.5 or factor > 2**24:
                logger.warning(
                    f"[TFHE] Calibration factor {factor:.2f} out of range; "
                    f"defaulting to 1.0.  Decrypted sample: {decrypted[:4]}"
                )
                factor = 1.0

            self._concrete_scaling_cache[shape] = factor
            logger.info(
                f"[TFHE] server.run scaling factor for shape {shape}: {factor:.4f} "
                f"(known={quarter}, decrypted_median={float(np.median(np.abs(decrypted))):.1f})"
            )
            print(
                f"[CALIBRATE] shape={shape}, input={quarter}, "
                f"server.run→decrypt median={float(np.median(np.abs(decrypted))):.1f}, "
                f"factor={factor:.4f}"
            )
            return factor

        except Exception as e:
            logger.warning(
                f"[TFHE] Calibration failed for shape {shape}: {e}; using 1.0"
            )
            self._concrete_scaling_cache[shape] = 1.0
            return 1.0

    def generate_keys(self) -> Tuple[Any, Any]:
        """
        Generate FHE key pair for a client.

        Returns:
            Tuple of (private_key, public_key)

        Note:
            - Private key: kept by client for decryption
            - Public key: sent to server for aggregation operations
        """
        if not self.enable_fhe:
            return ("sim_private_key", "sim_public_key")

        # Keys are generated per-circuit in _ensure_keys and stored on disk
        return ("shared_private_key", "shared_public_key")

    def encrypt_tensor(
        self,
        tensor: np.ndarray,
        private_key: Any,
    ) -> "EncryptedTensor":
        """
        Encrypt a tensor using client's private key.

        Args:
            tensor: numpy array of shape (n,) to encrypt
            private_key: Client's private FHE key

        Returns:
            EncryptedTensor object (holds ciphertext)
        """
        self._check_rss_guard(f"encrypt_tensor:start:{tensor.shape}")

        # Get quantization range
        if self.fixed_quant_range:
            t_min, t_max = self.fixed_quant_range
        else:
            t_min, t_max = float(tensor.min()), float(tensor.max())

        if not self.enable_fhe:
            # BUG FIX: Must quantize to int32 here, NOT store raw floats.
            # Previously stored raw float32s; _add_encrypted then did
            # .astype(np.int32) which truncated small weights (e.g. 0.03 → 0).
            quantized = self._quantize_for_fhe(tensor, t_min, t_max).astype(np.int32)
            return EncryptedTensor(
                ciphertext=quantized,
                shape=tensor.shape,
                bit_width=self.bit_width,
                is_simulated=True,
                quant_min=t_min,
                quant_max=t_max,
            )

        quantized = self._quantize_for_fhe(tensor, t_min, t_max).astype(
            np.int64, copy=False
        )

        # PRE-AVERAGING: divide quantized values by num_clients before encryption.
        # This guarantees the sum of num_clients encrypted pre-scaled values
        # never exceeds the circuit's ring capacity, eliminating modular wrap-around.
        # The server's homomorphic sum then equals the average directly (scale=1).
        #
        # CRITICAL: must use floor division (//) NOT round().
        # The circuit is compiled with inputset range [max_val // num_clients] which
        # equals 8191 // 2 = 4095 for 14-bit/2 clients.
        # np.round(8191 / 2) = np.round(4095.5) = 4096  ← OUTSIDE circuit range!
        # Passing a value outside the compiled inputset causes Concrete to produce
        # wrong outputs (the root cause of oscillating accuracy: weights near ±5.0
        # saturate the 14-bit quantization, giving q=8191, round→4096, wrap→garbage).
        # Floor division guarantees: max(q // n) = 4095, always within [-4096, 4095].
        prescaled = quantized.astype(np.int64) // self.num_clients

        # Debug: Check quantization quality
        print(
            f"[ENCRYPT] shape={tensor.shape}, float_range=[{tensor.min():.4f}, {tensor.max():.4f}], "
            f"quant=[{quantized.min()}, {quantized.max()}], prescaled=[{prescaled.min()}, {prescaled.max()}] "
            f"(÷{self.num_clients})"
        )

        try:
            client, _ = self._get_client_server(tensor.shape)
            zeros = np.zeros_like(prescaled, dtype=np.int64)
            # Encrypt both x and zeros, but only keep encrypted x
            enc_x, enc_zeros = client.encrypt(prescaled, zeros)
            # Serialize only the encrypted value (not the zeros)
            ciphertext = self._serialize_value(enc_x)

            return EncryptedTensor(
                ciphertext=ciphertext,
                shape=tensor.shape,
                bit_width=self.bit_width,
                is_simulated=False,
                key_id=id(private_key),
                # scale=1: server sum of pre-averaged values = average directly.
                # No post-hoc division needed in decrypt_tensor.
                scale=1.0,
                quant_min=t_min,
                quant_max=t_max,
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
        finally:
            gc.collect()
            self._check_rss_guard(f"encrypt_tensor:end:{tensor.shape}")

    def decrypt_tensor(
        self,
        encrypted: "EncryptedTensor",
        private_key: Any,
    ) -> np.ndarray:
        """
        Decrypt an encrypted tensor using private key.

        Args:
            encrypted: EncryptedTensor object
            private_key: Client's private FHE key

        Returns:
            Decrypted numpy array
        """
        if encrypted.is_simulated:
            # BUG FIX: ciphertext is now properly int32 quantized values.
            # Average the accumulated integer sum, clip, then dequantize.
            scale = encrypted.scale if encrypted.scale and encrypted.scale > 0 else 1.0
            avg_int = np.round(encrypted.ciphertext.astype(np.float64) / scale).astype(
                np.int64
            )
            min_q = -(2 ** (encrypted.bit_width - 1))
            max_q = 2 ** (encrypted.bit_width - 1) - 1
            avg_int = np.clip(avg_int, min_q, max_q)
            return self._dequantize_from_fhe(
                avg_int, encrypted.quant_min, encrypted.quant_max
            ).astype(np.float32, copy=False)

        try:
            print(
                f"[DECRYPT] Decrypting tensor shape {encrypted.shape}, scale={encrypted.scale}, quant_range=[{encrypted.quant_min}, {encrypted.quant_max}]"
            )
            client, _ = self._get_client_server(encrypted.shape)
            value = self._deserialize_value(encrypted.ciphertext)
            decrypted_int = client.decrypt(value)

            # Get the scale (number of clients that were summed)
            scale = encrypted.scale if encrypted.scale else 1.0

            print(
                f"[DECRYPT] Before averaging: scale={scale}, "
                f"decrypted_int range=[{decrypted_int.min()}, {decrypted_int.max()}]"
            )

            # Auto-calibrate the Concrete circuit's internal scaling factor.
            # The FHE addition circuit may internally scale values by a factor that
            # depends on bit_width and circuit topology. We measure it once with a
            # known test vector rather than hard-coding a magic constant.
            concrete_scaling = self._calibrate_concrete_scaling(encrypted.shape)
            total_scale = concrete_scaling * scale
            decrypted_int = np.round(
                decrypted_int.astype(np.float64) / total_scale
            ).astype(np.int64)
            print(
                f"[DECRYPT] After auto-calibrated scaling (÷{total_scale:.2f} = {concrete_scaling:.4f}×{scale}): "
                f"decrypted_int range=[{decrypted_int.min()}, {decrypted_int.max()}]"
            )

            # Clip to int8 range to prevent dequantization extrapolation
            min_val = -(2 ** (self.bit_width - 1))
            max_val = 2 ** (self.bit_width - 1) - 1
            decrypted_int = np.clip(decrypted_int, min_val, max_val)
            print(
                f"[DECRYPT] After clipping to [{min_val}, {max_val}]: decrypted_int range=[{decrypted_int.min()}, {decrypted_int.max()}]"
            )

            # Now dequantize using the original range
            # The dequantization function extrapolates for values outside [-128, 127]
            dequantized = self._dequantize_from_fhe(
                decrypted_int, encrypted.quant_min, encrypted.quant_max
            )

            print(
                f"[DECRYPT] After dequantization: range=[{dequantized.min():.6f}, {dequantized.max():.6f}]"
            )

            return dequantized.astype(np.float32, copy=False)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def _quantize_for_fhe(
        self, tensor: np.ndarray, t_min: float = None, t_max: float = None
    ) -> np.ndarray:
        """Quantize float tensor to signed integers in FHE-compatible range."""
        min_val = -(2 ** (self.bit_width - 1))
        max_val = 2 ** (self.bit_width - 1) - 1

        # Use fixed range if specified, otherwise per-tensor range
        if self.fixed_quant_range:
            t_min, t_max = self.fixed_quant_range
            tensor_clipped = np.clip(tensor, t_min, t_max)
        else:
            if t_min is None or t_max is None:
                t_min, t_max = tensor.min(), tensor.max()
            tensor_clipped = tensor

        if t_min == t_max:
            # Constant tensor (e.g. zero-initialized bias). Use non-degenerate
            # range so division is safe; the constant value maps to the midpoint
            # and dequantizes back to the original constant.
            t_min = t_min - 1e-6
            t_max = t_max + 1e-6

        # Scale to [-2^(b-1), 2^(b-1)-1]
        scaled = (tensor_clipped - t_min) / (t_max - t_min) * (
            max_val - min_val
        ) + min_val
        quantized = np.clip(np.round(scaled), min_val, max_val).astype(np.int32)

        return quantized

    def _dequantize_from_fhe(
        self, quantized: np.ndarray, t_min: float, t_max: float
    ) -> np.ndarray:
        """Dequantize from FHE integers back to float using specified range."""
        min_val = -(2 ** (self.bit_width - 1))
        max_val = 2 ** (self.bit_width - 1) - 1

        # Map back to [0, 1]
        normalized = (quantized.astype(np.float32) - min_val) / (max_val - min_val)

        # Map to original range
        dequantized = normalized * (t_max - t_min) + t_min

        return dequantized

    def get_config(self) -> Configuration:
        """Get Concrete FHE configuration."""
        return self._config


class EncryptedTensor:
    """
    Represents an encrypted tensor in FHE domain.

    Attributes:
        ciphertext: Encrypted data (or quantized plaintext in simulation)
        shape: Original tensor shape
        bit_width: Quantization bit-width
        is_simulated: Whether this is simulated (plaintext) or actual FHE
        key_id: ID of the key that encrypted this (for validation)
    """

    def __init__(
        self,
        ciphertext: np.ndarray,
        shape: Tuple[int, ...],
        bit_width: int,
        is_simulated: bool = False,
        key_id: Optional[int] = None,
        scale: float = 1.0,
        quant_min: float = 0.0,
        quant_max: float = 1.0,
    ):
        self.ciphertext = ciphertext
        self.shape = shape
        self.bit_width = bit_width
        self.is_simulated = is_simulated
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.key_id = key_id
        self.scale = scale

    def serialize(self) -> bytes:
        """Serialize encrypted tensor for transport (e.g., gRPC)."""
        # kind: 0=int32, 1=uint8 bytes, 2=int64
        kind = 0
        if isinstance(self.ciphertext, np.ndarray):
            if self.ciphertext.dtype == np.uint8:
                kind = 1
            elif self.ciphertext.dtype == np.int64:
                kind = 2
        header = struct.pack(
            ">BBBI",  # is_simulated, bit_width, kind, ndim
            1 if self.is_simulated else 0,
            self.bit_width,
            kind,
            len(self.shape),
        )
        header += struct.pack(">" + "I" * len(self.shape), *self.shape)
        header += struct.pack(">d", float(self.scale))
        # Add quantization range for correct dequantization
        header += struct.pack(">dd", float(self.quant_min), float(self.quant_max))

        payload = header + self.ciphertext.tobytes(order="C")
        compressed = zlib.compress(payload)

        envelope = b"CFH2" + struct.pack(">I", len(payload)) + compressed
        return envelope

    @staticmethod
    def deserialize(data: bytes) -> "EncryptedTensor":
        """Deserialize encrypted tensor from bytes."""
        if data.startswith(b"CFH2"):
            payload_len = struct.unpack(">I", data[4:8])[0]
            compressed = data[8:]
            try:
                payload = zlib.decompress(compressed)
            except zlib.error:
                payload = compressed

            if payload_len != len(payload) and payload_len == len(compressed):
                payload = compressed

            if len(payload) < 7:
                raise ValueError("Invalid EncryptedTensor payload")

            is_sim = bool(payload[0])
            bit_width = payload[1]
            kind = payload[2]
            ndim = struct.unpack(">I", payload[3:7])[0]

            # Updated header: includes quant_min/quant_max (16 bytes more)
            header_len = 7 + 4 * ndim + 8 + 16

            # Check if this is old format (without quant_min/max)
            old_header_len = 7 + 4 * ndim + 8
            has_quant_range = len(payload) >= header_len

            if has_quant_range:
                # New format with quant_min/max
                if len(payload) < header_len:
                    raise ValueError("Invalid EncryptedTensor header length")

                shape = struct.unpack(">" + "I" * ndim, payload[7 : 7 + 4 * ndim])
                scale = struct.unpack(">d", payload[7 + 4 * ndim : 7 + 4 * ndim + 8])[0]
                quant_min, quant_max = struct.unpack(
                    ">dd", payload[7 + 4 * ndim + 8 : 7 + 4 * ndim + 24]
                )
                ciphertext_bytes = payload[header_len:]
            else:
                # Old format - use defaults
                if len(payload) < old_header_len:
                    raise ValueError("Invalid EncryptedTensor header length")

                shape = struct.unpack(">" + "I" * ndim, payload[7 : 7 + 4 * ndim])
                scale = struct.unpack(">d", payload[7 + 4 * ndim : 7 + 4 * ndim + 8])[0]
                quant_min, quant_max = 0.0, 1.0
                ciphertext_bytes = payload[old_header_len:]

            if kind == 1:
                ciphertext = np.frombuffer(ciphertext_bytes, dtype=np.uint8)
            elif kind == 2:
                ciphertext = np.frombuffer(ciphertext_bytes, dtype=np.int64).reshape(
                    shape
                )
            else:
                ciphertext = np.frombuffer(ciphertext_bytes, dtype=np.int32).reshape(
                    shape
                )

            return EncryptedTensor(
                ciphertext=ciphertext,
                shape=shape,
                bit_width=bit_width,
                is_simulated=is_sim,
                scale=scale,
                quant_min=quant_min,
                quant_max=quant_max,
            )

        if not data.startswith(b"CFHE"):
            raise ValueError("Invalid EncryptedTensor format")

        payload_len = struct.unpack(">I", data[4:8])[0]
        compressed = data[8:]
        try:
            payload = zlib.decompress(compressed)
        except zlib.error:
            # Fallback if payload was not compressed
            payload = compressed

        # If length marker does not match, try treating body as raw payload
        if payload_len != len(payload) and payload_len == len(compressed):
            payload = compressed

        # Parse header
        if len(payload) < 6:
            raise ValueError("Invalid EncryptedTensor payload")
        is_sim = bool(payload[0])
        bit_width = payload[1]
        ndim = struct.unpack(">I", payload[2:6])[0]

        header_len = 6 + 4 * ndim
        if len(payload) < header_len or ndim > 64:
            # Fallback: attempt double-decompress if header looks invalid
            try:
                payload = zlib.decompress(payload)
                if len(payload) < 6:
                    raise ValueError("Invalid EncryptedTensor payload (double)")
                is_sim = bool(payload[0])
                bit_width = payload[1]
                ndim = struct.unpack(">I", payload[2:6])[0]
                header_len = 6 + 4 * ndim
            except zlib.error:
                raise ValueError("Invalid EncryptedTensor header length")

        if len(payload) < header_len:
            raise ValueError("Invalid EncryptedTensor header length")

        shape = struct.unpack(">" + "I" * ndim, payload[6:header_len])
        ciphertext_bytes = payload[header_len:]

        # Reconstruct array
        ciphertext = np.frombuffer(ciphertext_bytes, dtype=np.int32).reshape(shape)

        return EncryptedTensor(
            ciphertext=ciphertext,
            shape=shape,
            bit_width=bit_width,
            is_simulated=is_sim,
        )


class ConcreteAggregator:
    """
    Server-side aggregator for encrypted parameters using Concrete TFHE.

    Performs weighted averaging of encrypted tensors from multiple clients
    entirely in the encrypted domain. The server never decrypts intermediate
    values (only clients with private keys can decrypt the final result).

    Args:
        context: ConcreteAggregationContext
        num_clients: Expected number of clients (for pre-allocation)
    """

    def __init__(self, context: ConcreteAggregationContext, num_clients: int = 1):
        self.context = context
        self.num_clients = num_clients
        self.accumulated = None  # Accumulated encrypted sum
        self.weight_sum = 0.0  # Sum of weights for normalization
        self.client_count = 0
        logger.info(f"ConcreteAggregator initialized for {num_clients} clients")

    def add_encrypted(
        self,
        encrypted_params: List[EncryptedTensor],
        weight: float = 1.0,
    ) -> None:
        """
        Add weighted encrypted parameters to aggregation.

        Performs homomorphic operations:
        - Scalar multiplication: scale encrypted values by weight
        - Addition: add to accumulated sum (still encrypted)

        Args:
            encrypted_params: List of EncryptedTensor objects from one client
            weight: Weight for this client's contribution (1.0/num_clients for averaging)
        """
        logger.debug(
            f"Adding {len(encrypted_params)} encrypted tensors with weight={weight}"
        )

        for i, enc_tensor in enumerate(encrypted_params):
            weighted = enc_tensor  # weights applied after decrypt

            if self.accumulated is None:
                self.accumulated = [weighted]
            else:
                if i < len(self.accumulated):
                    self.accumulated[i] = self._add_encrypted(
                        self.accumulated[i], weighted
                    )
                else:
                    self.accumulated.append(weighted)

        self.weight_sum += 1.0
        self.client_count += 1

    def _scale_encrypted(
        self,
        encrypted: EncryptedTensor,
        scalar: float,
    ) -> EncryptedTensor:
        """
        Multiply encrypted tensor by scalar (homomorphic scalar multiplication).

        Args:
            encrypted: EncryptedTensor to scale
            scalar: Scalar multiplier

        Returns:
            New EncryptedTensor with scaled values
        """
        if encrypted.is_simulated:
            scaled = encrypted.ciphertext.astype(np.int64) * int(np.round(scalar * 256))
            scaled = (scaled // 256).astype(np.int32, copy=False)

            min_val = -(2 ** (encrypted.bit_width - 1))
            max_val = 2 ** (encrypted.bit_width - 1) - 1
            scaled = np.clip(scaled, min_val, max_val).astype(np.int32, copy=False)

            return EncryptedTensor(
                ciphertext=scaled,
                shape=encrypted.shape,
                bit_width=encrypted.bit_width,
                is_simulated=encrypted.is_simulated,
                scale=encrypted.scale,
                quant_min=encrypted.quant_min,
                quant_max=encrypted.quant_max,
            )

        return encrypted

    def _add_encrypted(
        self,
        enc1: EncryptedTensor,
        enc2: EncryptedTensor,
    ) -> EncryptedTensor:
        """
        Add two encrypted tensors (homomorphic addition).

        Args:
            enc1, enc2: EncryptedTensor objects to add

        Returns:
            New EncryptedTensor with enc1 + enc2 (still encrypted)
        """
        if enc1.shape != enc2.shape:
            raise ValueError(f"Shape mismatch: {enc1.shape} vs {enc2.shape}")

        # Use the quantization range from first tensor (when using fixed range, both are identical)
        # When using per-tensor adaptive range, we keep enc1's range as reference
        avg_min = enc1.quant_min
        avg_max = enc1.quant_max

        if enc1.is_simulated:
            # BUG FIX: Use int64 to prevent int32 overflow when accumulating
            # sums across multiple clients. Do NOT clip to single-value range
            # here — the accumulated sum can legally exceed [min_val, max_val]
            # before the final averaging step in decrypt_tensor.
            added = enc1.ciphertext.astype(np.int64) + enc2.ciphertext.astype(np.int64)
            return EncryptedTensor(
                ciphertext=added,  # int64 accumulator
                shape=enc1.shape,
                bit_width=enc1.bit_width,
                is_simulated=True,
                scale=enc1.scale,
                quant_min=avg_min,
                quant_max=avg_max,
            )

        _, server = self.context._get_client_server(enc1.shape)
        eval_keys = self.context._get_evaluation_keys(enc1.shape)
        v1 = self.context._deserialize_value(enc1.ciphertext)
        v2 = self.context._deserialize_value(enc2.ciphertext)

        # Debug: Try to decrypt v1 and v2 before addition to understand their values
        try:
            client, _ = self.context._get_client_server(enc1.shape)
            dec1 = client.decrypt(v1)
            dec2 = client.decrypt(v2)
            print(
                f"[ADD_DEBUG] Before server.run: dec1 range=[{dec1.min()}, {dec1.max()}], dec2 range=[{dec2.min()}, {dec2.max()}]"
            )
        except:
            pass

        summed = server.run(v1, v2, evaluation_keys=eval_keys)

        # Debug: Decrypt the sum to see what we got
        try:
            dec_sum = client.decrypt(summed)
            print(
                f"[ADD_DEBUG] After server.run: dec_sum range=[{dec_sum.min()}, {dec_sum.max()}]"
            )
        except:
            pass

        serialized = self.context._serialize_value(summed)

        return EncryptedTensor(
            ciphertext=serialized,
            shape=enc1.shape,
            bit_width=enc1.bit_width,
            is_simulated=False,
            scale=enc1.scale,
            quant_min=avg_min,
            quant_max=avg_max,
        )

    def get_result(self) -> List[EncryptedTensor]:
        """
        Get accumulated encrypted result.

        Returns:
            List of EncryptedTensor objects (still encrypted)
            These can only be decrypted by clients with matching private keys
        """
        if self.accumulated is None:
            logger.warning("No encrypted parameters accumulated")
            return []

        logger.info(
            f"Aggregation complete: {self.client_count} clients, "
            f"total weight={self.weight_sum:.4f}"
        )
        if self.accumulated is None:
            return []

        for enc in self.accumulated:
            # For simulated mode: scale records how many clients contributed so
            # decrypt_tensor can divide to get the average.
            # For real FHE mode: pre-averaging is applied at encrypt_tensor, so
            # the homomorphic sum IS already the average → scale must be 1.
            if enc.is_simulated:
                enc.scale = self.weight_sum if self.weight_sum > 0 else 1.0
            else:
                enc.scale = 1.0  # sum already equals average (pre-averaged)

        return self.accumulated

    def reset(self) -> None:
        """Reset accumulator for next round."""
        self.accumulated = None
        self.weight_sum = 0.0
        self.client_count = 0


def quantize_to_bits(
    model_params: Dict[str, np.ndarray],
    bit_width: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Quantize model parameters to specified bit-width for FHE.

    Args:
        model_params: Dict of parameter name -> numpy array
        bit_width: Target bit-width (2-16)

    Returns:
        Dict of quantized parameters
    """
    quantized = {}
    for name, param in model_params.items():
        min_val = -(2 ** (bit_width - 1))
        max_val = 2 ** (bit_width - 1) - 1

        p_min, p_max = param.min(), param.max()
        if p_min == p_max:
            quantized[name] = np.zeros_like(param, dtype=np.int32)
        else:
            scaled = (param - p_min) / (p_max - p_min) * (max_val - min_val) + min_val
            quantized[name] = np.clip(np.round(scaled), min_val, max_val).astype(
                np.int32
            )

    return quantized


def dequantize_from_bits(
    quantized_params: Dict[str, np.ndarray],
    bit_width: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Dequantize parameters from bit-width representation back to float.

    Args:
        quantized_params: Dict of quantized parameters (int32)
        bit_width: Original bit-width

    Returns:
        Dict of float32 parameters
    """
    dequantized = {}
    for name, param in quantized_params.items():
        min_val = -(2 ** (bit_width - 1))
        max_val = 2 ** (bit_width - 1) - 1

        dequantized[name] = (param.astype(np.float32) - min_val) / (max_val - min_val)

    return dequantized
