"""
fl.core.params — model parameter get/set helpers.

These move weights between a torch model and the list of arrays that crosses
the wire, decrypting or encrypting per HE backend. They live here rather than
in fl.core.common so the library path does not import the benchmark-only
utilities in that module.
"""

from __future__ import annotations

import gc
import os
import struct
import zlib
from collections import OrderedDict
from typing import Any, List, Optional

import torch

from fl.core.security import _concrete_quantize, deserialized_layer, simulate_concrete_encrypt
from fl.core.zkp import ZKPLayer, zkp_commit_model

# Track emulated Concrete payload size for benchmarking (bytes)
_LAST_CONCRETE_EMU_BYTES = 0


def get_last_concrete_emulated_bytes() -> int:
    return _LAST_CONCRETE_EMU_BYTES


def _concrete_cpu_burn(byte_count: int, kind: str) -> int:
    """Consume CPU cycles proportional to byte_count (for timing emulation)."""
    if byte_count <= 0:
        return 0

    per_mb_env = f"FL_CONCRETE_CPU_OPS_{kind.upper()}_PER_MB"
    per_mb = int(
        os.environ.get(per_mb_env, os.environ.get("FL_CONCRETE_CPU_OPS_PER_MB", "0"))
        or 0
    )
    if per_mb <= 0:
        return 0

    iterations = int((byte_count / (1024 * 1024)) * per_mb)
    if iterations <= 0:
        return 0

    # Lightweight deterministic loop to consume CPU without extra allocations
    acc = 0
    for i in range(iterations):
        acc = (acc * 1664525 + i + 1013904223) & 0xFFFFFFFF
    return acc

def estimate_concrete_emulated_size(params: List[np.ndarray]) -> int:
    """Estimate size of Concrete emulation payloads (bytes)."""
    total = 0
    for arr in params:
        arr_f = arr.astype(np.float32, copy=False)
        shape = arr_f.shape
        ndim = len(shape)
        header = b"CARR" + struct.pack(">B", 1) + struct.pack(">B", ndim)
        header += struct.pack(">" + "I" * ndim, *shape)
        raw_payload = header + arr_f.tobytes(order="C")

        compressed = zlib.compress(raw_payload)
        envelope = b"CTE2" + struct.pack(">I", len(raw_payload)) + compressed
        total += len(envelope)
    return total

def get_parameters2(
    net, context_client=None, zkp_context=None, he_backend: str = "tenseal"
) -> List[np.ndarray]:
    """
    Get the parameters of the network
    :param net: network to get the parameters (weights and biases)
    :param context_client: context of the crypted weights (if None, return the clear weights)
    :param zkp_context: ZKP context for creating commitments (if None, no ZKP)
    :return: list of parameters (weights and biases) of the network
    """
    if zkp_context:
        # Create ZKP commitments for the model
        zkp_layers = zkp_commit_model(net.state_dict(), zkp_context)
        return zkp_layers

    elif context_client and he_backend == "tenseal":
        # Encrypt with TenSEAL
        # Respect FL_ENCRYPT_LAYERS environment variable (comma-separated list, or 'ALL' for all layers)
        encrypt_env = os.environ.get("FL_ENCRYPT_LAYERS", "model.0.weight,model.0.bias")
        if encrypt_env and encrypt_env.upper() != "ALL":
            encrypt_layers = [s.strip() for s in encrypt_env.split(",") if s.strip()]
        else:
            encrypt_layers = None
        encrypted_tensor = crypte(net.state_dict(), context_client, encrypt_layers)
        # Serialize TenSEAL tensors to bytes for transport over gRPC
        # Convert bytes to numpy uint8 arrays (compatible with Flower's serialization)
        serialized_params = []
        total_orig_size = 0
        total_compressed_size = 0
        encrypted_layers_count = 0

        for layer_idx, layer in enumerate(encrypted_tensor):
            weight = layer.get_weight()
            if hasattr(weight, "serialize"):  # TenSEAL CKKSTensor
                # Serialize to bytes, compress, and convert to numpy uint8 array
                serialized_bytes = weight.serialize()
                total_orig_size += len(serialized_bytes)
                compressed = zlib.compress(serialized_bytes)
                total_compressed_size += len(compressed)
                encrypted_layers_count += 1
                # Debug: log serialized sizes to help diagnose transport issues
                try:
                    print(
                        f"CLIENT_SERIALIZE: layer={layer.get_name()} orig_bytes={len(serialized_bytes):.0f} compressed_bytes={len(compressed):.0f} ratio={100*len(compressed)/len(serialized_bytes):.1f}%"
                    )
                except Exception:
                    print(
                        "CLIENT_SERIALIZE: could not compute sizes for layer",
                        layer.get_name(),
                    )
                serialized_params.append(np.frombuffer(compressed, dtype=np.uint8))
            else:
                # Plain numpy array (not encrypted)
                serialized_params.append(weight)

            # Explicit cleanup every few layers
            if (layer_idx + 1) % 10 == 0:
                gc.collect()

        if total_orig_size > 0:
            print(
                f"CLIENT_SERIALIZE: Total - Original: {total_orig_size / (1024*1024):.2f} MB, Compressed: {total_compressed_size / (1024*1024):.2f} MB, Ratio: {100*total_compressed_size/total_orig_size:.1f}%"
            )
        else:
            if encrypt_layers is not None:
                available_layers = ", ".join(list(net.state_dict().keys()))
                selected_layers = ", ".join(encrypt_layers)
                print(
                    "CLIENT_SERIALIZE: WARNING - no layers were encrypted. "
                    f"Check FL_ENCRYPT_LAYERS. Selected=[{selected_layers}] Available=[{available_layers}]"
                )
            else:
                print(
                    "CLIENT_SERIALIZE: WARNING - no encrypted payload produced in TenSEAL path."
                )

        print(
            f"CLIENT_SERIALIZE: encrypted_layers={encrypted_layers_count}/{len(encrypted_tensor)}"
        )

        # Final cleanup of encrypted tensors
        del encrypted_tensor
        gc.collect()

        return serialized_params
    elif context_client and he_backend == "concrete":
        # Concrete-ML: Simulate FHE preprocessing via quantization
        # Respect FL_ENCRYPT_LAYERS environment variable (comma-separated list, or 'ALL' for all layers)
        encrypt_env = os.environ.get("FL_ENCRYPT_LAYERS", "model.0.weight,model.0.bias")
        if encrypt_env and encrypt_env.upper() != "ALL":
            encrypt_layers = [s.strip() for s in encrypt_env.split(",") if s.strip()]
        else:
            encrypt_layers = None

        # This simulates the computational cost of FHE operations without actual circuit compilation
        quantized_arrays = simulate_concrete_encrypt(
            net.state_dict(), n_bits=8, encrypt_layers=encrypt_layers
        )

        # Concrete can operate in two modes for fair comparison:
        # - 'native' : return plain dequantized numpy arrays (fast, low communication)
        # - 'tenseal': emulate TenSEAL-like transport by serializing+compressing each
        #              quantized array to bytes (increasing CPU + communication)
        concrete_mode = os.environ.get("FL_CONCRETE_MODE", "native").lower()
        if concrete_mode == "tenseal":
            # Serialize into CTE2 envelope (uint8) to emulate TenSEAL transport
            serialized_params = []
            total_bytes = 0
            for arr in quantized_arrays:
                arr_f = arr.astype(np.float32, copy=False)
                shape = arr_f.shape
                ndim = len(shape)
                header = b"CARR" + struct.pack(">B", 1) + struct.pack(">B", ndim)
                if ndim > 0:
                    header += struct.pack(">" + "I" * ndim, *shape)
                raw_payload = header + arr_f.tobytes(order="C")

                compressed = zlib.compress(raw_payload)
                envelope = b"CTE2" + struct.pack(">I", len(raw_payload)) + compressed
                total_bytes += len(envelope)
                serialized_params.append(np.frombuffer(envelope, dtype=np.uint8))

            global _LAST_CONCRETE_EMU_BYTES
            _LAST_CONCRETE_EMU_BYTES = total_bytes
            _concrete_cpu_burn(_LAST_CONCRETE_EMU_BYTES, "enc")
            return serialized_params

        return quantized_arrays

    elif context_client and he_backend == "concrete_tfhe":
        # Concrete TFHE: serialize encrypted parameters for transport
        private_key = getattr(context_client, "private_key", None)
        bit_width = int(os.environ.get("FL_CONCRETE_TFHE_BIT_WIDTH", "14"))
        encrypted_params = get_parameters_concrete_tfhe(
            net,
            bit_width=bit_width,
            enable_fhe=getattr(context_client, "enable_fhe", True),
            context=context_client,
            private_key=private_key,
        )
        return encrypted_params

    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(
    net,
    parameters: List[np.ndarray],
    context_client=None,
    zkp_context=None,
    he_backend: str = "tenseal",
    encrypt_layers: List[str] = None,
):
    """
    Update the parameters of the network with the given parameters (weights and biases)
    :param net: network to set the parameters (weights and biases)
    :param parameters: list of parameters (weights and biases) to set
    :param context_client: context of the crypted weights (if None, set the clear weights)
    :param zkp_context: ZKP context (if None, no ZKP)
    """
    # Handle ZKP layers
    if zkp_context and parameters and isinstance(parameters[0], ZKPLayer):
        params_dict = zip(
            net.state_dict().keys(), [layer.get_weights() for layer in parameters]
        )
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        print("Updated model from ZKP layers")
        return

    params_dict = zip(net.state_dict().keys(), parameters)
    # For TenSEAL, require a real context object (has `secret_key`)
    if hasattr(context_client, "secret_key") and he_backend == "tenseal":
        secret_key = context_client.secret_key()
        dico = {k: deserialized_layer(k, v, context_client) for k, v in params_dict}

        # Decrypt layer by layer
        decrypted_params = {}
        for k, layer_obj in dico.items():
            decrypted_params[k] = layer_obj.decrypt(secret_key)

        state_dict = OrderedDict(
            {k: torch.Tensor(decrypted_params[k]) for k in net.state_dict().keys()}
        )

        del dico, decrypted_params
    elif context_client and he_backend == "concrete":
        # Concrete-ML: Parameters are already "decrypted" (dequantized)
        # We simulate the decryption cost without actual FHE circuit execution
        params_list = list(parameters)

        # If encrypt_layers is not provided, try to read from environment to remain compatible
        if encrypt_layers is None:
            encrypt_env = os.environ.get(
                "FL_ENCRYPT_LAYERS", "model.0.weight,model.0.bias"
            )
            if encrypt_env and encrypt_env.upper() != "ALL":
                encrypt_layers = [
                    s.strip() for s in encrypt_env.split(",") if s.strip()
                ]
            else:
                encrypt_layers = None

        # Allow Concrete to emulate TenSEAL transport if requested
        concrete_mode = os.environ.get("FL_CONCRETE_MODE", "native").lower()
        if concrete_mode == "tenseal":
            _concrete_cpu_burn(estimate_concrete_emulated_size(params_list), "dec")

        dequantized = []
        for i, (k, param) in enumerate(zip(net.state_dict().keys(), params_list)):
            # Only apply quantization/dequantization simulation to selected layers
            if encrypt_layers is None or k in encrypt_layers:
                # If param is a uint8 array produced by our 'tenseal' emulation,
                # parse CTE2 envelope and recover float32 array
                if (
                    isinstance(param, np.ndarray)
                    and param.dtype == np.uint8
                    and concrete_mode == "tenseal"
                ):
                    try:
                        raw = param.tobytes()
                        if not raw.startswith(b"CTE2") or len(raw) < 8:
                            raise ValueError("invalid CTE2 envelope")

                        expected_len = struct.unpack(">I", raw[4:8])[0]
                        compressed = raw[8:]
                        decompressed = zlib.decompress(compressed)
                        if len(decompressed) != expected_len:
                            raise ValueError(
                                f"decompressed length mismatch: got {len(decompressed)} expected {expected_len}"
                            )

                        if not decompressed.startswith(b"CARR"):
                            raise ValueError("invalid CARR payload")
                        dtype_code = decompressed[4]
                        ndim = decompressed[5]
                        offset = 6
                        if ndim > 0:
                            shape = struct.unpack(
                                ">" + "I" * ndim,
                                decompressed[offset : offset + 4 * ndim],
                            )
                            offset += 4 * ndim
                        else:
                            shape = ()
                        data = decompressed[offset:]

                        if dtype_code != 1:
                            raise ValueError(f"unsupported dtype_code {dtype_code}")

                        arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
                        dequantized.append(arr.astype(np.float32, copy=False))
                        continue
                    except Exception:
                        # Fall back to treating as raw bytes if something goes wrong
                        pass

                # Native path: apply quantize/dequantize simulation
                deq = _concrete_quantize(param, n_bits=8)
                dequantized.append(deq)
            else:
                dequantized.append(param.astype(np.float32, copy=False))

        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(net.state_dict().keys(), dequantized)}
        )
    elif context_client and he_backend == "concrete_tfhe":
        # Concrete TFHE: decrypt aggregated encrypted parameters
        private_key = getattr(context_client, "private_key", None)
        expected_shapes = [v.shape for v in net.state_dict().values()]
        decrypted = decrypt_parameters_concrete_tfhe(
            parameters,
            context=context_client,
            private_key=private_key,
            expected_shapes=expected_shapes,
        )
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(net.state_dict().keys(), decrypted)}
        )
    else:
        dico = {k: torch.Tensor(v) for k, v in params_dict}
        state_dict = OrderedDict(dico)

    net.load_state_dict(state_dict, strict=True)
    print("Updated model")

def get_parameters_concrete_tfhe(
    net,
    bit_width: int = 8,
    enable_fhe: bool = True,
    context: Optional[Any] = None,
    private_key: Optional[Any] = None,
) -> List[np.ndarray]:
    """
    Get parameters encrypted with Concrete TFHE (Option C).

    Uses actual FHE for parameter aggregation, not simulation.
    Server aggregates encrypted parameters without decryption.

    Args:
        net: PyTorch model
        bit_width: Quantization bit-width (default: 8)
        enable_fhe: Use real FHE (True) or simulation (False)

    Returns:
        List of numpy arrays (uint8) containing serialized EncryptedTensor objects
    """
    try:
        from .security import encrypt_parameters_concrete_tfhe

        encrypted = encrypt_parameters_concrete_tfhe(
            net.state_dict(),
            bit_width=bit_width,
            context=context,
            private_key=private_key,
            enable_fhe=enable_fhe,
        )

        # Serialize EncryptedTensor objects to bytes (numpy uint8 arrays)
        # so they can be transported over gRPC without pickle
        serialized = []
        for enc_tensor in encrypted:
            try:
                # Each EncryptedTensor has a serialize() method
                serialized_bytes = enc_tensor.serialize()
                # Convert to numpy uint8 array for Flower transport
                np_arr = np.frombuffer(serialized_bytes, dtype=np.uint8)
                serialized.append(np_arr)
            except Exception as e:
                logger.warning(f"Failed to serialize encrypted tensor: {e}")
                # Return empty array as placeholder
                serialized.append(np.array([], dtype=np.uint8))

        return serialized
    except Exception as e:
        logger.error(f"Concrete TFHE encryption failed: {e}")
        raise

def set_parameters_concrete_tfhe(
    net,
    encrypted_parameters: List[Any],
):
    """
    Set parameters from Concrete TFHE decrypted result.

    Args:
        net: PyTorch model
        encrypted_parameters: List of numpy arrays (decrypted on client)
    """
    try:
        params_dict = zip(net.state_dict().keys(), encrypted_parameters)
        state_dict = OrderedDict(
            {k: torch.Tensor(v.astype(np.float32)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        logger.info("Updated model from Concrete TFHE decrypted parameters")
    except Exception as e:
        logger.error(f"Failed to set Concrete TFHE parameters: {e}")
        raise


__all__ = ["get_parameters2", "set_parameters", "set_parameters_concrete_tfhe", "get_parameters_concrete_tfhe", "estimate_concrete_emulated_size"]
