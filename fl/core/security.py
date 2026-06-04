from typing import List, Tuple, cast, Optional
from typing import Dict, Any
import io
import struct
import numpy as np
import time
import os
import zlib

# For the homomorphic encryption
import pickle
import tenseal as ts
from flwr.common import NDArray, NDArrays, Parameters
from functools import reduce

# ── CVEC binary format helpers ────────────────────────────────────────────
# A CKKSVector (all elements packed into one ciphertext) is prefixed with a
# small CVEC header so that receivers know the original tensor shape and can
# call ts.ckks_vector_from() instead of ts.ckks_tensor_from().
#
#   b"CVEC"              4 bytes  magic
#   uint32 BE ndim       4 bytes  number of shape dimensions
#   uint32 BE × ndim     4 bytes  per dimension
#   <ckks_vector bytes>
_CVEC_MAGIC: bytes = b"CVEC"


def _make_cvec(shape: tuple, ckks_bytes: bytes) -> bytes:
    """Prefix *ckks_bytes* with a CVEC header encoding *shape*."""
    header = (
        _CVEC_MAGIC
        + struct.pack(">I", len(shape))
        + struct.pack(f">{len(shape)}I", *shape)
    )
    return header + ckks_bytes


def _parse_cvec(raw: bytes):
    """Parse a CVEC-prefixed buffer.

    Returns *(shape_tuple, ckks_bytes)* if *raw* starts with _CVEC_MAGIC,
    otherwise returns *(None, raw)*.
    """
    if raw[:4] != _CVEC_MAGIC:
        return None, raw
    ndim = struct.unpack(">I", raw[4:8])[0]
    dims = struct.unpack(f">{ndim}I", raw[8 : 8 + 4 * ndim])
    ckks_start = 8 + 4 * ndim
    return tuple(dims), raw[ckks_start:]


# /////////////////////// Homomorphic Encryption \\\\\\\\\\\\\\\\\\\\\\\\\\


def context():
    """
    This function is used to create the context of the homomorphic encryption:
    it is used to create the keys and the parameters of the encryption scheme (CKKS).

    :return: the context of the homomorphic encryption
    """
    cont = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        # This means that the coefficient modulus will contain 4 primes of 60 bits, 40 bits, 40 bits, and 60 bits.
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )

    cont.generate_galois_keys()  # You can create the Galois keys by calling generate_galois_keys
    cont.global_scale = (
        2**40
    )  # global_scale: the scaling factor, here set to 2**40 (same that pow(2, 40))
    return cont


# Alias used by fl.keys.he_tenseal and fl.privacy.he_tenseal
make_tenseal_context = context


class Layer:
    """
    This class is used to represent the weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the weights of the layer
    """

    def __init__(self, name_layer, weight):
        self.name = name_layer
        self.weight_array = weight

    def get_name(self):
        """
        This function is used to get the name of the layer.
        """
        return self.name

    def get_weight(self):
        """
        This function is used to get the weights of the layer.
        """
        return self.weight_array

    def __add__(self, other):
        """
        This function is used to add the weights of two layers.

        :param other: the other layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        """
        This function is used to substract the weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        """
        This function is used to multiply the weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        return Layer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        """
        This function is used to divide the weights of two layers.
        It works only if the weights of the other layer are a number or a layer object (weights with the same shape).

        :param other: the other layer object

        :return: the division result in the form of a layer object
        """
        weights = other.get_weight() if type(other) == Layer else other
        weights = self.weight_array * (1 / weights)
        return Layer(self.name, weights)

    def __len__(self):
        """
        This function is used to get the number of weights of the layer (the size of the layer).
        """
        somme = 1
        for elem in self.weight_array.shape():
            somme *= elem
        return somme

    def shape(self):
        """
        This function is used to get the shape of the layer.
        """
        return self.weight_array.shape()

    def sum(self, axis=0):
        """
        This function is used to get the sum of the weights of the layer.

        :param axis: the axis of the sum (default: 0 for the sum of the columns)

        :return: the sum of the weights of the layer in the form of a layer object
        """
        return Layer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the mean of the weights of the layer in the form of a layer object
        """
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return Layer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None).
        Not used in this class but this is to respect the same structure as the CryptedLayer class.

        :return: the decrypted weights of the layer in the form of a list of weights
        """
        return self.weight_array.tolist()

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        with the name of the layer as key and the weights as value
        """

        return {self.name: self.weight_array}


class CryptedLayer(Layer):
    """
    This class is used to represent the crypted weights of a layer of a neural network.

    :param name_layer: the name of the layer
    :param weight: the crypted weights of the layer
    :param contexte: the context of the encryption
    """

    def __init__(self, name_layer, weight, contexte=None):
        super(CryptedLayer, self).__init__(name_layer, weight)
        self._original_shape = None
        if (
            type(weight) == ts.tensors.CKKSTensor
            or type(weight) == ts.tensors.CKKSVector
            or type(weight) == bytes
        ):
            # Already encrypted or raw serialised bytes — store as-is.
            self.weight_array = weight

        else:
            # Encrypt with ckks_vector: pack ALL elements into (at most a few)
            # ciphertexts instead of one ciphertext per element.
            # For poly_modulus_degree=8192 the usable slot count is 4096, so
            # even a [64,30]=1920-element layer fits in a single ciphertext.
            arr = weight.cpu().detach().numpy()
            self._original_shape = arr.shape
            self.weight_array = ts.ckks_vector(contexte, arr.flatten().tolist())

    def __add__(self, other):
        """
        This function is used to add the crypted weights of two layers.

        :param other: the other layer object

        :return: the addition result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == CryptedLayer else other
        return CryptedLayer(self.name, self.weight_array + weights)

    def __sub__(self, other):
        """
        This function is used to substract the crypted weights of two layers.

        :param other: the other layer object

        :return: the substraction result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == CryptedLayer else other
        return CryptedLayer(self.name, self.weight_array - weights)

    def __mul__(self, other):
        """
        This function is used to multiply the crypted weights of two layers.

        :param other: the other layer object

        :return: the multiplication result in the form of a crypted layer object
        """
        weights = other.get_weight() if type(other) == CryptedLayer else other
        return CryptedLayer(self.name, self.weight_array * weights)

    def __truediv__(self, other):
        """
        This function is used to divide the crypted weights of two layers.

        :param other: the other layer object

        :return: the division result in the form of a crypted layer object
        """
        try:
            # We try to divide the weights of the layer by the weights of the other layer or by a number
            # It is possible only if the denominator is a number or a tensor of non-crypted weights
            weights = other.get_weight() if type(other) == CryptedLayer else other
            weights = self.weight_array * (1 / weights)

        except:
            print("Error: the division operator isn't supported by SEAL")
            weights = []

        return CryptedLayer(self.name, weights)

    def shape(self):
        """
        This function is used to get the shape of the layer.

        :return: the shape of the layer in the form of a tuple
        """
        # CKKSVector stores all elements flat; return the original tensor shape
        # recorded at encryption time (always set when encrypting from a tensor).
        if self._original_shape is not None:
            return self._original_shape
        # Legacy CKKSTensor path: shape is an attribute on the tensor object.
        return self.weight_array.shape

    def sum(self, axis=0):
        """
        This function is used to get the sum of the weights of the layer.

        :param axis: the axis of the sum (default: 0 for the sum of the columns)

        :return: the sum of the weights of the layer in the form of a crypted layer object
        """
        if isinstance(self.weight_array, ts.tensors.CKKSVector):
            raise NotImplementedError(
                "CryptedLayer.sum() is not supported for CKKSVector-backed layers "
                "(ckks_vector packs all elements into one ciphertext without axis metadata). "
                "This method is only available for legacy CKKSTensor layers."
            )
        return CryptedLayer(f"sum_{self.name}", self.weight_array.sum(axis=axis))

    def mean(self, axis=0):
        """
        This function is used to get the mean of the weights of the layer.

        :param axis: the axis of the mean (default: 0 for the mean of the columns)

        :return: the average of the weights of the layer in the form of a crypted layer object
        """
        if isinstance(self.weight_array, ts.tensors.CKKSVector):
            raise NotImplementedError(
                "CryptedLayer.mean() is not supported for CKKSVector-backed layers. "
                "This method is only available for legacy CKKSTensor layers."
            )
        weights = self.weight_array.sum(axis=axis) * (1 / self.weight_array.shape[axis])
        return CryptedLayer(f"sum_{self.name}", weights)

    def decrypt(self, sk=None):
        """
        This function is used to decrypt the weights of the layer.

        :param sk: the secret key used to decrypt the weights (default: None)

        :return: the decrypted weights of the layer in the form of a (nested)
                 list matching the original tensor shape, or a flat list for
                 1-D layers.
        """
        raw = self.weight_array.decrypt(sk) if sk else self.weight_array.decrypt()
        # raw is always a plain Python list from TenSEAL
        shape = getattr(self, "_original_shape", None)
        if shape is not None and len(shape) > 1 and len(raw) == int(np.prod(shape)):
            return np.array(raw, dtype=np.float32).reshape(shape).tolist()
        # 1-D or no stored shape — return flat list (works for both bias vectors
        # and legacy CKKSTensor paths).
        return list(raw)

    def serialize(self):
        """
        This function is used to serialize the weights of the layer.

        :return: the serialized weights of the layer in the form of a dictionary
        """
        return {self.name: self.weight_array.serialize()}


def crypte(client_w, context_c, encrypt_layers=None):
    """
    Encrypt selected layers of a model using the provided HE context.

    :param client_w: dictionary of the weights of a neural network
    :param context_c: the context of the encryption
    :param encrypt_layers: iterable of layer names to encrypt; if None, encrypt all
    :return: a list of Layer objects (crypted or plain depending on selection)
    """
    # Default: encrypt all layers
    target_layers = set(encrypt_layers) if encrypt_layers is not None else None

    encrypted = []
    for name_layer, weight_array in client_w.items():
        start_time = time.time()

        # Encrypt if layer is targeted (or if target list is not provided)
        should_encrypt = target_layers is None or name_layer in target_layers
        if should_encrypt:
            encrypted.append(CryptedLayer(name_layer, weight_array, context_c))
        else:
            encrypted.append(Layer(name_layer, weight_array))

        print(name_layer, (time.time() - start_time))

    return encrypted


def read_query(file_path):
    """
    This function is used to read a pickle file.

    :param file_path: the path of the file to read
    :return: the query and the context
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            """
            # pickle.load(f)  # load to read file object

            file_str = f.read()
            client_query1 = pickle.loads(file_str)  # loads to read str class
            """
            query_str = pickle.load(file)

        contexte = query_str["contexte"]  # ts.context_from(query["contexte"])
        del query_str["contexte"]
        return query_str, contexte

    else:
        print("The file doesn't exist")


def write_query(file_path, client_query):
    """
    This function is used to write a pickle file.

    :param file_path: the path of the file to write

    :param client_query: the query to write
    """
    with open(file_path, "wb") as file:  # 'ab' to add existing file
        encode_str = pickle.dumps(client_query)
        file.write(encode_str)


def deserialized_layer(name_layer, weight_array, ctx):
    """
    This function is used to deserialized a layer (crypted or not).
    i.e. to transform the weights of a layer in the correct format (not bytes).

    :param name_layer: the name of the layer
    :param weight_array: the weights of the layer
    :param ctx: the context (if the layer is crypted)
    :return: the object Layer or CryptedLayer with the weights of the layer in the correct format
    """
    # Handle serialized TenSEAL tensors (as uint8 numpy arrays from gRPC transport)
    if isinstance(weight_array, np.ndarray) and weight_array.dtype == np.uint8:
        raw = weight_array.tobytes()
        # If the bytes look zlib-compressed, decompress once (small arrays from
        # _pack_raw; large CKKS chunks arrive uncompressed and won't have 0x78
        # as their first byte since CKKS protobuf starts with 0x0a).
        if raw[:2] in (b"\x78\x9c", b"\x78\xda", b"\x78\x01", b"\x78\x5e"):
            try:
                raw = zlib.decompress(raw)
            except Exception as e_decomp:
                print(
                    f"DESERIALIZE_DEBUG: zlib.decompress failed for {name_layer}: {e_decomp}"
                )
                # Fall-through: raw is still compressed; CKKS parse will fail below
        # Plain layer sent as numpy npy format
        if raw[:6] == b"\x93NUMPY":
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            return Layer(name_layer, arr)
        # CVEC-prefixed ckks_vector (new compact format — one ciphertext total)
        if raw[:4] == _CVEC_MAGIC:
            shape, ckks_bytes = _parse_cvec(raw)
            layer_obj = CryptedLayer(
                name_layer, ts.ckks_vector_from(ctx, ckks_bytes), ctx
            )
            layer_obj._original_shape = shape
            return layer_obj
        # Legacy CKKS tensor (one ciphertext per element)
        try:
            return CryptedLayer(name_layer, ts.ckks_tensor_from(ctx, raw), ctx)
        except Exception as e_ckks:
            print(
                f"DESERIALIZE_DEBUG: CKKS parse failed for {name_layer},"
                f" len={len(raw)} first4={raw[:4].hex()} err={e_ckks}"
            )
            raise

    if type(weight_array) == bytes:
        # CVEC-prefixed path for bare bytes
        if weight_array[:4] == _CVEC_MAGIC:
            shape, ckks_bytes = _parse_cvec(weight_array)
            layer_obj = CryptedLayer(
                name_layer, ts.ckks_vector_from(ctx, ckks_bytes), ctx
            )
            layer_obj._original_shape = shape
            return layer_obj
        try:
            return CryptedLayer(name_layer, ts.ckks_tensor_from(ctx, weight_array), ctx)
        except Exception:
            try:
                decompressed = zlib.decompress(weight_array)
                return CryptedLayer(
                    name_layer, ts.ckks_tensor_from(ctx, decompressed), ctx
                )
            except Exception:
                return CryptedLayer(
                    name_layer, ts.ckks_tensor_from(ctx, weight_array), ctx
                )

    elif type(weight_array) in (ts.tensors.CKKSTensor, ts.tensors.CKKSVector):
        return CryptedLayer(name_layer, weight_array, ctx)

    else:
        return Layer(name_layer, weight_array)


def deserialized_model(client_query, ctx):
    """
    This function is used to deserialized a model (crypted or not).
    i.e. to transform the weights of a model in the correct format (not bytes).

    :param client_query: the model
    :param ctx: the context (if the model is crypted)
    :return: the model with the weights in the correct format
    """
    return [
        deserialized_layer(name_layer, weight_array, ctx)
        for name_layer, weight_array in client_query.items()
    ]


# Redefine the aggregate function (defined in Flower)
def aggregate_custom(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average.
    Args:
        results: List of tuples containing the model weights and the number of samples used to compute the weights.
    Returns: A list of model weights averaged according to the number of samples used to compute the weights.
    """
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Before reducing, validate that each client supplied the same number of layers
    lengths = [len(w) for w, _ in results]
    if len(set(lengths)) != 1:
        # Diagnostic output to help debug mismatched parameter lists
        print("AGGREGATE_DEBUG: mismatched parameter counts from clients:", lengths)
        for idx, (w, _) in enumerate(results):
            shapes = [
                getattr(x, "shape", lambda: None)() if hasattr(x, "shape") else None
                for x in w
            ]
            print(f"  client[{idx}] param count={len(w)} shapes={shapes}")
        raise ValueError(f"Mismatched parameter list lengths across clients: {lengths}")

    # Compute average weights of each layer, but detect shape mismatches per-layer
    weights_prime: NDArrays = []
    for layer_idx, layer_updates in enumerate(zip(*weighted_weights)):
        # Check shapes for this layer across clients
        shapes = [getattr(arr, "shape", None) for arr in layer_updates]
        if len(set(shapes)) != 1:
            # Print diagnostics and raise a clear error
            print(f"AGGREGATE_DEBUG: shape mismatch at layer {layer_idx}: {shapes}")
            for client_idx, arr in enumerate(layer_updates):
                try:
                    print(
                        f"  client[{client_idx}] layer{layer_idx} dtype={getattr(arr, 'dtype', None)} shape={getattr(arr, 'shape', None)} size={getattr(arr, 'size', None)}"
                    )
                except Exception:
                    print(
                        f"  client[{client_idx}] layer{layer_idx} could not introspect"
                    )
            raise ValueError(
                f"Operands could not be broadcast together at layer {layer_idx}: shapes={shapes}"
            )

        # Safe to reduce
        summed = reduce(np.add, layer_updates)
        weights_prime.append(summed * (1 / num_examples_total))

    return weights_prime


# /////////////////////// Differential Privacy \\\\\\\\\\\\\\\\\\\\\\\\\\


class DifferentialPrivacyParams:
    """
    Parameters for Differential Privacy mechanism.

    Args:
        epsilon: Privacy budget (smaller = more private, typical: 0.1-10)
        delta: Probability of privacy breach (typical: 1e-5)
        noise_multiplier: Multiplier for noise scale (auto-computed from epsilon/delta)
        max_grad_norm: Maximum gradient norm for clipping (typical: 1.0)
        mechanism: Noise mechanism ('gaussian' or 'laplace')
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_multiplier: float = None,
        max_grad_norm: float = 1.0,
        mechanism: str = "gaussian",
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism

        # Auto-compute noise multiplier if not provided
        if noise_multiplier is None:
            # Using RDP accountant approximation
            self.noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        else:
            self.noise_multiplier = noise_multiplier

    def to_dict(self):
        """Convert parameters to dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "mechanism": self.mechanism,
        }

    @classmethod
    def from_dict(cls, params_dict):
        """Create parameters from dictionary."""
        return cls(**params_dict)


def clip_gradients(parameters: NDArrays, max_norm: float) -> Tuple[NDArrays, float]:
    """
    Clip gradients to maximum L2 norm.

    Args:
        parameters: List of parameter arrays (gradients)
        max_norm: Maximum L2 norm

    Returns:
        Clipped parameters and the actual norm before clipping
    """
    # Compute global L2 norm across all parameters
    total_norm = np.sqrt(sum(np.sum(p**2) for p in parameters))

    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        clipped = [p * clip_coef for p in parameters]
    else:
        clipped = parameters

    return clipped, float(total_norm)


def add_gaussian_noise(
    parameters: NDArrays, noise_scale: float, sensitivity: float = 1.0
) -> NDArrays:
    """
    Add Gaussian noise to parameters for differential privacy.

    Args:
        parameters: List of parameter arrays
        noise_scale: Scale of noise (sigma = noise_scale * sensitivity)
        sensitivity: Sensitivity of the function (typically max_grad_norm)

    Returns:
        Parameters with added noise
    """
    sigma = noise_scale * sensitivity
    noisy_params = []

    for param in parameters:
        noise = np.random.normal(0, sigma, param.shape).astype(param.dtype)
        noisy_params.append(param + noise)

    return noisy_params


def add_laplace_noise(
    parameters: NDArrays, noise_scale: float, sensitivity: float = 1.0
) -> NDArrays:
    """
    Add Laplace noise to parameters for differential privacy.

    Args:
        parameters: List of parameter arrays
        noise_scale: Scale of noise (b = sensitivity / epsilon)
        sensitivity: Sensitivity of the function (typically max_grad_norm)

    Returns:
        Parameters with added noise
    """
    scale = sensitivity / noise_scale if noise_scale > 0 else 0
    noisy_params = []

    for param in parameters:
        noise = np.random.laplace(0, scale, param.shape).astype(param.dtype)
        noisy_params.append(param + noise)

    return noisy_params


def apply_differential_privacy(
    parameters: NDArrays, dp_params: DifferentialPrivacyParams, clip_norm: bool = True
) -> Tuple[NDArrays, dict]:
    """
    Apply differential privacy to model parameters.

    Args:
        parameters: List of parameter arrays
        dp_params: Differential privacy parameters
        clip_norm: Whether to clip gradients first

    Returns:
        DP-protected parameters and statistics dictionary
    """
    stats = {}

    # Step 1: Clip gradients
    if clip_norm:
        clipped_params, original_norm = clip_gradients(
            parameters, dp_params.max_grad_norm
        )
        stats["original_norm"] = original_norm
        stats["clipped"] = original_norm > dp_params.max_grad_norm
    else:
        clipped_params = parameters
        stats["clipped"] = False

    # Step 2: Add noise
    if dp_params.mechanism == "gaussian":
        noisy_params = add_gaussian_noise(
            clipped_params, dp_params.noise_multiplier, dp_params.max_grad_norm
        )
    elif dp_params.mechanism == "laplace":
        noisy_params = add_laplace_noise(
            clipped_params, dp_params.epsilon, dp_params.max_grad_norm
        )
    else:
        raise ValueError(f"Unknown DP mechanism: {dp_params.mechanism}")

    stats["noise_multiplier"] = dp_params.noise_multiplier
    stats["epsilon"] = dp_params.epsilon
    stats["delta"] = dp_params.delta

    return noisy_params, stats


def save_dp_params(params: DifferentialPrivacyParams, filepath: str):
    """Save DP parameters to file."""
    with open(filepath, "wb") as f:
        pickle.dump(params.to_dict(), f)


def load_dp_params(filepath: str) -> DifferentialPrivacyParams:
    """Load DP parameters from file."""
    with open(filepath, "rb") as f:
        params_dict = pickle.load(f)
    return DifferentialPrivacyParams.from_dict(params_dict)


# /////////////////////// Concrete-ML (Enhanced Integration) \\\\\\\\\\\\\

"""
Enhanced Concrete-ML Integration for Privacy-Preserving Federated Learning

This section provides mature Concrete-ML integration based on Zama's use cases:
- Quantization-Aware Training (QAT) with Brevitas
- FHE Model Compilation and Deployment
- Advanced Quantization Strategies
- FHE Client/Server Architecture

References:
- https://docs.zama.ai/concrete-ml/
- Zama use cases: CIFAR, Sentiment Analysis, Credit Scoring, Federated Learning
"""

try:
    from concrete.ml.quantization import QuantizedArray
    from concrete.ml.torch.compile import (
        compile_torch_model,
        compile_brevitas_qat_model,
    )
    from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
    from concrete.fhe import Configuration
    import torch.nn as nn

    CONCRETE_ML_AVAILABLE = True
except ImportError:
    CONCRETE_ML_AVAILABLE = False
    QuantizedArray = None
    print(
        "Warning: concrete-ml not fully available. Install with: pip install concrete-ml"
    )


class ConcreteMLConfig:
    """
    Configuration for Concrete-ML FHE compilation.

    Based on Zama's use case examples for optimal FHE performance.

    Args:
        n_bits: Number of bits for weight/activation quantization (2-8, default: 8)
        rounding_threshold_bits: Rounding parameter for PBS optimization (default: None)
        p_error: Probability of error for FHE operations (default: 0.01)
        global_p_error: Global error probability budget (default: None)
        verbose: Enable verbose compilation output (default: False)
        show_mlir: Show MLIR intermediate representation (default: False)
        enable_unsafe_features: Allow unsafe optimizations (default: False)
        use_insecure_key_cache: Cache keys for faster dev iteration (default: False)
        jit: Enable JIT compilation (default: False)

    Performance Guidelines (from Zama examples):
        n_bits=2-4:  Very fast inference (~100ms), accuracy loss 5-15%
        n_bits=5-6:  Fast inference (~500ms), accuracy loss 2-5%
        n_bits=7-8:  Slower inference (~1-5s), minimal accuracy loss <2%

        p_error=0.001-0.01: Low error, slower, more secure
        p_error=0.01-0.05:  Balanced (recommended)
        p_error=0.05-0.1:   High error, faster, less secure
    """

    def __init__(
        self,
        n_bits: int = 8,
        rounding_threshold_bits: int = None,
        p_error: float = 0.01,
        global_p_error: float = None,
        verbose: bool = False,
        show_mlir: bool = False,
        enable_unsafe_features: bool = False,
        use_insecure_key_cache: bool = False,
        jit: bool = False,
    ):
        self.n_bits = n_bits
        self.rounding_threshold_bits = rounding_threshold_bits
        self.p_error = p_error
        self.global_p_error = global_p_error
        self.verbose = verbose
        self.show_mlir = show_mlir
        self.enable_unsafe_features = enable_unsafe_features
        self.use_insecure_key_cache = use_insecure_key_cache
        self.jit = jit

    def to_configuration(self):
        """Convert to Concrete Configuration object."""
        if not CONCRETE_ML_AVAILABLE:
            raise ImportError("concrete-ml is required for FHE compilation")

        config = Configuration(
            enable_unsafe_features=self.enable_unsafe_features,
            use_insecure_key_cache=self.use_insecure_key_cache,
            insecure_key_cache_location="./keys_cache",
        )

        if self.show_mlir:
            config.dump_artifacts_on_unexpected_failures = True

        return config

    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            "n_bits": self.n_bits,
            "rounding_threshold_bits": self.rounding_threshold_bits,
            "p_error": self.p_error,
            "global_p_error": self.global_p_error,
            "verbose": self.verbose,
            "show_mlir": self.show_mlir,
            "enable_unsafe_features": self.enable_unsafe_features,
            "use_insecure_key_cache": self.use_insecure_key_cache,
            "jit": self.jit,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary."""
        return cls(**config_dict)


def _concrete_quantize(arr: np.ndarray, n_bits: int = 8) -> np.ndarray:
    """
    Use Concrete-ML's QuantizedArray for realistic quantization.

    Follows Zama's quantization approach from CIFAR and Sentiment Analysis examples.

    Args:
        arr: Input numpy array
        n_bits: Number of bits for quantization (default: 8)

    Returns:
        Dequantized float32 array
    """
    if not CONCRETE_ML_AVAILABLE or QuantizedArray is None:
        # Fallback to simple numpy quantization
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)

        arr_min, arr_max = arr.min(), arr.max()
        if arr_min == arr_max:
            return arr.astype(np.float32, copy=False)

        scale = (arr_max - arr_min) / (2**n_bits - 1)
        quantized = np.round((arr - arr_min) / scale)
        dequantized = quantized * scale + arr_min
        return dequantized.astype(np.float32, copy=False)

    if arr.size == 0:
        return arr.astype(np.float32, copy=False)

    # Flatten for quantization, then reshape back
    orig_shape = arr.shape
    arr_flat = arr.flatten()

    # Create QuantizedArray with specified bit-width
    # is_signed=True for symmetric quantization around zero (Zama recommended)
    qarray = QuantizedArray(
        n_bits=n_bits,
        values=arr_flat,
        is_signed=True,
    )

    # Quantize and dequantize to simulate the FHE preprocessing step
    quantized = qarray.qvalues  # Integer representation
    dequantized = qarray.dequant()  # Back to float

    return dequantized.reshape(orig_shape).astype(np.float32, copy=False)


def simulate_concrete_encrypt(
    state_dict: Dict[str, Any], n_bits: int = 8, encrypt_layers: List[str] = None
) -> List[np.ndarray]:
    """
    Simulate Concrete-ML style preprocessing/encryption work by performing
    per-tensor quantization + dequantization using Concrete-ML's QuantizedArray.

    This is used for timing/benchmark purposes in simulation mode and returns
    numpy arrays suitable for transport in the current Flower setup.

    Args:
        state_dict: Model state dictionary (PyTorch tensors)
        n_bits: Number of bits for quantization (default: 8)

    Returns:
        List of quantized/dequantized numpy arrays

    Note: This does not perform FHE circuit compilation or encrypted inference.
    Real Concrete-ML FHE requires compiling an inference circuit which is
    out-of-scope for FL parameter transport.

    Raises:
        ImportError: If concrete-ml is not installed
    """
    simulated = []

    for name, tensor in state_dict.items():
        np_arr = tensor.detach().cpu().numpy()
        # If encrypt_layers is provided, only quantize those layers; otherwise quantize all
        if encrypt_layers is None or name in encrypt_layers:
            deq = _concrete_quantize(np_arr, n_bits=n_bits)
            simulated.append(deq)
        else:
            # Keep original parameter as numpy array
            simulated.append(np_arr.astype(np.float32, copy=False))

    return simulated


def compile_model_for_fhe(
    model: "nn.Module",
    input_sample: np.ndarray,
    config: ConcreteMLConfig,
    is_brevitas_qat: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compile PyTorch model to FHE circuit using Concrete-ML.

    Supports both standard PyTorch models and Brevitas QAT models.
    Based on Zama's CIFAR and Federated Learning examples.

    Args:
        model: PyTorch model to compile
        input_sample: Representative input for tracing (batch_size=1)
        config: Concrete-ML configuration
        is_brevitas_qat: Whether model uses Brevitas QAT

    Returns:
        Tuple of (quantized_module, compilation_stats)

    Example:
        >>> from core.model_builder import build_model
        >>> model = build_model()
        >>> config = ConcreteMLConfig(n_bits=8, p_error=0.01)
        >>> input_sample = np.random.randn(1, 3, 32, 32)
        >>> quantized_module, stats = compile_model_for_fhe(model, input_sample, config)
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError(
            "concrete-ml is not installed. " "Install with: pip install concrete-ml"
        )

    print(f"[Concrete-ML] Compiling model for FHE...")
    print(f"[Concrete-ML] Config: {config.n_bits}-bit, p_error={config.p_error}")
    start_time = time.time()

    try:
        # Choose compilation method based on model type
        if is_brevitas_qat:
            print(f"[Concrete-ML] Using Brevitas QAT compilation...")
            quantized_module = compile_brevitas_qat_model(
                model,
                input_sample,
                n_bits=config.n_bits,
                rounding_threshold_bits=config.rounding_threshold_bits,
                p_error=config.p_error,
                global_p_error=config.global_p_error,
                verbose=config.verbose,
                configuration=config.to_configuration(),
            )
        else:
            print(f"[Concrete-ML] Using standard PyTorch compilation...")
            quantized_module = compile_torch_model(
                model,
                input_sample,
                n_bits=config.n_bits,
                rounding_threshold_bits=config.rounding_threshold_bits,
                p_error=config.p_error,
                verbose=config.verbose,
                configuration=config.to_configuration(),
            )

        compile_time = time.time() - start_time

        # Gather compilation statistics
        stats = {
            "compile_time": compile_time,
            "n_bits": config.n_bits,
            "p_error": config.p_error,
            "is_compiled": True,
            "complexity": (
                quantized_module.fhe_circuit.complexity
                if quantized_module.fhe_circuit
                else 0
            ),
        }

        print(f"[Concrete-ML] [OK] Compilation successful in {compile_time:.2f}s")
        print(f"[Concrete-ML] Circuit complexity: {stats['complexity']}")

        return quantized_module, stats

    except Exception as e:
        compile_time = time.time() - start_time
        print(f"[Concrete-ML] [ERROR] Compilation failed: {e}")
        return None, {
            "compile_time": compile_time,
            "is_compiled": False,
            "error": str(e),
        }


def fhe_inference_simulated(
    quantized_module: Any,
    x: np.ndarray,
) -> np.ndarray:
    """
    Run simulated FHE inference (no actual encryption, for testing).

    Args:
        quantized_module: Compiled Concrete-ML QuantizedModule
        x: Input numpy array

    Returns:
        Simulated FHE output

    Example:
        >>> output = fhe_inference_simulated(quantized_module, test_input)
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError("concrete-ml is required")

    return quantized_module.forward(x, fhe="simulate")


def fhe_inference_encrypted(
    quantized_module: Any,
    x: np.ndarray,
) -> np.ndarray:
    """
    Run real FHE inference on encrypted input.

    Args:
        quantized_module: Compiled Concrete-ML QuantizedModule
        x: Input numpy array (cleartext, will be encrypted internally)

    Returns:
        Decrypted output predictions

    Example:
        >>> output = fhe_inference_encrypted(quantized_module, test_input)
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError("concrete-ml is required")

    return quantized_module.forward(x, fhe="execute")


def save_fhe_deployment(
    quantized_module: Any,
    path: str,
):
    """
    Save FHE model artifacts for client/server deployment.

    Creates client.zip and server.zip for production deployment,
    following Zama's deployment pattern.

    Args:
        quantized_module: Compiled Concrete-ML QuantizedModule
        path: Directory path to save artifacts

    Example:
        >>> save_fhe_deployment(quantized_module, "./fhe_deploy/")
        >>> # Creates: ./fhe_deploy/client.zip, ./fhe_deploy/server.zip
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError("concrete-ml is required")

    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    print(f"[Concrete-ML] Saving deployment artifacts to {path}")

    # Create FHEModelDev for export
    fhe_dev = FHEModelDev(path=str(path), model=quantized_module)
    fhe_dev.save()

    print(f"[Concrete-ML] [OK] Saved: client.zip, server.zip")


def load_fhe_client(deployment_path: str):
    """
    Load FHE client from deployment artifacts.

    Args:
        deployment_path: Path to deployment directory with client.zip

    Returns:
        FHEModelClient instance

    Example:
        >>> client = load_fhe_client("./fhe_deploy/")
        >>> # Use client for encryption/decryption
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError("concrete-ml is required")

    return FHEModelClient(path=deployment_path)


def load_fhe_server(deployment_path: str):
    """
    Load FHE server from deployment artifacts.

    Args:
        deployment_path: Path to deployment directory with server.zip

    Returns:
        FHEModelServer instance

    Example:
        >>> server = load_fhe_server("./fhe_deploy/")
        >>> # Use server for FHE inference
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError("concrete-ml is required")

    return FHEModelServer(path=deployment_path)


def quantize_parameters_concrete(
    state_dict: Dict[str, Any],
    n_bits: int = 8,
) -> List[np.ndarray]:
    """
    Quantize model parameters using Concrete-ML quantization.

    This is useful for FL parameter aggregation with quantization awareness,
    reducing communication overhead while maintaining accuracy.

    Args:
        state_dict: PyTorch state dict
        n_bits: Quantization bit-width (2-8)

    Returns:
        List of quantized numpy arrays

    Example:
        >>> from core.model_builder import build_model
        >>> model = build_model()
        >>> quantized = quantize_parameters_concrete(model.state_dict(), n_bits=6)
    """
    quantized = []
    for name, tensor in state_dict.items():
        np_arr = tensor.detach().cpu().numpy()
        quantized_arr = _concrete_quantize(np_arr, n_bits=n_bits)
        quantized.append(quantized_arr)

    return quantized


def benchmark_fhe_inference(
    quantized_module: Any,
    test_inputs: np.ndarray,
    n_samples: int = 10,
    mode: str = "simulate",
) -> Dict[str, float]:
    """
    Benchmark Concrete-ML FHE inference performance.

    Args:
        quantized_module: Compiled Concrete-ML model
        test_inputs: Test input array (n_samples, ...)
        n_samples: Number of samples to benchmark
        mode: 'simulate' or 'execute' (real FHE)

    Returns:
        Timing statistics dictionary

    Example:
        >>> stats = benchmark_fhe_inference(quantized_module, test_data, n_samples=10, mode="simulate")
        >>> print(f"Mean inference time: {stats['mean_inference_time']:.3f}s")
    """
    if not CONCRETE_ML_AVAILABLE:
        raise ImportError("concrete-ml is required")

    timings = []

    for i in range(min(n_samples, len(test_inputs))):
        x = test_inputs[i : i + 1]

        start = time.time()
        if mode == "simulate":
            _ = quantized_module.forward(x, fhe="simulate")
        else:
            _ = quantized_module.forward(x, fhe="execute")
        elapsed = time.time() - start

        timings.append(elapsed)

    return {
        "mean_inference_time": np.mean(timings),
        "std_inference_time": np.std(timings),
        "min_inference_time": np.min(timings),
        "max_inference_time": np.max(timings),
        "total_time": np.sum(timings),
        "mode": mode,
        "n_samples": len(timings),
    }


def save_concrete_config(config: ConcreteMLConfig, filepath: str):
    """Save Concrete-ML configuration to file."""
    with open(filepath, "wb") as f:
        pickle.dump(config.to_dict(), f)


def load_concrete_config(filepath: str) -> ConcreteMLConfig:
    """Load Concrete-ML configuration from file."""
    with open(filepath, "rb") as f:
        config_dict = pickle.load(f)
    return ConcreteMLConfig.from_dict(config_dict)


# /////////////////////// Concrete TFHE Integration (Option C) \\\\\\\\\\\\\\

"""
Integration of Concrete TFHE compiler for encrypted parameter aggregation.
This provides end-to-end homomorphic encryption for FL without simulation.
"""


def get_concrete_aggregation_context(
    bit_width: int = 14,
    vector_size: int = 1000,
    enable_fhe: bool = True,
    adaptive_quant: bool = True,
    num_clients: int = 2,
) -> Any:
    """
    Get Concrete TFHE context for parameter aggregation.

    Args:
        bit_width: Quantization bit-width (2-16, default: 14).
                   14-bit gives 16384 levels (vs 256 at 8-bit), dramatically
                   reducing quantization error in encrypted FL aggregation.
        vector_size: Expected parameter vector size
        enable_fhe: Whether to use actual FHE (True) or simulation (False)
        adaptive_quant: If True (default), use per-tensor min/max range for
                        quantization instead of a fixed global range.
        num_clients: Number of FL clients. Used for pre-averaging: each client
                     divides quantized values by num_clients before encryption so
                     the homomorphic sum never overflows the TFHE ring.

    Returns:
        ConcreteAggregationContext instance
    """
    try:
        from .concrete_agg import ConcreteAggregationContext

        # None → per-tensor adaptive range; tuple → fixed global range
        fixed_quant_range = None if adaptive_quant else (-5.0, 5.0)

        return ConcreteAggregationContext(
            bit_width=bit_width,
            vector_size=vector_size,
            enable_fhe=enable_fhe,
            fixed_quant_range=fixed_quant_range,
            num_clients=num_clients,
        )
    except ImportError:
        raise ImportError(
            "fl.core.concrete_agg module not available. "
            "Ensure concrete_agg.py is in fl/core/ directory."
        )


def encrypt_parameters_concrete_tfhe(
    state_dict: Dict[str, Any],
    bit_width: int = 8,
    context: Optional[Any] = None,
    private_key: Optional[Any] = None,
    enable_fhe: bool = True,
) -> List[Any]:
    """
    Encrypt model parameters using Concrete TFHE.

    Args:
        state_dict: PyTorch model state dict
        bit_width: Quantization bit-width

    Returns:
        List of EncryptedTensor objects
    """
    try:
        from .concrete_agg import ConcreteAggregationContext

        if context is None:
            context = ConcreteAggregationContext(
                bit_width=bit_width,
                enable_fhe=enable_fhe,
            )
        if private_key is None:
            private_key, _ = context.generate_keys()

        encrypted_list = []
        for name, tensor in state_dict.items():
            np_arr = tensor.detach().cpu().numpy().astype(np.float32)
            enc_tensor = context.encrypt_tensor(np_arr, private_key)
            encrypted_list.append(enc_tensor)

        return encrypted_list
    except ImportError as e:
        raise ImportError(f"Concrete TFHE encryption failed: {e}")


def aggregate_encrypted_concrete(
    encrypted_params_list: List[List[Any]],
    weights: List[float],
) -> List[Any]:
    """
    Aggregate encrypted parameters from multiple clients using Concrete TFHE.

    Args:
        encrypted_params_list: List of [EncryptedTensor] from each client
        weights: Weights for each client (should sum to 1 for averaging)

    Returns:
        List of aggregated EncryptedTensor objects (still encrypted)
    """
    try:
        from .concrete_agg import ConcreteAggregationContext, ConcreteAggregator

        if not encrypted_params_list:
            return []

        # Get context from first encrypted tensor (all should be same context)
        first_params = encrypted_params_list[0]
        if not first_params:
            return []

        bit_width = first_params[0].bit_width
        context = ConcreteAggregationContext(
            bit_width=bit_width,
            enable_fhe=not first_params[0].is_simulated,
        )

        num_clients = len(encrypted_params_list)
        aggregator = ConcreteAggregator(context, num_clients=num_clients)

        # Add each client's encrypted parameters
        for encrypted_params, weight in zip(encrypted_params_list, weights):
            aggregator.add_encrypted(encrypted_params, weight=weight)

        return aggregator.get_result()
    except ImportError as e:
        raise ImportError(f"Concrete aggregation failed: {e}")


# /////////////////////// Concrete-ML Inference Integration (Option B) \\\\\\\\\\\\\\

"""
Integration of Concrete-ML for encrypted inference.
Enables server to run predictions on encrypted client data.
"""


def compile_model_to_concrete_fhe(
    model: Any,
    calibration_data: np.ndarray,
    output_dir: str,
    n_bits: int = 8,
    p_error: float = 0.01,
) -> str:
    """
    Compile PyTorch model to Concrete-ML FHE for encrypted inference.

    Args:
        model: PyTorch model
        calibration_data: Representative data for quantization calibration
        output_dir: Directory to save compiled artifacts
        n_bits: Quantization bit-width
        p_error: Probability of error for FHE

    Returns:
        Path to compiled model directory

    Example:
        >>> from core.security import compile_model_to_concrete_fhe
        >>> from core.model_builder import build_model
        >>> model = build_model()
        >>> compiled_path = compile_model_to_concrete_fhe(
        ...     model, calibration_data, "./fhe_model", n_bits=8
        ... )
    """
    try:
        from .concrete_inference import FHEModelCompiler, ConcreteMLInferenceServer

        compiler = FHEModelCompiler(
            model=model,
            quantization_params={
                "n_bits": n_bits,
                "p_error": p_error,
            },
        )

        compiled_path = compiler.compile_with_calibration(
            calibration_data=calibration_data,
            output_dir=output_dir,
        )

        return compiled_path
    except ImportError as e:
        raise ImportError(
            f"Concrete-ML compilation failed: {e}. " "Install: pip install concrete-ml"
        )


def get_concrete_inference_server(deployed_model_dir: str) -> Any:
    """
    Get server instance for Concrete-ML FHE inference.

    Args:
        deployed_model_dir: Directory with deployed FHE artifacts

    Returns:
        ConcreteMLInferenceServer instance
    """
    try:
        from .concrete_inference import ConcreteMLInferenceServer

        return ConcreteMLInferenceServer(deployed_model_dir)
    except ImportError as e:
        raise ImportError(f"Failed to initialize Concrete inference server: {e}")


def get_concrete_inference_client(deployed_model_dir: str) -> Any:
    """
    Get client instance for Concrete-ML FHE inference.

    Args:
        deployed_model_dir: Directory with deployed FHE artifacts

    Returns:
        ConcreteMLInferenceClient instance
    """
    try:
        from .concrete_inference import ConcreteMLInferenceClient

        return ConcreteMLInferenceClient(deployed_model_dir)
    except ImportError as e:
        raise ImportError(f"Failed to initialize Concrete inference client: {e}")


def decrypt_parameters_concrete_tfhe(
    encrypted_params: List[np.ndarray],
    context: Optional[Any] = None,
    private_key: Optional[Any] = None,
    expected_shapes: Optional[List[Tuple[int, ...]]] = None,
) -> List[np.ndarray]:
    """
    Decrypt parameters encrypted with Concrete TFHE.

    Args:
        encrypted_params: List of numpy uint8 arrays containing serialized EncryptedTensor objects
        private_keys: Private keys for decryption (if None, assumes params are plain)

    Returns:
        List of numpy float32 arrays (decrypted parameters)
    """
    try:
        from .concrete_agg import EncryptedTensor, ConcreteAggregationContext

        if context is None:
            context = ConcreteAggregationContext(bit_width=8, enable_fhe=True)

        decrypted = []
        for idx, param in enumerate(encrypted_params):
            if isinstance(param, np.ndarray) and param.dtype == np.uint8:
                try:
                    enc_tensor = EncryptedTensor.deserialize(param.tobytes())
                    decrypted_arr = context.decrypt_tensor(enc_tensor, private_key)
                    decrypted_arr = decrypted_arr.astype(np.float32, copy=False)
                    # Debug: Check for NaN or extreme values
                    if idx == 0:
                        print(
                            f"[TFHE DEBUG] Decrypted param {idx}: shape={decrypted_arr.shape}, "
                            f"mean={np.mean(decrypted_arr):.6f}, std={np.std(decrypted_arr):.6f}, "
                            f"min={np.min(decrypted_arr):.6f}, max={np.max(decrypted_arr):.6f}, "
                            f"has_nan={np.any(np.isnan(decrypted_arr))}"
                        )
                    if expected_shapes is not None and idx < len(expected_shapes):
                        if tuple(decrypted_arr.shape) != tuple(expected_shapes[idx]):
                            raise ValueError(
                                f"decrypted shape mismatch idx={idx} got={decrypted_arr.shape} expected={expected_shapes[idx]}"
                            )
                    decrypted.append(decrypted_arr)
                except Exception as e:
                    logger.warning(f"Failed to decrypt tensor: {e}")
                    raise
            else:
                if isinstance(param, np.ndarray):
                    decrypted.append(param.astype(np.float32, copy=False))
                else:
                    decrypted.append(np.array(param, dtype=np.float32))

        return decrypted
    except ImportError as e:
        logger.warning(f"Concrete TFHE decryption unavailable: {e}, passing through")
        return [
            (
                np.array(p, dtype=np.float32)
                if isinstance(p, np.ndarray)
                else np.frombuffer(p, dtype=np.float32)
            )
            for p in encrypted_params
        ]
