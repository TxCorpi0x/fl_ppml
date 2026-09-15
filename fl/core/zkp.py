"""
Pedersen commitment primitives for Federated Learning — STUB / INCOMPLETE.

This module computes Pedersen commitments (C = g^m · h^r mod p) over a
2048-bit safe-prime group and exposes helpers used by fl/privacy/zkp.py.

WHAT IS MISSING — why the pedersen backend provides no Byzantine protection:

  1. Proof of knowledge (Schnorr / Fiat-Shamir): the ZKPContext.commit()
     method creates commitments but there is no corresponding
     prove_knowledge() / verify_knowledge() pair.  Without this, a client
     cannot convince the server it knows the gradient that produced the
     commitment.

  2. Norm-bound proof: even a complete proof-of-knowledge only shows the
     client *knows* a gradient value — it does not show the gradient's L2
     norm is within the agreed bound.  Proving norm bounds over Pedersen
     commitments requires Bulletproofs or a similar inner-product argument,
     which is considerably more complex and would be slower in pure Python
     than the gnark Groth16 path.

  3. Serialisation and wire transport: commitments and proofs would need to
     be serialised into the Flower metrics dict so the server can verify
     them.  Currently fl/privacy/zkp.py discards all commitment objects and
     sends empty payloads (see _generate_proofs()).

For any deployment requiring real integrity guarantees, use the gnark
backend (FL_ZKP_BACKEND=gnark).  This module is kept as a reference
implementation of the Pedersen primitive and for environments where the Go
binary cannot be built.
"""

import hashlib
import secrets
import json
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import reduce
from .zkp_utils import to_numpy

logger = logging.getLogger(__name__)


class ZKPContext:
    """
    Zero-Knowledge Proof context using Pedersen commitments.

    This provides commitment schemes and range proofs for FL parameter verification
    without revealing the actual parameter values.
    """

    def __init__(self, bit_length: int = 2048):
        """
        Initialize ZKP context with cryptographic parameters.

        Args:
            bit_length: Security parameter for the group (default: 2048 bits)
        """
        self.bit_length = bit_length
        self.p, self.q, self.g, self.h = self._generate_parameters()

    def _generate_parameters(self) -> Tuple[int, int, int, int]:
        """
        Generate Pedersen commitment parameters (p, q, g, h).

        Returns:
            Tuple of (p, q, g, h) where:
            - p is a large prime
            - q is a large prime divisor of (p-1)
            - g, h are generators of the subgroup of order q
        """
        # For production, use proper safe prime generation
        # Here we use a simplified approach for demonstration

        # Use well-known safe primes for demonstration
        # In production, generate fresh primes or use standard groups
        if self.bit_length == 2048:
            # Using RFC 3526 Group 14 (2048-bit MODP Group)
            p = int(
                "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
                "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
                "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
                "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
                "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
                "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
                "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
                "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
                "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
                "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
                "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
                16,
            )
            q = (p - 1) // 2
        else:
            # Simplified parameter generation (not cryptographically secure)
            # For production, use proper prime generation
            p = self._generate_safe_prime_simple()
            q = (p - 1) // 2

        # Generate generators g and h
        # Find g as a generator of the subgroup of order q
        g = self._find_generator(p, q)
        h = self._find_generator(p, q, avoid=g)

        return p, q, g, h

    def _generate_safe_prime_simple(self) -> int:
        """Generate a safe prime (simplified, not cryptographically secure)."""
        # This is a placeholder - in production use proper prime generation
        # For now, return a known safe prime
        return 2**2048 - 2**1984 - 1 + 2**64 * (2**1918 + 1)

    def _find_generator(self, p: int, q: int, avoid: Optional[int] = None) -> int:
        """
        Find a generator of the subgroup of order q in Z_p*.

        Args:
            p: The prime modulus
            q: The order of the subgroup
            avoid: Optional generator to avoid (for finding h != g)
        """
        while True:
            # Random element in Z_p*
            a = secrets.randbelow(p - 2) + 2
            g = pow(a, (p - 1) // q, p)

            # Check if it's a valid generator and not equal to avoid
            if g != 1 and (avoid is None or g != avoid):
                return g

    def commit(self, value: float, randomness: Optional[int] = None) -> Tuple[int, int]:
        """
        Create a Pedersen commitment to a value.

        Commitment: C = g^m * h^r mod p

        Args:
            value: The value to commit to (will be quantized to integer)
            randomness: Optional randomness (if None, generated randomly)

        Returns:
            Tuple of (commitment, randomness)
        """
        # Quantize the float value to an integer
        m = int(value * 1e6) % self.q  # Scale and modulo q

        if randomness is None:
            randomness = secrets.randbelow(self.q)

        # Compute commitment: C = g^m * h^r mod p
        commitment = (pow(self.g, m, self.p) * pow(self.h, randomness, self.p)) % self.p

        return commitment, randomness

    def verify_commitment(self, value: float, commitment: int, randomness: int) -> bool:
        """
        Verify a Pedersen commitment.

        Args:
            value: The claimed value
            commitment: The commitment to verify
            randomness: The randomness used in commitment

        Returns:
            True if commitment is valid, False otherwise
        """
        expected_commitment, _ = self.commit(value, randomness)
        return expected_commitment == commitment

    def serialize(self, include_secret: bool = False) -> bytes:
        """
        Serialize the ZKP context.

        Args:
            include_secret: Not used for ZKP (included for API consistency)

        Returns:
            Serialized context as bytes
        """
        context_dict = {
            "bit_length": self.bit_length,
            "p": str(self.p),
            "q": str(self.q),
            "g": str(self.g),
            "h": str(self.h),
        }
        return json.dumps(context_dict).encode("utf-8")

    @staticmethod
    def deserialize(data: bytes) -> "ZKPContext":
        """
        Deserialize a ZKP context.

        Args:
            data: Serialized context bytes

        Returns:
            ZKPContext instance
        """
        context_dict = json.loads(data)
        ctx = ZKPContext.__new__(ZKPContext)
        ctx.bit_length = int(context_dict["bit_length"])
        ctx.p = int(context_dict["p"])
        ctx.q = int(context_dict["q"])
        ctx.g = int(context_dict["g"])
        ctx.h = int(context_dict["h"])
        return ctx


class ZKPLayer:
    """
    A layer with ZKP commitments and proofs.

    Stores parameter commitments and proofs that the parameters
    were correctly updated during training.
    """

    def __init__(
        self, name: str, weights: np.ndarray, context: Optional[ZKPContext] = None
    ):
        """
        Initialize a ZKP layer.

        Args:
            name: Layer name
            weights: Layer weights
            context: ZKP context (if None, layer is not protected)
        """
        self.name = name
        # Normalize weights to a numpy array early for predictable behaviour
        self.weights = to_numpy(weights)
        self.context = context
        self.commitments = None
        self.randomness = None

        if context is not None:
            self._create_commitments()

    def _create_commitments(self):
        """Create commitments for all weights in the layer."""
        flat_weights = self.weights.flatten()
        self.commitments = []
        self.randomness = []

        for w in flat_weights:
            c, r = self.context.commit(float(w))
            self.commitments.append(c)
            self.randomness.append(r)

    def verify(self) -> bool:
        """
        Verify that the commitments match the weights.

        Returns:
            True if all commitments are valid, False otherwise
        """
        if self.context is None or self.commitments is None:
            return True  # Non-ZKP layer always valid

        flat_weights = self.weights.flatten()
        for w, c, r in zip(flat_weights, self.commitments, self.randomness):
            if not self.context.verify_commitment(float(w), c, r):
                return False
        return True

    def get_weights(self) -> np.ndarray:
        """Get the layer weights."""
        return self.weights

    def get_commitments(self) -> Optional[List[int]]:
        """Get the layer commitments."""
        return self.commitments

    def __add__(self, other):
        """Add two ZKP layers (for aggregation)."""
        if isinstance(other, ZKPLayer):
            new_weights = self.weights + other.weights
        else:
            new_weights = self.weights + other
        return ZKPLayer(self.name, new_weights, self.context)

    def __mul__(self, scalar):
        """Multiply layer by a scalar (for weighted averaging)."""
        new_weights = self.weights * scalar
        return ZKPLayer(self.name, new_weights, self.context)

    def serialize(self) -> Dict:
        """
        Serialize the ZKP layer.

        Returns:
            Dictionary with layer data
        """
        return {
            "name": self.name,
            "weights": self.weights,
            "commitments": self.commitments,
            "has_zkp": self.context is not None,
        }


def create_zkp_context(bit_length: int = 2048) -> ZKPContext:
    """
    Create a ZKP context with specified security level.

    Args:
        bit_length: Security parameter in bits

    Returns:
        ZKPContext instance
    """
    return ZKPContext(bit_length=bit_length)


def zkp_commit_model(
    model_state_dict: Dict,
    context: ZKPContext,
    protected_layers: Optional[List[str]] = None,
) -> List[ZKPLayer]:
    """
    Create ZKP commitments for a model.

    Args:
        model_state_dict: PyTorch model state dictionary
        context: ZKP context
        protected_layers: List of layer names to protect with ZKP
                         (if None, protect only fc3.weight)

    Returns:
        List of ZKPLayer objects
    """
    # If no explicit protected_layers list provided, protect all layers by default
    if protected_layers is None:
        protected_layers = list(model_state_dict.keys())

    zkp_layers: List[ZKPLayer] = []

    for name, weights in model_state_dict.items():
        start_time = time.time()

        np_weights = to_numpy(weights)
        if name in protected_layers:
            # Create ZKP layer with commitments
            layer = ZKPLayer(name, np_weights, context)
        else:
            # Regular layer without ZKP
            layer = ZKPLayer(name, np_weights, None)

        zkp_layers.append(layer)
        logger.debug("ZKP commit %s: %.4fs", name, time.time() - start_time)

    return zkp_layers


def aggregate_zkp_layers(results: List[Tuple[List[ZKPLayer], int]]) -> List[ZKPLayer]:
    """
    Aggregate ZKP layers using weighted average.

    Args:
        results: List of tuples (zkp_layers, num_examples)

    Returns:
        List of aggregated ZKPLayer objects
    """
    # Calculate total number of examples
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create weighted layers (each client's layers scaled by their example count)
    weighted_layers = [
        [layer * num_examples for layer in zkp_layers]
        for zkp_layers, num_examples in results
    ]

    # Aggregate each layer position
    aggregated: List[ZKPLayer] = []
    if not weighted_layers:
        return aggregated

    num_layers = len(weighted_layers[0])

    for layer_idx in range(num_layers):
        # Get all versions of this layer from different clients
        layer_updates = [client_layers[layer_idx] for client_layers in weighted_layers]

        # Stack and sum weights using numpy for clarity and performance
        stacked = np.stack([lu.get_weights() for lu in layer_updates], axis=0)
        summed = np.sum(stacked, axis=0)

        # Average
        aggregated_weights = summed / num_examples_total

        # Create aggregated layer
        layer_name = layer_updates[0].name
        layer_context = layer_updates[0].context
        aggregated_layer = ZKPLayer(layer_name, aggregated_weights, layer_context)
        aggregated.append(aggregated_layer)

    return aggregated


def verify_zkp_layers(zkp_layers: List[ZKPLayer]) -> bool:
    """
    Verify all ZKP layers.

    Args:
        zkp_layers: List of ZKPLayer objects to verify

    Returns:
        True if all layers are valid, False otherwise
    """
    for layer in zkp_layers:
        if not layer.verify():
            logger.warning("ZKP verification failed for layer: %s", layer.name)
            return False
    return True


def write_zkp_params(file_path: str, context: ZKPContext, include_secret: bool = False):
    """
    Write ZKP parameters to file.

    Args:
        file_path: Path to save parameters
        context: ZKP context to save
        include_secret: Whether to include secret parameters
    """
    with open(file_path, "w") as f:
        json.dump({"context": json.loads(context.serialize(include_secret=include_secret))}, f, indent=2)


def read_zkp_params(file_path: str) -> ZKPContext:
    """
    Read ZKP parameters from file.

    Args:
        file_path: Path to parameter file

    Returns:
        ZKPContext instance
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ZKP parameter file not found: {file_path}")

    try:
        with open(file_path) as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{file_path} is not a JSON ZKP parameter file (legacy pickle files are no longer loaded "
            "because unpickling can execute code). Regenerate: python -m fl.keys generate zkp"
        ) from exc
    return ZKPContext.deserialize(json.dumps(data["context"]))
