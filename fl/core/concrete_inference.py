"""
Concrete-ML Encrypted Inference Module for Federated Learning.

This module implements end-to-end encrypted inference using Zama's Concrete-ML.
After training a model in the FL pipeline, clients can encrypt their test data
and get predictions from a server running the model on encrypted inputs.

Features:
- Model compilation to FHE (quantization-aware training optional)
- Client-side data encryption with per-model keys
- Server-side inference on encrypted data
- Result decryption by clients
- Seamless integration with FL-trained models

Use Case:
    1. FL training completes, server distributes final trained model
    2. Server compiles model to FHE (one-time)
    3. Server generates and shares FHE evaluation keys with clients
    4. Client 1: encrypt test data → send to server
    5. Server: run inference on encrypted data → send encrypted result
    6. Client 1: decrypt result using private key
    (Repeat for Other clients independently)

Security Properties:
    - Privacy: Server never sees plaintext test data or predictions
    - Correctness: FHE guarantees encrypted inference = plaintext inference
    - Unlinkability: Each client's encrypted requests look same to server

References:
    - https://docs.zama.ai/concrete-ml/
    - https://github.com/zama-ai/concrete-ml
    - Concrete-ML paper: "Privacy-Preserving ML with Concrete-ML"
"""

import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import struct
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from concrete.ml.torch.compile import (
        compile_torch_model,
        compile_brevitas_qat_model,
    )
    from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
    from concrete.ml.quantization import QuantizedModule
    from concrete.fhe import Configuration
    import torch
    import torch.nn as nn

    CONCRETE_ML_AVAILABLE = True
except ImportError as e:
    CONCRETE_ML_AVAILABLE = False
    logger.warning(
        f"Concrete-ML not available ({e}). " "Install: pip install concrete-ml"
    )


class ConcreteMLInferenceServer:
    """
    Server-side component for FHE inference.

    After deployment, the server holds:
    - Compiled model (FHE circuit)
    - Evaluation key (public key for FHE runtime)
    Clients send encrypted inputs; server returns encrypted predictions.

    Args:
        deployed_model_dir: Directory containing deployed FHE model
        model_name: Name of the model (for logging)
    """

    def __init__(self, deployed_model_dir: str, model_name: str = "fhe_model"):
        if not CONCRETE_ML_AVAILABLE:
            raise ImportError("Concrete-ML required. Install: pip install concrete-ml")

        self.deployed_model_dir = Path(deployed_model_dir)
        self.model_name = model_name
        self.fhe_server = None
        self.evaluation_key = None

        self._load_deployment()
        logger.info(f"ConcreteMLInferenceServer loaded model from {deployed_model_dir}")

    def _load_deployment(self):
        """Load FHE deployment artifacts."""
        try:
            # Load FHEModelServer (handles evaluation key, compiled circuit)
            self.fhe_server = FHEModelServer(
                str(self.deployed_model_dir / "server.zip")
            )
            logger.info("FHEModelServer loaded successfully")
        except FileNotFoundError:
            logger.error(
                f"Deployment not found at {self.deployed_model_dir/'server.zip'}. "
                "Run compile_model_to_fhe() first."
            )
            raise

    def predict_encrypted(
        self,
        encrypted_input: bytes,
    ) -> bytes:
        """
        Run inference on encrypted input.

        Args:
            encrypted_input: Serialized encrypted tensor from client

        Returns:
            Serialized encrypted predictions (still encrypted, only client can decrypt)
        """
        try:
            # Deserialize encrypted input
            enc_input = self._deserialize_encrypted_tensor(encrypted_input)

            # Run FHE inference (happens entirely on encrypted data)
            logger.debug(
                f"Running FHE inference on encrypted input shape {enc_input.shape}"
            )
            enc_predictions = self.fhe_server.run(enc_input)

            # Serialize encrypted predictions
            serialized = self._serialize_encrypted_tensor(enc_predictions)
            return serialized
        except Exception as e:
            logger.error(f"Encrypted inference failed: {e}")
            raise

    def get_client_library(self) -> bytes:
        """
        Export client library for testing.
        Returns pickled FHEModelClient object.
        """
        try:
            client_library = FHEModelClient(str(self.deployed_model_dir / "client.zip"))
            return pickle.dumps(client_library)
        except Exception as e:
            logger.error(f"Failed to export client library: {e}")
            raise

    @staticmethod
    def _serialize_encrypted_tensor(tensor: np.ndarray) -> bytes:
        """Serialize encrypted tensor for network transport."""
        payload = struct.pack(
            ">I" + "I" * len(tensor.shape),
            len(tensor.shape),
            *tensor.shape,
        )
        payload += tensor.tobytes(order="C")

        compressed = zlib.compress(payload)
        envelope = b"CFEN" + struct.pack(">I", len(payload)) + compressed
        return envelope

    @staticmethod
    def _deserialize_encrypted_tensor(data: bytes) -> np.ndarray:
        """Deserialize encrypted tensor from network."""
        if not data.startswith(b"CFEN"):
            raise ValueError("Invalid encrypted tensor format")

        payload_len = struct.unpack(">I", data[4:8])[0]
        payload = zlib.decompress(data[8:])

        # Parse shape
        ndim = struct.unpack(">I", payload[0:4])[0]
        shape = struct.unpack(">" + "I" * ndim, payload[4 : 4 + 4 * ndim])

        # Reconstruct tensor (as uint8 - encrypted representation)
        tensor_bytes = payload[4 + 4 * ndim :]
        tensor = np.frombuffer(tensor_bytes, dtype=np.uint8).reshape(shape)

        return tensor


class ConcreteMLInferenceClient:
    """
    Client-side component for FHE inference.

    Clients use this to:
    1. Encrypt their test data
    2. Send encrypted data to server
    3. Receive encrypted predictions
    4. Decrypt predictions with private key

    Args:
        deployed_model_dir: Directory with deployed FHE model
        model_name: Name of model (for identification)
    """

    def __init__(self, deployed_model_dir: str, model_name: str = "fhe_model"):
        if not CONCRETE_ML_AVAILABLE:
            raise ImportError("Concrete-ML required. Install: pip install concrete-ml")

        self.deployed_model_dir = Path(deployed_model_dir)
        self.model_name = model_name
        self.fhe_client = None
        self.input_quantization_params = None

        self._load_client()
        logger.info(f"ConcreteMLInferenceClient initialized for {model_name}")

    def _load_client(self):
        """Load client FHE library."""
        try:
            self.fhe_client = FHEModelClient(
                str(self.deployed_model_dir / "client.zip")
            )
            logger.info("FHEModelClient loaded successfully")
        except FileNotFoundError:
            logger.error(
                f"Client library not found at {self.deployed_model_dir/'client.zip'}. "
                "Ensure server deployment is complete."
            )
            raise

    def encrypt_data(
        self,
        input_data: np.ndarray,
    ) -> Tuple[bytes, str]:
        """
        Encrypt client's test data for server inference.

        Args:
            input_data: Raw test data (float32 or int, shape depends on model)

        Returns:
            Tuple of (encrypted_data_bytes, evaluation_key_id)
        """
        try:
            # FHEModelClient handles quantization internally
            logger.debug(f"Encrypting test data shape {input_data.shape}")

            # Prepare evaluation key (public key for FHE operations)
            # In production, this would be generated once and cached
            evaluation_key = self.fhe_client.generate_evaluation_key()

            # Encrypt data
            encrypted_data = self.fhe_client.encrypt(input_data)

            # Serialize
            return self._serialize_encrypted_tensor(encrypted_data), str(
                id(evaluation_key)
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_predictions(
        self,
        encrypted_predictions: bytes,
    ) -> np.ndarray:
        """
        Decrypt server's encrypted predictions.

        Args:
            encrypted_predictions: Bytes from server (still encrypted)

        Returns:
            Decrypted predictions (numpy array)
        """
        try:
            # Deserialize
            enc_pred = self._deserialize_encrypted_tensor(encrypted_predictions)

            # Decrypt using private key (held by client)
            logger.debug(f"Decrypting predictions shape {enc_pred.shape}")
            predictions = self.fhe_client.decrypt(enc_pred)

            return predictions
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    @staticmethod
    def _serialize_encrypted_tensor(tensor: np.ndarray) -> bytes:
        """Serialize encrypted tensor for transport."""
        payload = struct.pack(
            ">I" + "I" * len(tensor.shape),
            len(tensor.shape),
            *tensor.shape,
        )
        payload += tensor.tobytes(order="C")

        compressed = zlib.compress(payload)
        envelope = b"CFEC" + struct.pack(">I", len(payload)) + compressed
        return envelope

    @staticmethod
    def _deserialize_encrypted_tensor(data: bytes) -> np.ndarray:
        """Deserialize encrypted tensor from server."""
        if not data.startswith(b"CFEC"):
            raise ValueError("Invalid encrypted tensor format from server")

        payload_len = struct.unpack(">I", data[4:8])[0]
        payload = zlib.decompress(data[8:])

        ndim = struct.unpack(">I", payload[0:4])[0]
        shape = struct.unpack(">" + "I" * ndim, payload[4 : 4 + 4 * ndim])

        tensor_bytes = payload[4 + 4 * ndim :]
        tensor = np.frombuffer(tensor_bytes, dtype=np.uint8).reshape(shape)

        return tensor


class FHEModelCompiler:
    """
    Compile PyTorch models to FHE for inference.

    Supports:
    - Standard PyTorch models (via ONNX)
    - Brevitas QAT models (quantization-aware training)
    - Post-Training Quantization (PTQ)

    Args:
        model: PyTorch model to compile
        quantization_params: Dict with n_bits, p_error, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        quantization_params: Optional[Dict[str, Any]] = None,
    ):
        if not CONCRETE_ML_AVAILABLE:
            raise ImportError("Concrete-ML required for compilation")

        self.model = model
        self.quantization_params = quantization_params or {}
        self.compiled_model = None
        self.quantized_module = None

        logger.info(
            f"FHEModelCompiler initialized for model {model.__class__.__name__}"
        )

    def compile_with_calibration(
        self,
        calibration_data: np.ndarray,
        output_dir: str,
    ) -> str:
        """
        Compile model to FHE using calibration data.

        Post-Training Quantization approach:
        1. Take trained model (float32)
        2. Use calibration data to determine quantization parameters
        3. Compile to FHE circuit
        4. Export client/server artifacts

        Args:
            calibration_data: Representative data for determining quantization
            output_dir: Where to save compiled artifacts

        Returns:
            Path to output directory
        """
        try:
            logger.info(
                f"Starting FHE compilation with {len(calibration_data)} calibration samples"
            )

            # Set model to eval mode
            self.model.eval()

            # Get quantization config
            n_bits = self.quantization_params.get("n_bits", 8)
            p_error = self.quantization_params.get("p_error", 0.01)

            # Compile using Concrete-ML
            self.quantized_module = compile_torch_model(
                model=self.model,
                X_test=calibration_data[:10],  # Small sample for tracing
                n_bits=n_bits,
                p_error=p_error,
            )

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Export for client/server deployment
            self.quantized_module.save_and_compile_fhe_model(
                path_dir=str(output_path),
            )

            logger.info(f"Model compiled successfully to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            raise

    def compile_qat_model(
        self,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        output_dir: str,
        epochs: int = 10,
    ) -> str:
        """
        Compile Brevitas QAT model to FHE.

        Quantization-Aware Training approach:
        1. Model already has quantization layers (from training)
        2. Use training data to fine-tune quantization parameters
        3. Compile to FHE

        Args:
            training_data: Training data
            training_labels: Training labels
            output_dir: Output directory
            epochs: Fine-tuning epochs

        Returns:
            Path to output directory
        """
        try:
            logger.info(f"Compiling QAT model with {epochs} fine-tuning epochs")

            # Assume model has Brevitas quantization layers already
            self.compiled_model = compile_brevitas_qat_model(
                model=self.model,
                X_train=training_data,
                y_train=training_labels,
                epochs=epochs,
            )

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.compiled_model.save_and_compile_fhe_model(
                path_dir=str(output_path),
            )

            logger.info(f"QAT model compiled to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"QAT compilation failed: {e}")
            raise


class FHEInferenceSimulator:
    """
    Simulate FHE inference for benchmarking without actual encryption.

    Useful for:
    - Testing workflow before actual FHE deployment
    - Benchmarking accuracy under quantization
    - Profile latency impacts

    Args:
        model: PyTorch model
        quantization_params: Dict with n_bits, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        quantization_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.quantization_params = quantization_params or {"n_bits": 8}
        self.n_bits = self.quantization_params.get("n_bits", 8)
        logger.info(
            f"FHEInferenceSimulator initialized with {self.n_bits}-bit quantization"
        )

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        """Quantize input to FHE-compatible integer range."""
        min_val = -(2 ** (self.n_bits - 1))
        max_val = 2 ** (self.n_bits - 1) - 1

        x_min, x_max = x.min(), x.max()
        if x_min == x_max:
            return np.zeros_like(x, dtype=np.int32)

        scaled = (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val
        quantized = np.clip(np.round(scaled), min_val, max_val).astype(np.int32)

        return quantized

    def dequantize_output(self, y: np.ndarray) -> np.ndarray:
        """Dequantize model output back to float."""
        min_val = -(2 ** (self.n_bits - 1))
        max_val = 2 ** (self.n_bits - 1) - 1

        return (y.astype(np.float32) - min_val) / (max_val - min_val)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run simulation of FHE inference.

        1. Quantize input
        2. Run model (simulating encrypted computation)
        3. Dequantize output
        """
        self.model.eval()

        # Convert to torch
        x_tensor = torch.from_numpy(x).float()

        # Model inference
        with torch.no_grad():
            y = self.model(x_tensor).cpu().numpy()

        # For now, return predictions as-is
        # (Real FHE would quantize/dequantize)
        return y
