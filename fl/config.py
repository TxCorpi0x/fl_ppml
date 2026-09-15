"""
Single source of truth for all FL experiment configuration.

FLConfig replaces the scattered argparse namespaces and 20-parameter
constructors found in the legacy client.py / server.py / main_*.py files.
All code in fl/ takes an FLConfig instance; no module-level globals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FLConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset: str = "creditcard"
    data_path: str = "./data/"
    num_classes: int = 2

    # ── Training ──────────────────────────────────────────────────────────────
    num_clients: int = 3
    num_rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"
    num_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    val_split: int = 10  # % of training data reserved for validation per client

    # ── Federation sampling ───────────────────────────────────────────────────
    frac_fit: float = 1.0
    frac_eval: float = 1.0
    min_fit_clients: int = 2
    min_eval_clients: Optional[int] = None  # defaults to num_clients // 2 at runtime
    min_avail_clients: int = 2

    # ── Data heterogeneity ────────────────────────────────────────────────────
    # Dirichlet concentration parameter for non-IID data partitioning.
    # None (default) → uniform IID split.
    # 0.1  → extreme non-IID (each client mostly has one class).
    # 0.5  → moderate non-IID.
    # 1.0  → mild non-IID.
    # 10.0 → near-IID.
    dirichlet_alpha: Optional[float] = None

    # ── Privacy mode ──────────────────────────────────────────────────────────
    # One of: baseline | he_tenseal | he_concrete_tfhe | zkp | dp
    #           he_tenseal_zkp | he_concrete_tfhe_zkp
    privacy_mode: str = "baseline"

    # ── TenSEAL HE params ─────────────────────────────────────────────────────
    he_tenseal_secret_path: str = "keys/he_tenseal/secret_context.bin"
    he_tenseal_public_path: str = "keys/he_tenseal/public_context.bin"
    he_poly_modulus: int = 8192
    he_coeff_mod_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    he_scale: int = 40  # global_scale = 2**he_scale

    # ── Verifiable ElGamal HE params (he_elgamal_zkp) ─────────────────────────
    he_elgamal_secret_path: str = "keys/he_elgamal/secret_key.json"
    he_elgamal_public_path: str = "keys/he_elgamal/public_key.json"

    # ── Concrete TFHE params ──────────────────────────────────────────────────
    he_tfhe_bit_width: int = 14
    he_tfhe_adaptive_quant: bool = False

    # ── ZKP params ────────────────────────────────────────────────────────────
    zkp_backend: str = "gnark"  # gnark | pedersen
    zkp_params_path: str = "keys/zkp/zkp_params.json"
    zkp_gnark_host: str = "localhost:9000"

    # ── DP params ─────────────────────────────────────────────────────────────
    dp_params_path: str = "keys/dp/dp_params.json"
    dp_epsilon: float = 10.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 0.484481

    # ── Output ────────────────────────────────────────────────────────────────
    results_dir: str = "./results/"
    model_save: str = "./model.pth"
    benchmark: bool = True

    # ── HE layer selection ────────────────────────────────────────────────────
    # Comma-separated layer names to encrypt ("ALL" means every layer).
    encrypt_layers: str = "ALL"

    # ── Simulation flag ───────────────────────────────────────────────────────
    # False → encrypt/decrypt real tensors end-to-end (default)
    # True  → in-process simulation: HE modes measure crypto cost but transport
    #         plain numpy, so their results say nothing about encrypted transport.
    #         Only simulation entry points (simulation.py, fl.runner) set it.
    sim_mode: bool = False

    # ── Blockchain ledger ─────────────────────────────────────────────────────
    # backend: "mock" (in-process, zero deps) | "web3" (requires web3.py +
    #          running Ethereum node) | "none" (disabled)
    chain_backend: str = "mock"
    chain_rpc_url: str = "http://127.0.0.1:8545"
    chain_contract_addr: str = ""  # deployed FLLedger.sol address (web3 only)
    chain_ledger_path: str = "./results/ledger.json"  # path for the JSON ledger file

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "FLConfig":
        """Construct from a plain dict (e.g. converted argparse Namespace)."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_namespace(cls, ns) -> "FLConfig":
        """Construct directly from an argparse.Namespace."""
        return cls.from_dict(vars(ns))

    # Convenience helpers ─────────────────────────────────────────────────────

    @property
    def is_he(self) -> bool:
        return self.privacy_mode.startswith("he_")

    @property
    def is_zkp(self) -> bool:
        return self.privacy_mode == "zkp"

    @property
    def is_dp(self) -> bool:
        return self.privacy_mode == "dp"

    @property
    def effective_min_eval_clients(self) -> int:
        return (
            self.min_eval_clients if self.min_eval_clients else (self.num_clients // 2)
        )

    @property
    def encrypt_layer_list(self) -> Optional[List[str]]:
        """Return None (all layers) or list of specific layer names."""
        v = self.encrypt_layers.strip().upper()
        if not v or v == "ALL":
            return None
        return [s.strip() for s in self.encrypt_layers.split(",") if s.strip()]
