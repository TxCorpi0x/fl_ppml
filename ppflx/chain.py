"""
Blockchain ledger integration for FL audit trail.

Two backends are provided:

  mock  — in-process append-only log; zero external dependencies.
           Writes a JSON ledger file after every round.  Suitable for
           development, testing, and research demonstrations.

  web3  — connects to a local Ethereum-compatible node (Hardhat / Anvil /
           any EVM chain) via web3.py.  If a deployed FLLedger.sol contract
           address is supplied the module calls commitModel() and
           anchorProofs() on it.  If no contract is deployed, events are
           encoded as calldata on self-transfers (cheaply indexed by the
           node's transaction history).

  none  — disables chain integration entirely; get_chain() returns None.

Usage::

    from ppflx.chain import get_chain

    chain = get_chain(config)           # MockChain, Web3Chain, or None

    model_hash = chain.commit_model(
        round=1,
        model_hash="0xabc...",
        client_hashes=["0x1...", "0x2..."],
        metadata={"mode": "he_tenseal_zkp"},
    )
    chain.anchor_proofs(
        round=1,
        proof_hashes=["0xp1...", "0xp2..."],
        client_ids=["client-0", "client-1"],
    )
    chain.save("./results/ledger.json")

Environment variables (override config values):
    FL_CHAIN_BACKEND        — "mock" | "web3" | "none"
    FL_CHAIN_RPC_URL        — e.g. "http://127.0.0.1:8545"
    FL_CHAIN_PRIVATE_KEY    — hex private key for signing transactions
    FL_CHAIN_CONTRACT_ADDR  — deployed FLLedger.sol address
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ── Utilities ─────────────────────────────────────────────────────────────────


def sha256_hex(data: bytes) -> str:
    """Return a 0x-prefixed SHA-256 hex digest of *data*."""
    return "0x" + hashlib.sha256(data).hexdigest()


def hash_ndarrays(arrays) -> str:
    """Combine-hash a list of numpy arrays into a single SHA-256 digest."""
    h = hashlib.sha256()
    for arr in arrays:
        try:
            h.update(arr.tobytes())
        except Exception:
            h.update(bytes(arr))
    return "0x" + h.hexdigest()


def hash_proof_payload(payload: dict) -> str:
    """
    Deterministically hash a single gnark proof payload dict.

    The dict is canonicalised (sorted keys, no spaces) before hashing so
    that key-ordering differences do not produce different digests.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical.encode())


# ── Abstract base ─────────────────────────────────────────────────────────────


class ChainLedger(ABC):
    """Abstract blockchain ledger for the FL audit trail."""

    @abstractmethod
    def commit_model(
        self,
        round: int,
        model_hash: str,
        client_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a global model commitment for *round*.

        Args:
            round:         FL training round number (1-based).
            model_hash:    0x-prefixed SHA-256 of the aggregated model parameters.
            client_hashes: 0x-prefixed SHA-256 of each accepted client update.
            metadata:      Arbitrary extra data (mode name, num_clients, …).

        Returns:
            Transaction hash or entry-ID string.
        """
        ...

    @abstractmethod
    def anchor_proofs(
        self,
        round: int,
        proof_hashes: List[str],
        client_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Anchor ZKP proof hashes for *round*.

        Args:
            round:        FL training round number (1-based).
            proof_hashes: 0x-prefixed SHA-256 of accepted gnark proof payloads.
            client_ids:   Client identifiers corresponding to each proof hash.

        Returns:
            Transaction hash or entry-ID string.
        """
        ...

    @abstractmethod
    def get_ledger(self) -> List[Dict]:
        """Return the full audit trail as a list of event dicts."""
        ...

    def save(self, path: str) -> None:
        """Persist the ledger to a JSON file, creating parent dirs as needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"ledger": self.get_ledger()}, f, indent=2)
        print(f"[Chain] Ledger saved → {path}")


# ── MockChain — in-process, zero dependencies ─────────────────────────────────


class MockChain(ChainLedger):
    """
    In-process mock blockchain.

    Maintains an append-only list of events with monotonically increasing
    block numbers and wall-clock timestamps.  Provides the same interface as
    Web3Chain so the rest of the codebase is backend-agnostic.

    Each event is a dict::

        {
            "block":     int,      # simulated block number
            "timestamp": float,    # Unix timestamp
            "type":      str,      # "ModelCommit" | "ProofAnchor"
            "round":     int,      # FL training round
            "tx":        str,      # deterministic 0x-prefixed entry ID
            ...                    # type-specific fields
        }
    """

    def __init__(self) -> None:
        self._ledger: List[Dict] = []
        self._block: int = 0
        print("[Chain] MockChain initialised (in-process ledger, no external node)")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _next_block(self) -> int:
        self._block += 1
        return self._block

    @staticmethod
    def _entry_id(block: int) -> str:
        return f"0x{block:064x}"

    # ── Public interface ───────────────────────────────────────────────────────

    def commit_model(
        self,
        round: int,
        model_hash: str,
        client_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        block = self._next_block()
        client_hashes = client_hashes or []
        entry: Dict[str, Any] = {
            "block": block,
            "timestamp": time.time(),
            "type": "ModelCommit",
            "round": round,
            "model_hash": model_hash,
            "client_hashes": client_hashes,
            "num_clients": len(client_hashes),
            "metadata": metadata or {},
            "tx": self._entry_id(block),
        }
        self._ledger.append(entry)
        print(
            f"[Chain] [OK] ModelCommit  round={round}  "
            f"clients={entry['num_clients']}  "
            f"hash={model_hash[:18]}…  block={block}"
        )
        return entry["tx"]

    def anchor_proofs(
        self,
        round: int,
        proof_hashes: List[str],
        client_ids: Optional[List[str]] = None,
    ) -> str:
        block = self._next_block()
        entry: Dict[str, Any] = {
            "block": block,
            "timestamp": time.time(),
            "type": "ProofAnchor",
            "round": round,
            "proof_hashes": proof_hashes,
            "client_ids": client_ids or [],
            "num_proofs": len(proof_hashes),
            "tx": self._entry_id(block),
        }
        self._ledger.append(entry)
        print(
            f"[Chain] [OK] ProofAnchor  round={round}  "
            f"proofs={len(proof_hashes)}  block={block}"
        )
        return entry["tx"]

    def get_ledger(self) -> List[Dict]:
        return list(self._ledger)


# ── Web3Chain — real Ethereum node via web3.py ────────────────────────────────


class Web3Chain(ChainLedger):
    """
    Thin web3.py wrapper for committing FL audit events to an
    Ethereum-compatible node (Hardhat, Anvil, or any EVM chain).

    **With a deployed FLLedger.sol contract** (``FL_CHAIN_CONTRACT_ADDR`` is
    set), the module calls ``commitModel()`` and ``anchorProofs()`` directly.

    **Without a contract** it encodes a keccak fingerprint into calldata and
    sends a zero-value self-transfer, which is indexed in the node's
    transaction history and costs ~21 000 gas.

    Environment variables (take precedence over constructor args):
        FL_CHAIN_RPC_URL       — node HTTP URL  (default: http://127.0.0.1:8545)
        FL_CHAIN_PRIVATE_KEY   — hex private key of the signing account
        FL_CHAIN_CONTRACT_ADDR — deployed FLLedger.sol address
    """

    # Minimal ABI — only the two write functions and their events
    _ABI: List[Dict] = [
        {
            "name": "commitModel",
            "type": "function",
            "inputs": [
                {"name": "round", "type": "uint256"},
                {"name": "modelHash", "type": "bytes32"},
                {"name": "clientHashes", "type": "bytes32[]"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
        {
            "name": "anchorProofs",
            "type": "function",
            "inputs": [
                {"name": "round", "type": "uint256"},
                {"name": "proofHashes", "type": "bytes32[]"},
                {"name": "clientIds", "type": "bytes32[]"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
        {
            "name": "ModelCommitted",
            "type": "event",
            "inputs": [
                {"name": "round", "type": "uint256", "indexed": True},
                {"name": "modelHash", "type": "bytes32", "indexed": False},
                {"name": "numClients", "type": "uint256", "indexed": False},
                {"name": "timestamp", "type": "uint256", "indexed": False},
            ],
        },
        {
            "name": "ProofsAnchored",
            "type": "event",
            "inputs": [
                {"name": "round", "type": "uint256", "indexed": True},
                {"name": "numProofs", "type": "uint256", "indexed": False},
                {"name": "timestamp", "type": "uint256", "indexed": False},
            ],
        },
    ]

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_addr: Optional[str] = None,
    ) -> None:
        try:
            from web3 import Web3  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "web3 is required for Web3Chain.  Install it with:  pip install web3"
            ) from exc

        rpc_url = os.environ.get("FL_CHAIN_RPC_URL", rpc_url or "http://127.0.0.1:8545")
        private_key = os.environ.get("FL_CHAIN_PRIVATE_KEY", private_key or "")
        contract_addr = os.environ.get("FL_CHAIN_CONTRACT_ADDR", contract_addr or "")

        self._Web3 = Web3
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self._w3.is_connected():
            raise ConnectionError(
                f"[Chain] Cannot connect to Ethereum node at {rpc_url}. "
                "Start a local node with:  anvil   or   npx hardhat node"
            )

        # Account setup
        if private_key:
            from eth_account import Account  # type: ignore

            self._signer = Account.from_key(private_key)
            self._account = self._signer.address
            self._use_signer = True
        else:
            # Use the first pre-funded Hardhat/Anvil development account
            self._account = self._w3.eth.accounts[0]
            self._use_signer = False

        # Optional contract binding
        self._contract = None
        if contract_addr:
            self._contract = self._w3.eth.contract(
                address=Web3.to_checksum_address(contract_addr),
                abi=self._ABI,
            )

        self._ledger: List[Dict] = []
        print(
            f"[Chain] Web3Chain connected → {rpc_url}  "
            f"account={self._account}  "
            f"contract={'yes (' + contract_addr[:10] + '…)' if self._contract else 'calldata-only'}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_bytes32(hex_str: str) -> bytes:
        """Convert a 0x-prefixed hex string to exactly 32 bytes (zero-padded)."""
        raw = bytes.fromhex(hex_str.lstrip("0x"))
        if len(raw) >= 32:
            return raw[:32]
        return raw.ljust(32, b"\x00")

    def _send_tx(self, fn) -> str:
        """Build, sign (if needed), send a contract function call; return tx hash."""
        if self._use_signer:
            tx_params = {
                "from": self._account,
                "nonce": self._w3.eth.get_transaction_count(self._account),
                "gas": 500_000,
                "gasPrice": self._w3.eth.gas_price,
            }
            unsigned = fn.build_transaction(tx_params)
            signed = self._w3.eth.account.sign_transaction(unsigned, self._signer.key)
            tx_hash = self._w3.eth.send_raw_transaction(signed.rawTransaction)
        else:
            tx_hash = fn.transact({"from": self._account, "gas": 500_000})

        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        return "0x" + receipt["transactionHash"].hex()

    def _send_calldata(self, fingerprint: bytes) -> str:
        """Send a zero-value self-transfer with *fingerprint* as calldata."""
        tx_hash = self._w3.eth.send_transaction(
            {
                "from": self._account,
                "to": self._account,
                "value": 0,
                "data": fingerprint,
                "gas": 50_000,
            }
        )
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        return "0x" + receipt["transactionHash"].hex()

    # ── Public interface ───────────────────────────────────────────────────────

    def commit_model(
        self,
        round: int,
        model_hash: str,
        client_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        client_hashes = client_hashes or []
        mh_b32 = self._to_bytes32(model_hash)
        ch_b32 = [self._to_bytes32(h) for h in client_hashes]

        if self._contract:
            tx = self._send_tx(
                self._contract.functions.commitModel(round, mh_b32, ch_b32)
            )
        else:
            fingerprint = self._Web3.solidity_keccak(
                ["string", "uint256", "bytes32"],
                ["ModelCommit", round, mh_b32],
            )
            tx = self._send_calldata(fingerprint)

        entry: Dict[str, Any] = {
            "type": "ModelCommit",
            "round": round,
            "model_hash": model_hash,
            "client_hashes": client_hashes,
            "num_clients": len(client_hashes),
            "metadata": metadata or {},
            "tx": tx,
            "timestamp": time.time(),
        }
        self._ledger.append(entry)
        print(
            f"[Chain] [OK] ModelCommit  round={round}  "
            f"clients={len(client_hashes)}  tx={tx[:18]}…"
        )
        return tx

    def anchor_proofs(
        self,
        round: int,
        proof_hashes: List[str],
        client_ids: Optional[List[str]] = None,
    ) -> str:
        client_ids = client_ids or []
        ph_b32 = [self._to_bytes32(h) for h in proof_hashes]
        # Encode client IDs as bytes32 hashes
        cid_b32 = [
            (
                self._to_bytes32(sha256_hex(cid.encode()).lstrip("0x"))
                if cid
                else b"\x00" * 32
            )
            for cid in client_ids
        ]
        # Pad to same length
        while len(cid_b32) < len(ph_b32):
            cid_b32.append(b"\x00" * 32)
        cid_b32 = cid_b32[: len(ph_b32)]

        if self._contract:
            tx = self._send_tx(
                self._contract.functions.anchorProofs(round, ph_b32, cid_b32)
            )
        else:
            fingerprint = self._Web3.solidity_keccak(
                ["string", "uint256", "uint256"],
                ["ProofAnchor", round, len(proof_hashes)],
            )
            tx = self._send_calldata(fingerprint)

        entry: Dict[str, Any] = {
            "type": "ProofAnchor",
            "round": round,
            "proof_hashes": proof_hashes,
            "client_ids": client_ids,
            "num_proofs": len(proof_hashes),
            "tx": tx,
            "timestamp": time.time(),
        }
        self._ledger.append(entry)
        print(
            f"[Chain] [OK] ProofAnchor  round={round}  "
            f"proofs={len(proof_hashes)}  tx={tx[:18]}…"
        )
        return tx

    def get_ledger(self) -> List[Dict]:
        return list(self._ledger)


# ── Factory ───────────────────────────────────────────────────────────────────


def get_chain(config=None) -> Optional[ChainLedger]:
    """
    Return a ChainLedger instance (or None) based on configuration.

    Selection order:
        1. ``FL_CHAIN_BACKEND`` environment variable
        2. ``config.chain_backend`` attribute (if *config* is provided)
        3. Default: ``"mock"``

    Backends:
        mock  — MockChain (in-process, zero external dependencies)
        web3  — Web3Chain (requires ``web3`` package + running EVM node)
        none  — returns None (chain integration disabled)
    """
    backend = "mock"
    if config is not None and hasattr(config, "chain_backend"):
        backend = config.chain_backend
    backend = os.environ.get("FL_CHAIN_BACKEND", backend).lower()

    if backend in ("none", "off", "disabled", ""):
        return None

    if backend == "web3":
        rpc_url = getattr(config, "chain_rpc_url", None) if config else None
        contract = getattr(config, "chain_contract_addr", None) if config else None
        return Web3Chain(rpc_url=rpc_url, contract_addr=contract)

    # Default: mock
    return MockChain()
