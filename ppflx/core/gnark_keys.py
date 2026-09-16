"""
Pinned Groth16 key manifest (docs/ZKP.md, section 7).

`gnark_service setup` writes `manifest.json` and the verifying keys into a
committed directory, and the proving keys into a local cache. Python never
reads key material: it reads the manifest so the server can pin which
verifying key each proof must be checked under, and so both sides agree on
each circuit's fixed size.

Environment:
    FL_ZKP_KEYS_DIR  manifest and verifying keys (default: keys packaged with
                     this library, else zkp_gnark_service/keys beside it)
    FL_ZKP_PK_DIR    proving-key cache for the prover role
                     (default: ~/.cache/fl_ppml/gnark_pk)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

NORM_CIRCUIT = "norm"
ELGAMAL_CIRCUIT = "elgamal"

KEYS_DIR_ENV = "FL_ZKP_KEYS_DIR"
PK_DIR_ENV = "FL_ZKP_PK_DIR"
_REPO = Path(__file__).resolve().parents[2]
# Where the pinned manifest and verifying keys are looked up, in order: the
# environment, keys shipped with this package, then the proof service checked
# out beside it. Packaged keys are the trust anchor once the service lives in
# its own repository; the service itself is pointed at its keys with --keys-dir.
PACKAGED_KEYS_DIR = Path(__file__).resolve().parent / "gnark_keys_data"
SERVICE_KEYS_DIR = _REPO / "zkp_gnark_service" / "keys"
DEFAULT_PK_DIR = Path.home() / ".cache" / "fl_ppml" / "gnark_pk"

_cache: Dict[tuple, dict] = {}


def keys_dir() -> Path:
    """The pinned keys directory: the environment, else packaged keys, else the service's."""
    from_env = os.environ.get(KEYS_DIR_ENV)
    if from_env:
        return Path(from_env)
    for candidate in (PACKAGED_KEYS_DIR, SERVICE_KEYS_DIR):
        if (candidate / "manifest.json").exists():
            return candidate
    return SERVICE_KEYS_DIR


def pk_dir() -> Path:
    return Path(os.environ.get(PK_DIR_ENV, str(DEFAULT_PK_DIR)))


def load_manifest() -> dict:
    """Parsed manifest plus its SHA-256. Raises if keys were never set up."""
    path = keys_dir() / "manifest.json"
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No pinned ZKP key manifest at {path}. Point {KEYS_DIR_ENV} at the keys of the "
            f"proof service, or run: gnark_service setup --keys-dir {keys_dir()} --pk-dir {pk_dir()}"
        ) from exc
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        by_circuit = {}
        for entry in manifest.get("circuits", []):
            if entry["circuit"] in by_circuit:
                raise ValueError(f"{path}: circuit {entry['circuit']!r} listed more than once")
            by_circuit[entry["circuit"]] = entry
        for circuit in (NORM_CIRCUIT, ELGAMAL_CIRCUIT):
            if circuit not in by_circuit:
                raise ValueError(f"{path}: no entry for circuit {circuit!r}")
        _cache.clear()
        _cache[key] = {"sha256": hashlib.sha256(raw).hexdigest(), "circuits": by_circuit, "raw": manifest}
    return _cache[key]


def manifest_sha256() -> str:
    return load_manifest()["sha256"]


def circuit_size(circuit: str) -> int:
    """Fixed number of values one proof of this circuit covers."""
    return int(load_manifest()["circuits"][circuit]["n"])


def pinned_vk_sha256(circuit: str) -> str:
    """SHA-256 of the verifying key every proof of this circuit must be checked under."""
    return load_manifest()["circuits"][circuit]["vk_sha256"]


def missing_proving_keys() -> list:
    """Proving-key files named in the manifest that are absent from the local cache."""
    return [
        entry["pk_file"]
        for entry in load_manifest()["circuits"].values()
        if not (pk_dir() / entry["pk_file"]).exists()
    ]
