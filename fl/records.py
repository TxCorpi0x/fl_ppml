"""
Flower record helpers: dicts to ConfigRecord / MetricRecord, and the client
state that has to survive between ClientApp messages.

A SuperNode runs every message in a fresh ClientApp process, so state a mode
keeps across Flower rounds (the commit–challenge commitment) is stored in the
node's ``Context.state``. It is encoded as JSON plus arrays and byte strings,
never pickled, and only allowlisted dataclasses are reconstructed.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from typing import Any, Dict, Tuple

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord

_CONFIG_SCALARS = (bool, int, float, str, bytes)

# Dataclasses that client state may contain: class name → (module, class).
STATE_DATACLASSES = {"GlobalModel": ("fl.core.elgamal_gnark", "GlobalModel")}


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def config_record(values: Dict[str, Any]) -> ConfigRecord:
    """ConfigRecord from a dict of scalars. None entries are dropped; other types raise."""
    out = {}
    for key, value in values.items():
        value = _plain(value)
        if value is None:
            continue
        if not isinstance(value, _CONFIG_SCALARS):
            raise TypeError(f"config value {key!r} of type {type(value).__name__} is not a Flower config scalar")
        out[key] = value
    return ConfigRecord(out)


def metric_record(values: Dict[str, Any]) -> MetricRecord:
    """MetricRecord from the numeric entries of a dict; booleans become 0/1."""
    out = {}
    for key, value in values.items():
        value = _plain(value)
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (int, float)):
            out[key] = value
    return MetricRecord(out)


# ── Client state ─────────────────────────────────────────────────────────────


def pack_state(state: Dict[str, Any]) -> Tuple[ConfigRecord, ArrayRecord]:
    """Encode a dict of scalars, arrays, bytes, dicts and allowlisted dataclasses."""
    arrays: Dict[str, np.ndarray] = {}
    blobs: Dict[str, bytes] = {}
    tree = _pack(state, arrays, blobs)
    config = ConfigRecord({"tree": json.dumps(tree), **{f"bytes.{k}": v for k, v in blobs.items()}})
    array_record = ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in arrays.items()}) if arrays else ArrayRecord()
    return config, array_record


def unpack_state(config: ConfigRecord, arrays: ArrayRecord) -> Dict[str, Any]:
    """Inverse of pack_state."""
    blobs = {k[len("bytes.") :]: v for k, v in config.items() if k.startswith("bytes.")}
    loaded = {k: a.numpy() for k, a in arrays.items()}
    state = _unpack(json.loads(config["tree"]), loaded, blobs)
    if not isinstance(state, dict):
        raise ValueError("client state is not a mapping")
    return state


def _pack(value: Any, arrays: Dict[str, np.ndarray], blobs: Dict[str, bytes]) -> Any:
    value = _plain(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        key = str(len(arrays))
        arrays[key] = value
        return {"ndarray": key}
    if isinstance(value, (bytes, bytearray)):
        key = str(len(blobs))
        blobs[key] = bytes(value)
        return {"bytes": key}
    if isinstance(value, dict):
        return {"dict": {str(k): _pack(v, arrays, blobs) for k, v in value.items()}}
    name = type(value).__name__
    if dataclasses.is_dataclass(value) and name in STATE_DATACLASSES:
        module, cls = STATE_DATACLASSES[name]
        if f"{type(value).__module__}.{type(value).__qualname__}" != f"{module}.{cls}":
            raise TypeError(f"{type(value)} is not the allowlisted {module}.{cls}")
        fields = {f.name: _pack(getattr(value, f.name), arrays, blobs) for f in dataclasses.fields(value)}
        return {"dataclass": name, "fields": fields}
    raise TypeError(f"client state value of type {name} cannot be persisted")


def _unpack(node: Any, arrays: Dict[str, np.ndarray], blobs: Dict[str, bytes]) -> Any:
    if not isinstance(node, dict):
        return node
    if "ndarray" in node:
        return arrays[node["ndarray"]]
    if "bytes" in node:
        return blobs[node["bytes"]]
    if "dict" in node:
        return {k: _unpack(v, arrays, blobs) for k, v in node["dict"].items()}
    if "dataclass" in node:
        if node["dataclass"] not in STATE_DATACLASSES:
            raise ValueError(f"client state names a dataclass that is not allowlisted: {node['dataclass']!r}")
        module, cls = STATE_DATACLASSES[node["dataclass"]]
        return getattr(importlib.import_module(module), cls)(
            **{k: _unpack(v, arrays, blobs) for k, v in node["fields"].items()}
        )
    raise ValueError(f"malformed client state node: {sorted(node)}")
