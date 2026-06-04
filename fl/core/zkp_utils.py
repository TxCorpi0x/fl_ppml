"""Shared helpers for fl.core.zkp* modules.

This module centralises small utilities that are used by multiple ZKP
implementations (numpy conversion, HTTP JSON posting). Keeping them in one
place avoids duplication and makes testing easier.
"""

from typing import Dict, Tuple
import logging

import numpy as np
import requests

logger = logging.getLogger(__name__)


def to_numpy(x) -> np.ndarray:
    """Convert tensor-like or array-like inputs to a numpy array.

    - If `x` has a `detach()` method, attempt `x.detach().cpu().numpy()`.
    - If `x` is already a numpy array, return it unchanged.
    - Otherwise fall back to `np.asarray(x)`.
    """
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return np.asarray(x)
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def post_json(url: str, json_obj: Dict, timeout: Tuple[float, float]) -> Dict:
    """POST `json_obj` to `url` and return the parsed JSON response.

    Raises a RuntimeError with a clear message for network/HTTP/JSON errors.
    """
    try:
        resp = requests.post(url, json=json_obj, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
