from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 3
# cache/normal/<file>.npz, cache/beta/<spread>/<file>.npz
DEFAULT_CACHE_ROOT = Path("cache")


def value_token(value: float) -> str:
    """A float as a file name part: 0.3 -> 0p3."""
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def candidate_hash(candidates: np.ndarray) -> str:
    candidates = np.asarray(candidates, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(candidates.shape).encode())
    digest.update(candidates.tobytes(order="C"))
    return digest.hexdigest()[:8]


def metadata(kind: str, **values: Any) -> str:
    data = {"cache_version": CACHE_VERSION, "kind": kind, **values}
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def read_metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return json.loads(str(archive["metadata"]))
