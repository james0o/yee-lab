from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 3
# pixels/cache/normal/<file>.npz, pixels/cache/beta/<spread>/<file>.npz
DEFAULT_CACHE_ROOT = Path(__file__).parent / "cache"


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


def save_node_probabilities(
    path: Path,
    expected: str,
    rankings: np.ndarray,
    node_probabilities: np.ndarray,
    medians: np.ndarray,
    candidates: np.ndarray,
    **extra: np.ndarray,
) -> None:
    """Save probabilities at the node medians; `expected` is from metadata(...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rankings=rankings,
        node_probabilities=node_probabilities,
        medians=medians,
        candidates=candidates,
        metadata=np.array(expected),
        **extra,
    )


def load_node_probabilities(
    path: Path, expected: str, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(rankings, node_probabilities, medians) saved by save_node_probabilities,
    or None if the file is missing or was saved with other settings."""
    if not path.exists():
        return None
    with np.load(path) as archive:
        if json.loads(str(archive["metadata"])) != json.loads(expected) or \
                not np.array_equal(archive["candidates"], candidates):
            return None
        return archive["rankings"], archive["node_probabilities"], archive["medians"]
