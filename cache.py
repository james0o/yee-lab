from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 1
DEFAULT_CACHE_ROOT = Path("cache")


def _distance_token(distance: float) -> str:
    token = format(float(distance), ".12g")
    return token.replace("-", "m").replace(".", "p")


def _short_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def candidate_hash(candidates: np.ndarray) -> str:
    candidates = np.asarray(candidates, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(candidates.shape).encode())
    digest.update(candidates.tobytes(order="C"))
    return digest.hexdigest()[:8]


def params_identity(pixels: int, distance: float) -> str:
    return _short_hash({"pixels": int(pixels), "distance": float(distance)})


def weights_identity(pixels: int, n_part: int, distance: float) -> str:
    return _short_hash(
        {
            "pixels": int(pixels),
            "n_part": int(n_part),
            "distance": float(distance),
        }
    )


def params_path(
    pixels: int,
    distance: float,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    name = f"P{pixels}_D{_distance_token(distance)}_{params_identity(pixels, distance)}.npz"
    return Path(cache_root) / "params" / name


def weights_path(
    pixels: int,
    n_part: int,
    distance: float,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    name = (
        f"P{pixels}_N{n_part}_D{_distance_token(distance)}_"
        f"{weights_identity(pixels, n_part, distance)}.npz"
    )
    return Path(cache_root) / "weights" / name


def winners_path(
    pixels: int,
    n_part: int,
    distance: float,
    candidates: np.ndarray,
    method: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    candidate_id = candidate_hash(candidates)
    name = (
        f"P{pixels}_N{n_part}_D{_distance_token(distance)}_"
        f"C{candidate_id}_{method}.npz"
    )
    return Path(cache_root) / "winners" / name


def metadata(kind: str, **values: Any) -> str:
    data = {"cache_version": CACHE_VERSION, "kind": kind, **values}
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def read_metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return json.loads(str(archive["metadata"]))
