"""Voting methods on a weighted ballot profile.

Every method takes
    rankings: (R, C) candidate indices from best to worst, one row per ballot type
    probs:    (pixels, pixels, R) share of voters with each ballot type
and returns the winner per pixel, shape (pixels, pixels). The *_margin variants of
margin/methods.py pick the same winners from only the shares a method needs.

voronoi() is the reference diagram, not a method: it needs no voters at all.
"""

import numpy as np
from scipy.spatial.distance import cdist

from yeelab.voting import CYCLE, irv_rounds


def voronoi(candidates: np.ndarray, pixels: int) -> np.ndarray:
    """Nearest candidate to each pixel centre ((i + 1/2) / pixels, (j + 1/2) / pixels),
    the median (Beta) or mean (normal) of that pixel's voters. Every Condorcet method
    draws this diagram for normal voters. Same shape as the methods' winners."""
    centres = (np.arange(pixels) + 0.5) / pixels
    points = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1)
    nearest = cdist(points.reshape(-1, 2), candidates).argmin(axis=1)
    return nearest.reshape(pixels, pixels)


def fptp(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """First past the post."""
    first = np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    return (probs @ first).argmax(axis=-1)


def irv(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Instant runoff, see voting.irv_rounds."""
    return irv_rounds(rankings, probs)[0]


def borda(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Borda count: a ballot gives C-1 points to its first choice, C-2 to its
    second, ..., 0 to its last."""
    n_ballots, n_candidates = rankings.shape
    points = np.empty((n_ballots, n_candidates), dtype=probs.dtype)
    points[np.arange(n_ballots)[:, None], rankings] = np.arange(n_candidates)[::-1]
    return (probs @ points).argmax(axis=-1)


def _pairwise_preferences(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """d[..., x, y] = share of voters ranking x above y. Shape (pixels, pixels, C, C)."""
    n_ballots, n_candidates = rankings.shape
    position = np.empty_like(rankings)
    position[np.arange(n_ballots)[:, None], rankings] = np.arange(n_candidates)
    prefers = (position[:, :, None] < position[:, None, :]).astype(probs.dtype)
    d = probs @ prefers.reshape(n_ballots, -1)
    return d.reshape(*probs.shape[:2], n_candidates, n_candidates)


def schulze(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Schulze method: the winner beats or ties every other candidate by
    strength of the strongest (widest) path in the pairwise defeat graph."""
    d = _pairwise_preferences(rankings, probs)
    p = np.where(d > np.swapaxes(d, -1, -2), d, 0.0)
    for k in range(d.shape[-1]):
        p = np.maximum(p, np.minimum(p[..., :, k, None], p[..., None, k, :]))
    return (p >= np.swapaxes(p, -1, -2)).all(axis=-1).argmax(axis=-1)


def condorcet_cycle(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Condorcet winner (beats every other candidate head to head), or CYCLE
    where there is none, i.e. the pairwise majorities form a cycle."""
    d = _pairwise_preferences(rankings, probs)
    wins = (d > np.swapaxes(d, -1, -2)).sum(axis=-1)
    has_winner = wins.max(axis=-1) == rankings.shape[1] - 1
    return np.where(has_winner, wins.argmax(axis=-1), CYCLE)


METHODS = {
    "fptp": fptp,
    "irv": irv,
    "borda": borda,
    "schulze": schulze,
    "condorcet_cycle": condorcet_cycle,
}
