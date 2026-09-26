"""Voting methods on a weighted ballot profile.

Every method takes
    rankings: (R, C) candidate indices from best to worst, one row per ballot type
    probs:    (pixels, pixels, R) share of voters with each ballot type
and returns the winner per pixel, shape (pixels, pixels).
"""

import numpy as np
from scipy.spatial.distance import cdist

from const import CANDIDATES


def ideal(
    rankings: np.ndarray, probs: np.ndarray, candidates: np.ndarray = CANDIDATES
) -> np.ndarray:
    """Reference diagram: the median voter [x, y] of each pixel votes alone for
    the nearest candidate. The median of pixel (i, j) is its centre
    ((i + 1/2) / pixels, (j + 1/2) / pixels) by construction of the Beta parameters
    (and the mean of the normal distribution, where every Condorcet method gives this).
    """
    pixels = probs.shape[0]
    medians = (np.arange(pixels) + 0.5) / pixels
    points = np.stack(np.meshgrid(medians, medians, indexing="ij"), axis=-1)
    nearest = cdist(points.reshape(-1, 2), candidates).argmin(axis=1)
    return nearest.reshape(pixels, pixels)


def _transfer_matrices(rankings: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """T[s, r, c] = 1 if c is the highest ranked candidate of ballot r that is
    not in the eliminated set s (bit c of s set = c eliminated). Shape (2^C, R, C)."""
    n_ballots, n_candidates = rankings.shape
    states = np.arange(1 << n_candidates)
    eliminated = ((states[:, None] >> np.arange(n_candidates)) & 1).astype(bool)
    alive = ~eliminated[:, rankings]
    choice = rankings[np.arange(n_ballots), alive.argmax(axis=-1)]
    return np.eye(n_candidates, dtype=dtype)[choice]


def _votes_by_state(
    probs: np.ndarray, state: np.ndarray, transfer: np.ndarray
) -> np.ndarray:
    """First choice votes among remaining candidates; state: (pixels, pixels) bitmask."""
    votes = np.empty((*probs.shape[:2], transfer.shape[-1]), dtype=probs.dtype)
    for s in np.unique(state):
        mask = state == s
        votes[mask] = probs[mask] @ transfer[s]
    return votes


def fptp(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """First past the post."""
    first = np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    return (probs @ first).argmax(axis=-1)


def irv(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Instant runoff: repeatedly eliminate the candidate with the fewest votes.

    Ballots are complete rankings, so a candidate with a majority is never
    eliminated; eliminating until one candidate remains gives the IRV winner.
    """
    n_candidates = rankings.shape[1]
    candidate_bits = 1 << np.arange(n_candidates)
    transfer = _transfer_matrices(rankings, probs.dtype)
    state = np.zeros(probs.shape[:2], dtype=np.int64)
    for _ in range(n_candidates - 1):
        votes = _votes_by_state(probs, state, transfer)
        votes[(state[..., None] & candidate_bits) != 0] = np.inf
        state |= candidate_bits[votes.argmin(axis=-1)]
    return ((state[..., None] & candidate_bits) == 0).argmax(axis=-1)


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


CYCLE = -1


def condorcet_cycle(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Condorcet winner (beats every other candidate head to head), or CYCLE
    where there is none, i.e. the pairwise majorities form a cycle."""
    d = _pairwise_preferences(rankings, probs)
    wins = (d > np.swapaxes(d, -1, -2)).sum(axis=-1)
    has_winner = wins.max(axis=-1) == rankings.shape[1] - 1
    return np.where(has_winner, wins.argmax(axis=-1), CYCLE)


METHODS = {
    "ideal": ideal,
    "fptp": fptp,
    "irv": irv,
    "borda": borda,
    "schulze": schulze,
    "condorcet_cycle": condorcet_cycle,
}
