"""Voting methods on a weighted ballot profile.

Every method takes
    rankings: (R, C) candidate indices from best to worst, one row per ballot type
    probs:    (pixels, pixels, R) share of voters with each ballot type
and returns the winner per pixel, shape (pixels, pixels).
"""

import numpy as np


def _transfer_matrices(rankings: np.ndarray) -> np.ndarray:
    """T[s, r, c] = 1 if c is the highest ranked candidate of ballot r that is
    not in the eliminated set s (bit c of s set = c eliminated). Shape (2^C, R, C)."""
    n_ballots, n_candidates = rankings.shape
    states = np.arange(1 << n_candidates)
    eliminated = ((states[:, None] >> np.arange(n_candidates)) & 1).astype(bool)
    alive = ~eliminated[:, rankings]
    choice = rankings[np.arange(n_ballots), alive.argmax(axis=-1)]
    return np.eye(n_candidates)[choice]


def _votes_by_state(
    probs: np.ndarray, state: np.ndarray, transfer: np.ndarray
) -> np.ndarray:
    """First choice votes among remaining candidates; state: (pixels, pixels) bitmask."""
    votes = np.empty((*probs.shape[:2], transfer.shape[-1]))
    for s in np.unique(state):
        mask = state == s
        votes[mask] = probs[mask] @ transfer[s]
    return votes


def fptp(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """First past the post."""
    first = np.eye(rankings.shape[1])[rankings[:, 0]]
    return (probs @ first).argmax(axis=-1)


def irv(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Instant runoff: repeatedly eliminate the candidate with the fewest votes.

    Ballots are complete rankings, so a candidate with a majority is never
    eliminated; eliminating until one candidate remains gives the IRV winner.
    """
    n_candidates = rankings.shape[1]
    candidate_bits = 1 << np.arange(n_candidates)
    transfer = _transfer_matrices(rankings)
    state = np.zeros(probs.shape[:2], dtype=np.int64)
    for _ in range(n_candidates - 1):
        votes = _votes_by_state(probs, state, transfer)
        votes[(state[..., None] & candidate_bits) != 0] = np.inf
        state |= candidate_bits[votes.argmin(axis=-1)]
    return ((state[..., None] & candidate_bits) == 0).argmax(axis=-1)


METHODS = {
    "fptp": fptp,
    "irv": irv,
}
