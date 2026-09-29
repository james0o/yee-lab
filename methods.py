"""Voting methods on a weighted ballot profile.

Every method takes
    rankings: (R, C) candidate indices from best to worst, one row per ballot type
    probs:    (pixels, pixels, R) share of voters with each ballot type
and returns the winner per pixel, shape (pixels, pixels).

voronoi() is the reference diagram, not a method: it needs no voters at all.

The *_margin variants return (winner, margin) and take only the shares the method
needs (shares.py): first-choice shares (..., C) or pairwise shares d (..., C, C),
d[..., c, e] = share ranking c above e. The margin is >= 0, continuous in the
shares, and 0 on every border between two winners: the winner is decided by
comparing continuous functions of the shares, and the margin is the smallest gap
in a comparison that could change it. (It may also be 0 where such a comparison
ties but the winner stays.) regions.py draws the borders as its zero set.
"""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

import threads

CHUNK = 4096  # points per task of the compiled loops (threads.py)
_kernel = njit(cache=True, nogil=True, error_model="numpy")


def voronoi(candidates: np.ndarray, pixels: int) -> np.ndarray:
    """Nearest candidate to each pixel centre ((i + 1/2) / pixels, (j + 1/2) / pixels),
    the median (Beta) or mean (normal) of that pixel's voters. Every Condorcet method
    draws this diagram for normal voters. Same shape as the methods' winners."""
    centres = (np.arange(pixels) + 0.5) / pixels
    points = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1)
    nearest = cdist(points.reshape(-1, 2), candidates).argmin(axis=1)
    return nearest.reshape(pixels, pixels)


def _top_two(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Highest score (winner) and its lead over the second."""
    top = np.partition(scores, -2, axis=-1)
    return scores.argmax(axis=-1), top[..., -1] - top[..., -2]


def fptp_margin(first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First past the post on first-choice shares (..., C)."""
    return _top_two(first)


def fptp(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """First past the post."""
    first = np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    return (probs @ first).argmax(axis=-1)


@_kernel
def _irv_points(rankings, probs, winner, margin):
    """irv_margin for probs (P, R) into winner (P,) and margin (P,).

    Every ballot type points to its highest ranked remaining candidate, so an
    elimination only moves the ballots that pointed to the eliminated candidate:
    at most R C steps per point in all rounds together.
    """
    n_ballots, n_candidates = rankings.shape
    tally = np.empty(n_candidates)
    alive = np.empty(n_candidates, dtype=np.bool_)
    top = np.empty(n_ballots, dtype=np.int64)
    for p in range(probs.shape[0]):
        tally[:] = 0.0
        alive[:] = True
        for r in range(n_ballots):
            top[r] = 0
            tally[rankings[r, 0]] += probs[p, r]
        gap = np.inf
        for _ in range(n_candidates - 1):
            # the two lowest tallies; ties go to the first candidate, like argmin
            lowest, second = -1, -1
            for c in range(n_candidates):
                if not alive[c]:
                    continue
                if lowest < 0 or tally[c] < tally[lowest]:
                    second, lowest = lowest, c
                elif second < 0 or tally[c] < tally[second]:
                    second = c
            gap = min(gap, tally[second] - tally[lowest])
            alive[lowest] = False
            for r in range(n_ballots):
                if rankings[r, top[r]] == lowest:
                    k = top[r] + 1
                    while not alive[rankings[r, k]]:
                        k += 1
                    top[r] = k
                    tally[rankings[r, k]] += probs[p, r]
        for c in range(n_candidates):
            if alive[c]:
                winner[p] = c
        margin[p] = gap


def irv_margin(rankings: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instant runoff: repeatedly eliminate the candidate with the fewest votes.

    Ballots are complete rankings, so a candidate with a majority is never
    eliminated; eliminating until one candidate remains gives the IRV winner.
    The margin is the smallest gap between the two lowest tallies of any round:
    the winner can change only where some elimination flips, and the gap is 0 on
    both sides of such a flip.
    """
    shape = probs.shape[:-1]
    flat = np.ascontiguousarray(probs.reshape(-1, probs.shape[-1]))
    rankings = np.ascontiguousarray(rankings, dtype=np.int64)
    winner = np.empty(len(flat), dtype=np.int64)
    margin = np.empty(len(flat))
    threads.in_chunks(lambda a, b: _irv_points(rankings, flat[a:b], winner[a:b], margin[a:b]),
                      len(flat), CHUNK)
    return winner.reshape(shape), margin.reshape(shape)


def irv(rankings: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Instant runoff, see irv_margin."""
    return irv_margin(rankings, probs)[0]


def borda_margin(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Borda count on pairwise shares: a ballot gives c one point per candidate
    ranked below c, so c scores sum_e d[..., c, e]."""
    return _top_two(d.sum(axis=-1))


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


def condorcet_margin(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Condorcet winner, or CYCLE where there is none (see condorcet_cycle).
    worst[c] = min_e (d[c, e] - d[e, c]) is c's narrowest head-to-head result; c is
    the Condorcet winner where it is positive. The margin |max_c worst[c]| is the
    winner's narrowest win, or, in a cycle, how far every candidate is from beating
    everyone."""
    n = d.shape[-1]
    lead = d - np.swapaxes(d, -1, -2)
    worst = np.where(np.eye(n, dtype=bool), np.inf, lead).min(axis=-1)
    best = worst.max(axis=-1)
    return np.where(best > 0, worst.argmax(axis=-1), CYCLE), np.abs(best)


def _schulze_paths(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Schulze winner and margin from the widest paths (see schulze_margin)."""
    n = d.shape[-1]
    p = np.maximum(d - np.swapaxes(d, -1, -2), 0.0)
    through = np.empty_like(p)
    for k in range(n):  # in place: row and column k do not change in step k (p[k, k] = 0)
        np.minimum(p[..., :, k, None], p[..., None, k, :], out=through)
        np.maximum(p, through, out=p)
    beaten = (p - np.swapaxes(p, -1, -2)).max(axis=-2)  # [..., e] = max_f p[f, e] - p[e, f]
    winner = (beaten <= 0).argmax(axis=-1)
    others = np.where(np.arange(n) == winner[..., None], np.inf, beaten)
    return winner, np.maximum(others.min(axis=-1), 0.0)


def schulze_margin(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Schulze method on pairwise shares, with margins d - d^T as link strengths.
    For complete rankings d + d^T = 1, so the margin 2 d - 1 orders the links like
    the winning votes d of schulze() and the winners agree; unlike winning votes,
    the path strengths are then continuous in d.

    The winner w is the candidate that no one beats (p[e, w] <= p[w, e] for all e).
    It changes only where another candidate becomes unbeaten, so the margin is how
    far the others are from that: min over e != w of max_f (p[f, e] - p[e, f]).
    (min_e p[w, e] - p[e, w] would not do: it can be 0 on a whole area, where two
    widest paths share their weakest link.)

    A Condorcet winner is the Schulze winner, so the widest paths are only computed
    where there is none. Elsewhere the Condorcet margin stands in: it is positive,
    at most the margin above (p[w, e] >= d[w, e] - d[e, w], p[e, w] = 0), and 0 on
    the border of the Condorcet region, so it vanishes on the same borders.
    """
    winner, margin = condorcet_margin(d)
    cycle = winner == CYCLE
    if cycle.any():
        winner[cycle], margin[cycle] = _schulze_paths(d[cycle])
    return winner, margin


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
