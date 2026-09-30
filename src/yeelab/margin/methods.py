"""The Condorcet winner and Schulze on pairwise shares d (..., C, C), d[..., c, e] =
share ranking c above e (shares.py). The other methods are built from blocks in
yeelab.build, whose Schulze() is schulze_margin. pixels/methods.py has the same
methods on the complete profile of every pixel, winners only.

Both *_margin functions return (winner, margin). The margin is >= 0, continuous in the
shares, and 0 on every border between two winners: the winner is decided by
comparing continuous functions of the shares, and the margin is the smallest gap
in a comparison that could change it. (It may also be 0 where such a comparison
ties but the winner stays.) regions.py draws the borders as its zero set.
"""

import numpy as np

from yeelab.voting import CYCLE


def condorcet_margin(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Condorcet winner, or CYCLE where there is none (see pixels.methods.condorcet_cycle).
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
    the winning votes d of pixels.methods.schulze and the winners agree; unlike winning votes,
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
