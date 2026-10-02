"""Score ballots of voters in the plane: the score a voter gives each candidate.

A score ballot has `levels` scores, 0 to levels - 1. A voter at v gives the closest
candidate the top score, the farthest 0, and every candidate the score in proportion to
where its distance r_i = |v - c_i| is between those two, rounded to a whole score:

    score_i = round((r_max - r_i) / (r_max - r_min) * (levels - 1))

A half is rounded to the even score, like np.round. Nothing changes when all distances
are scaled, and the candidates in between do not move the scale: only the closest and
the farthest do. A voter as far from every candidate (r_max = r_min) gives them all the
top score.

With two levels this is an approval ballot: the voter approves the candidates closer
than halfway between the closest and the farthest. For three candidates that is the
ballot of yeelab.approval, at either cut; for more it is neither of them.

Borders. The score of candidate i changes where its part of the way is half a score,
r_max - r_i = (k + 1/2) / (levels - 1) * (r_max - r_min): a curve, like the borders of
the approval ballots, so the mean score is not a sum over polygons either. `unscored`
gives, for a grid of rectangles, the mean points below the top score of each candidate
in each rectangle; margin/shares.py weighs it with the voters of each rectangle.
"""

import math

import numpy as np
from numba import njit

from yeelab import threads
from yeelab.approval import ROWS, distances

BITS = 4  # of each candidate in the code of a ballot: its points below the top score
MASK = (1 << BITS) - 1
MAX_CANDIDATES = 63 // BITS  # a ballot is a code in an int64
MAX_LEVELS = MASK + 1  # the points below the top score fit the bits of a candidate


def scored(points, candidates, levels: int) -> np.ndarray:
    """The score a voter at each of the points (..., 2) gives each candidate, int
    (..., C) from 0 to levels - 1: the ballot from its definition, one voter at a time."""
    r = distances(points, candidates)
    far = r.max(axis=-1, keepdims=True)
    span = far - r.min(axis=-1, keepdims=True)
    part = np.divide(far - r, span, out=np.ones_like(r), where=span > 0)
    return np.rint(part * (levels - 1)).astype(np.int64)

# ---------------------------------------------------------------- Compiled


@njit(inline="always", error_model="numpy")
def _ballot(x, y, candidates, top, r):
    """scored() of the voter at (x, y) as a code: BITS bits per candidate, holding the
    points it is below the top score `top`. r (C,) is scratch."""
    n = candidates.shape[0]
    near, far = np.inf, -np.inf
    for c in range(n):
        r[c] = math.hypot(x - candidates[c, 0], y - candidates[c, 1])
        near, far = min(near, r[c]), max(far, r[c])
    span = far - near
    code = 0
    if span > 0:
        for c in range(n):
            code |= (top - np.int64(np.rint((far - r[c]) / span * top))) << (BITS * c)
    return code


@njit(cache=True, nogil=True, error_model="numpy")
def _unscored(xs, ys, candidates, top, sub, start, stop, out):
    """unscored() of the cells [xs[i], xs[i + 1]] with start <= i < stop, into
    out (C, X, Y)."""
    n = candidates.shape[0]
    r, points = np.empty(n), np.empty(n, dtype=np.int64)
    left, right = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    for j in range(ys.size):
        left[j] = _ballot(xs[start], ys[j], candidates, top, r)
    for i in range(start, stop):
        for j in range(ys.size):
            right[j] = _ballot(xs[i + 1], ys[j], candidates, top, r)
        for j in range(ys.size - 1):
            code = left[j]
            if code == left[j + 1] and code == right[j] and code == right[j + 1]:
                for c in range(n):
                    out[c, i, j] = (code >> (BITS * c)) & MASK
                continue
            points[:] = 0
            for a in range(sub):
                x = xs[i] + (a + 0.5) / sub * (xs[i + 1] - xs[i])
                for b in range(sub):
                    y = ys[j] + (b + 0.5) / sub * (ys[j + 1] - ys[j])
                    code = _ballot(x, y, candidates, top, r)
                    for c in range(n):
                        points[c] += (code >> (BITS * c)) & MASK
            for c in range(n):
                out[c, i, j] = points[c] / (sub * sub)
        left, right = right, left


def unscored(xs, ys, candidates, levels: int, sub: int) -> np.ndarray:
    """short[c, i, j] = mean points below the top score that the voters of the rectangle
    [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]] give candidate c, from 0 to levels - 1, shape
    (C, X, Y) for X + 1 and Y + 1 increasing grid lines. A rectangle whose four corners
    have the same ballot counts as all of that ballot; in the others, which a border
    crosses, the ballots at sub x sub points are averaged. Chunks of the rows run in
    parallel threads."""
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"score ballots need 2 to {MAX_CANDIDATES} candidates, got {len(candidates)}")
    if not 2 <= levels <= MAX_LEVELS:
        raise ValueError(f"score ballots need 2 to {MAX_LEVELS} levels, got {levels}")
    out = np.empty((len(candidates), len(xs) - 1, len(ys) - 1), dtype=np.float32)
    threads.in_chunks(lambda a, b: _unscored(xs, ys, candidates, levels - 1, sub, a, b, out),
                      len(xs) - 1, ROWS)
    return out
