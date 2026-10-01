"""Approval ballots of voters in the plane: who a voter approves.

A voter at v puts the candidates in order of distance, r_(1) <= ... <= r_(C) with
r_i = |v - c_i|, and approves the closest ones, down to a cut. Where the cut is depends
on the gaps between neighbours in that order, g_k = r_(k + 1) - r_(k), in one of two
ways (`cut`):

    HALF   the closest half of the candidates, C // 2 of them. With an odd number the
           middle candidate goes with the neighbour it is closer to: it is approved
           too if the gap above it is smaller than the gap below it (ties: it is not).
           So the cut is the larger of the two gaps next to the middle candidate
    GAP    the candidates above the largest gap of all (ties: the first such gap)

Either way the closest candidate is approved and the farthest is not, and nothing
changes when all distances are scaled. For three candidates the two are the same
ballot, and for an even number HALF does not look at the distances at all: it is the
top half of the ranking.

The gaps are those of the distances, not of their squares. A voter at squared distances
0, 0.35, 0.45, 0.55 and 1 has the gaps 0.59, 0.08, 0.07 and 0.26: the candidate the
voter stands on is far ahead of the others, and GAP approves only that one. (The squares
have the gaps 0.35, 0.1, 0.1 and 0.45, which would approve all but the last.)

Borders. Two ballots meet where two gaps are equal, r_a - r_b = r_c - r_d. That is a
curve, not a line like the bisectors, so the voters who approve a candidate are not a
union of polygons, and their share has no edge integrals. `coverage` gives, for a grid
of rectangles, the part of each rectangle that approves each candidate; margin/shares.py
weighs it with the voters of each rectangle.
"""

import math
from typing import Literal

import numpy as np
from numba import njit

from yeelab import threads

HALF = "half"
GAP = "gap"
Cut = Literal["half", "gap"]
CUTS: tuple[Cut, ...] = (HALF, GAP)
MAX_CANDIDATES = 62  # a ballot is a bit mask in an int64
ROWS = 8  # grid rows per task of the compiled loop (threads.py)


def distances(points, candidates) -> np.ndarray:
    """r[..., c] = |p - c| for points (..., 2) and candidates (C, 2)."""
    points = np.asarray(points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    return np.sqrt(((points[..., None, :] - candidates) ** 2).sum(axis=-1))


def approved(points, candidates, cut: Cut) -> np.ndarray:
    """Whether a voter at each of the points (..., 2) approves each candidate, bool
    (..., C): the ballot from its definition, one voter at a time."""
    r = distances(points, candidates)
    n = r.shape[-1]
    order = np.argsort(r, axis=-1, kind="stable")  # closest first; ties: the lowest index
    gaps = np.diff(np.take_along_axis(r, order, axis=-1), axis=-1)
    if cut == GAP:
        count = gaps.argmax(axis=-1) + 1
    else:
        count = np.full(r.shape[:-1], n // 2)
        if n % 2:  # the middle candidate: the gap below it against the gap above it
            count += gaps[..., n // 2] > gaps[..., n // 2 - 1]
    position = np.argsort(order, axis=-1, kind="stable")
    return position < count[..., None]

# ---------------------------------------------------------------- Compiled


@njit(inline="always", error_model="numpy")
def _ballot(x, y, candidates, gap, r, order):
    """approved() of the voter at (x, y) as a bit mask: bit c is set if candidate c is
    approved. `gap` is whether the cut is GAP; r and order (C,) are scratch."""
    n = candidates.shape[0]
    for c in range(n):  # insertion sort by distance; ties keep the lowest index first
        r[c] = math.hypot(x - candidates[c, 0], y - candidates[c, 1])
        k = c
        while k > 0 and r[order[k - 1]] > r[c]:
            order[k] = order[k - 1]
            k -= 1
        order[k] = c
    if gap:
        count, widest = 1, -1.0
        for k in range(n - 1):
            step = r[order[k + 1]] - r[order[k]]
            if step > widest:
                widest, count = step, k + 1
    else:
        count = n // 2
        if n % 2 == 1 and r[order[count + 1]] - r[order[count]] > r[order[count]] - r[order[count - 1]]:
            count += 1
    mask = 0
    for k in range(count):
        mask |= 1 << order[k]
    return mask


@njit(cache=True, nogil=True, error_model="numpy")
def _coverage(xs, ys, candidates, gap, sub, start, stop, out):
    """coverage() of the cells [xs[i], xs[i + 1]] with start <= i < stop, into
    out (C, X, Y)."""
    n = candidates.shape[0]
    r, order = np.empty(n), np.empty(n, dtype=np.int64)
    counts = np.empty(n, dtype=np.int64)
    left, right = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    for j in range(ys.size):
        left[j] = _ballot(xs[start], ys[j], candidates, gap, r, order)
    for i in range(start, stop):
        for j in range(ys.size):
            right[j] = _ballot(xs[i + 1], ys[j], candidates, gap, r, order)
        for j in range(ys.size - 1):
            mask = left[j]
            if mask == left[j + 1] and mask == right[j] and mask == right[j + 1]:
                for c in range(n):
                    out[c, i, j] = (mask >> c) & 1
                continue
            counts[:] = 0
            for a in range(sub):
                x = xs[i] + (a + 0.5) / sub * (xs[i + 1] - xs[i])
                for b in range(sub):
                    y = ys[j] + (b + 0.5) / sub * (ys[j + 1] - ys[j])
                    mask = _ballot(x, y, candidates, gap, r, order)
                    for c in range(n):
                        counts[c] += (mask >> c) & 1
            for c in range(n):
                out[c, i, j] = counts[c] / (sub * sub)
        left, right = right, left


def coverage(xs, ys, candidates, cut: Cut, sub: int) -> np.ndarray:
    """cover[c, i, j] = part of the rectangle [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]]
    whose voters approve candidate c, shape (C, X, Y) for X + 1 and Y + 1 increasing
    grid lines. A rectangle whose four corners have the same ballot counts as all of
    that ballot; in the others, which a border crosses, the ballots at sub x sub points
    are averaged. Chunks of the rows run in parallel threads."""
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"approval ballots need 2 to {MAX_CANDIDATES} candidates, got {len(candidates)}")
    if cut not in CUTS:
        raise ValueError(f"unknown cut {cut!r}; choose from {', '.join(CUTS)}")
    out = np.empty((len(candidates), len(xs) - 1, len(ys) - 1), dtype=np.float32)
    threads.in_chunks(lambda a, b: _coverage(xs, ys, candidates, cut == GAP, sub, a, b, out),
                      len(xs) - 1, ROWS)
    return out
