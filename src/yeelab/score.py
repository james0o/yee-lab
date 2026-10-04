"""Score ballots of voters in the plane: the score a voter gives each candidate.

A score ballot has `levels` scores, 0 to levels - 1. A voter at v gives the closest
candidate the top score and the farthest 0. Which score the others get depends on the
distances r_i = |v - c_i|, in one of three ways (`rule`):

    RANGE   the score in proportion to where r_i is between the closest and the
            farthest, rounded to a whole score:

                score_i = round((r_max - r_i) / (r_max - r_min) * (levels - 1))

            A half is rounded to the even score, like np.round. The candidates in
            between do not move the scale: only the closest and the farthest do. A
            voter as far from every candidate (r_max = r_min) gives them all the top
            score
    AVG     like RANGE, but with the mean distance rbar = (r_1 + ... + r_C) / C in the
            middle of the scale in place of halfway between the closest and the
            farthest: the part of the way goes linearly from 1 at the closest to 1/2
            at rbar, and from there to 0 at the farthest,

                part_i = 1/2 + (rbar - r_i) / (rbar - r_min) / 2   if r_i < rbar
                part_i = (r_max - r_i) / (r_max - rbar) / 2         otherwise
                score_i = round(part_i * (levels - 1))

            RANGE is this with the middle at (r_min + r_max) / 2. Unlike there, every
            candidate moves the scale, through rbar. A voter as far from every
            candidate gives them all the top score, as for RANGE
    DHONDT  the levels - 1 steps from the top score down to 0 are shared out among the
            gaps between neighbours in the order of distance, r_(1) <= ... <= r_(C)
            and g_k = r_(k + 1) - r_(k), by a divisor method: each step in turn goes
            to the gap with the largest g_k / (d_k + delta), d_k the steps it has so
            far (ties: the first such gap). delta = 1 is D'Hondt, which favours the
            large gaps; delta = 1/2 is Sainte-Laguë, which on average favours neither.
            The default, DELTA, is between them. A candidate gets the steps of the gaps
            below it:

                score_(m) = d_m + ... + d_(C - 1)

            A gap may get several steps or none, so candidates at nearly the same
            distance share a score however many levels there are; that is what keeps
            it from Borda, which it would be if every gap got one step. With two levels
            the one step goes to the largest gap, whatever delta: the ballot of
            approval.GAP

Nothing changes when all distances are scaled. With two levels RANGE and AVG are
approval ballots as well. AVG is approval.AVG: the voter approves the candidates closer
than the mean distance (at it, the half is rounded to 0: not approved). RANGE approves
those closer than halfway between the closest and the farthest. For three candidates
that is the ballot of yeelab.approval, at every cut; for more it is none of them.

As the number of levels grows, AVG comes closer to its part of the way, and RANGE and
DHONDT to the part of the way of RANGE, (r_max - r_i) / (r_max - r_min): for delta up to 1, DHONDT keeps every gap within C - 1
steps of its share of them, (levels - 1) g_k / (r_max - r_min), with delta = 1 never
below the share rounded down, and one more level adds a step to one gap and takes none
away. The large gaps gain about as many steps at any number of levels, so as a part of
the top score their advantage fades.

Borders. A RANGE score changes where its part of the way is half a score,
r_max - r_i = (k + 1/2) / (levels - 1) * (r_max - r_min), an AVG score likewise with rbar
for one end of its part of the way; a DHONDT step moves where two
gaps tie, (l + delta) (r_a - r_b) = (k + delta) (r_c - r_d). These are curves, like the
borders of the approval ballots, so the mean score is not a sum over polygons either.
`unscored` gives, for a grid of rectangles, the mean points below the top score of each
candidate in each rectangle; margin/shares.py weighs it with the voters of each rectangle.
"""

import math
from typing import Literal

import numpy as np
from numba import njit

from yeelab import threads
from yeelab.approval import ROWS, by_distance, distances

RANGE = "range"
DHONDT = "dhondt"
AVG = "avg"
Rule = Literal["range", "dhondt", "avg"]
RULES: tuple[Rule, ...] = (RANGE, DHONDT, AVG)
BITS = 4  # of each candidate in the code of a ballot: its points below the top score
MASK = (1 << BITS) - 1
MAX_CANDIDATES = 63 // BITS  # a ballot is a code in an int64
MAX_LEVELS = MASK + 1  # the points below the top score fit the bits of a candidate
DELTA = 0.8  # of DHONDT: each step to the largest g_k / (d_k + DELTA)


def _check(rule, delta):
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {', '.join(RULES)}")
    if not 0 < delta < math.inf:
        raise ValueError(f"delta must be a number above 0, got {delta!r}")


def scored(points, candidates, levels: int, rule: Rule = RANGE, delta: float = DELTA) -> np.ndarray:
    """The score a voter at each of the points (..., 2) gives each candidate, int
    (..., C) from 0 to levels - 1: the ballot from its definition, one voter at a time.
    delta is the divisor of DHONDT; the other rules do not use it."""
    _check(rule, delta)
    r = distances(points, candidates)
    if rule == DHONDT:
        return levels - 1 - _dhondt_below(r, levels - 1, delta)
    if rule == AVG:
        return np.rint(_avg_part(r) * (levels - 1)).astype(np.int64)
    far = r.max(axis=-1, keepdims=True)
    span = far - r.min(axis=-1, keepdims=True)
    part = np.divide(far - r, span, out=np.ones_like(r), where=span > 0)
    return np.rint(part * (levels - 1)).astype(np.int64)


def _avg_part(r):
    """The part of the way of AVG, (..., C), for the distances r (..., C)."""
    near, far = r.min(axis=-1, keepdims=True), r.max(axis=-1, keepdims=True)
    mean = r.mean(axis=-1, keepdims=True)
    closer = 0.5 + 0.5 * np.divide(mean - r, mean - near, out=np.zeros_like(r), where=r < mean)
    farther = 0.5 * np.divide(far - r, far - mean, out=np.zeros_like(r), where=(r >= mean) & (r < far))
    return np.where(far > near, np.where(r < mean, closer, farther), 1.0)


def _dhondt_below(r, top, delta):
    """Points below the top score `top` of each candidate on a DHONDT ballot with the
    divisor delta, int (..., C), for the distances r (..., C): the steps of the gaps
    above it."""
    order = np.argsort(r, axis=-1, kind="stable")  # closest first; ties: the lowest index
    gaps = np.diff(np.take_along_axis(r, order, axis=-1), axis=-1)
    steps = np.zeros(gaps.shape, dtype=np.int64)
    for _ in range(top):  # one step at a time; argmax takes the first of tied gaps
        steps += (gaps / (steps + delta)).argmax(axis=-1)[..., None] == np.arange(gaps.shape[-1])
    below = np.concatenate([np.zeros_like(steps[..., :1]), np.cumsum(steps, axis=-1)], axis=-1)
    return np.take_along_axis(below, np.argsort(order, axis=-1, kind="stable"), axis=-1)

# ---------------------------------------------------------------- Compiled


@njit(inline="always", error_model="numpy")
def _ballot(x, y, candidates, top, rule, delta, r, order, steps):
    """scored() of the voter at (x, y) as a code: BITS bits per candidate, holding the
    points it is below the top score `top`. `rule` is the index of the rule in RULES,
    delta the divisor of DHONDT; r, order and steps (C,) are scratch."""
    n = candidates.shape[0]
    code = 0
    if rule == 1:  # DHONDT
        by_distance(x, y, candidates, r, order)
        steps[:] = 0
        for _ in range(top):
            gap, widest = 0, -1.0
            for k in range(n - 1):
                share = (r[order[k + 1]] - r[order[k]]) / (steps[k] + delta)
                if share > widest:
                    widest, gap = share, k
            steps[gap] += 1
        below = 0  # the steps of the gaps above the candidate
        for k in range(n):
            code |= below << (BITS * order[k])
            below += steps[k]
        return code
    near, far, mean = np.inf, -np.inf, 0.0
    for c in range(n):
        r[c] = math.hypot(x - candidates[c, 0], y - candidates[c, 1])
        near, far = min(near, r[c]), max(far, r[c])
        mean += r[c]
    span = far - near
    if span > 0 and rule == 2:  # AVG
        mean /= n
        for c in range(n):
            if r[c] < mean:
                part = 0.5 + 0.5 * ((mean - r[c]) / (mean - near))
            elif r[c] < far:
                part = 0.5 * ((far - r[c]) / (far - mean))
            else:
                part = 0.0
            code |= (top - np.int64(np.rint(part * top))) << (BITS * c)
    elif span > 0:
        for c in range(n):
            code |= (top - np.int64(np.rint((far - r[c]) / span * top))) << (BITS * c)
    return code


@njit(cache=True, nogil=True, error_model="numpy")
def _unscored(xs, ys, candidates, top, rule, delta, sub, start, stop, out):
    """unscored() of the cells [xs[i], xs[i + 1]] with start <= i < stop, into
    out (C, X, Y)."""
    n = candidates.shape[0]
    r, points = np.empty(n), np.empty(n, dtype=np.int64)
    order, steps = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    left, right = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    for j in range(ys.size):
        left[j] = _ballot(xs[start], ys[j], candidates, top, rule, delta, r, order, steps)
    for i in range(start, stop):
        for j in range(ys.size):
            right[j] = _ballot(xs[i + 1], ys[j], candidates, top, rule, delta, r, order, steps)
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
                    code = _ballot(x, y, candidates, top, rule, delta, r, order, steps)
                    for c in range(n):
                        points[c] += (code >> (BITS * c)) & MASK
            for c in range(n):
                out[c, i, j] = points[c] / (sub * sub)
        left, right = right, left


def unscored(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
             delta: float = DELTA) -> np.ndarray:
    """short[c, i, j] = mean points below the top score that the voters of the rectangle
    [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]] give candidate c, from 0 to levels - 1, shape
    (C, X, Y) for X + 1 and Y + 1 increasing grid lines; delta as in scored(). A
    rectangle whose four corners have the same ballot counts as all of that ballot; in
    the others, which a border crosses, the ballots at sub x sub points are averaged.
    Chunks of the rows run in parallel threads."""
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"score ballots need 2 to {MAX_CANDIDATES} candidates, got {len(candidates)}")
    if not 2 <= levels <= MAX_LEVELS:
        raise ValueError(f"score ballots need 2 to {MAX_LEVELS} levels, got {levels}")
    _check(rule, delta)
    out = np.empty((len(candidates), len(xs) - 1, len(ys) - 1), dtype=np.float32)
    threads.in_chunks(
        lambda a, b: _unscored(xs, ys, candidates, levels - 1, RULES.index(rule), float(delta), sub,
                               a, b, out),
        len(xs) - 1, ROWS)
    return out
