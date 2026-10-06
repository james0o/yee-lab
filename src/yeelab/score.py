"""Score ballots of voters in the plane: the score a voter gives each candidate.

A score ballot has `levels` scores, 0 to levels - 1. A voter at v gives the closest
candidate the top score and the farthest 0. Which score the others get depends on the
distances r_i = |v - c_i|, in one of five ways (`rule`):

    RANGE   the score by where r_i is between the closest and the farthest, its part
            of the way to the power `power`, rounded to a whole score:

                score_i = round(((r_max - r_i) / (r_max - r_min))^power * (levels - 1))

            A half is rounded to the even score, like np.round. power = 1, the
            default POWER, is in proportion to the distance; above 1 the top scores
            are kept for the candidates near the closest (with two levels the voter
            approves those beyond 2^(-1 / power) of the way, not halfway), below 1 they
            reach farther. The web UI has a slider for it. The candidates in between do
            not move the scale: only the closest and the farthest do. A voter as far
            from every candidate (r_max = r_min) gives them all the top score
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
            it from Borda, which it would be if every gap got one step
    HYBRID  DHONDT for the candidates closer than the midrange m = (r_min + r_max) / 2
            and RANGE for the others, each on its own part of the scale. Of the
            T = levels - 1 steps, the upper ceil(T / 2) are shared out by DHONDT's
            divisor method among the gaps between the closer candidates and the gap
            from the last of them to m; the farther candidates get the lower
            floor(T / 2) in proportion to where they are between m and the farthest:

                score_(i) = floor(T / 2) + d_i + ... + d_k                    if r_(i) < m
                score_i = round((r_max - r_i) / (r_max - m) * floor(T / 2))  otherwise

            with d_k the steps of the gap m - r_(k) from the last closer candidate.
            Every closer candidate gets floor(T / 2) or more and every other one that
            or less, so a closer candidate never gets a lower score. The voter tells the
            near candidates apart by their gaps, the far ones by distance alone, and
            the far ones do not move the cut among the near ones. A voter as far from
            every candidate gives them all the top score
    CLUSTER RANGE, unless that splits a cluster of candidates: the scores s_i from 0
            to T = levels - 1, never less for a closer candidate, T for the closest
            and 0 for the farthest, that minimize

                sum_i (s_i / T - part_i)^2
                    + mu * sum_k w_k * max(0, 1 - T g_k) * [s_(k) != s_(k + 1)]

            with part_i RANGE's part of the way, g_k the gap between the k-th and the
            next distinct distance as a part of r_max - r_min, and w_k how firmly gap
            k holds a cluster: the strongest run of neighbouring distances it lies in,
            1 - kappa * (largest gap inside the run) / (smaller gap around it), 0 at
            least. A split costs nothing at a gap of a step or more (T g_k >= 1), nor at
            the gaps next to the closest and the farthest, whose scores are fixed. The
            defaults are MU and KAPPA; the web UI has a slider for each. mu = 0 is
            RANGE; with few levels a tight cluster keeps one score, with many it is
            graded like RANGE. A voter as far from every candidate gives them all the
            top score

Nothing changes when all distances are scaled. With two levels every rule is an
approval ballot: the voter approves the candidates with the top score, from the closest
alone to all but the farthest. RANGE approves those closer than halfway between the
closest and the farthest, AVG those closer than the mean distance (at it, the half is
rounded to 0: not approved), DHONDT, whatever delta, those above the largest gap
(ties: the first) and HYBRID, whatever delta, those above the largest gap among the
candidates closer than halfway, the gap from the last of them to halfway included
(ties: the first). CLUSTER approves those RANGE approves, unless that cut splits a
cluster and another gap costs less. AVG's is the best ballot of a voter with the utility -r_i who takes
every pair of candidates to be as likely to tie (Weber): approving c is worth
sum_e (r_e - r_c), which is positive for these. For three candidates RANGE, AVG and
DHONDT are the same ballot, and HYBRID approves the middle one only closer than a
quarter of the way, r_(2) < (3 r_(1) + r_(3)) / 4; for more candidates all four differ.
The gaps are those of the distances, not of their squares: a voter at the squared
distances 0, 0.35, 0.45, 0.55 and 1 has the gaps 0.59, 0.08, 0.07 and 0.26, so DHONDT
approves only the candidate the voter stands on.

As the number of levels grows, AVG comes closer to its part of the way, and RANGE,
DHONDT and HYBRID to that of RANGE, (r_max - r_i) / (r_max - r_min): for delta up to 1,
DHONDT keeps every gap within C - 1 steps of its share of them, (levels - 1) g_k /
(r_max - r_min), with delta = 1 never below the share rounded down, and one more level
adds a step to one gap and takes none away. The large gaps gain about as many steps at
any number of levels, so as a part of the top score their advantage fades. HYBRID's two
halves are the two halves of that part of the way.

Borders. A RANGE score changes where its part of the way is half a score,
r_max - r_i = ((k + 1/2) / (levels - 1))^(1 / power) * (r_max - r_min), an AVG score likewise with
rbar for one end of its part of the way; a DHONDT step moves where two gaps tie,
(l + delta) (r_a - r_b) = (k + delta) (r_c - r_d). A HYBRID score changes where one of
its DHONDT steps moves, with m for r_a or r_c at the gap to the midrange, where a
candidate crosses m, 2 r_i = r_min + r_max, and where a RANGE score of its lower half is
half a score. A CLUSTER score changes where two choices of scores cost the same. Unlike
the bisectors these are curves, so the mean score is not a sum
over polygons. `unscored` gives, for a grid of rectangles, the mean points below the
top score of each candidate in each rectangle; margin/shares.py weighs it with the
voters of each rectangle.
"""

import math
from typing import Literal

import numpy as np
from numba import njit

from yeelab import threads

RANGE = "range"
DHONDT = "dhondt"
AVG = "avg"
HYBRID = "hybrid"
CLUSTER = "cluster"
Rule = Literal["range", "dhondt", "avg", "hybrid", "cluster"]
RULES: tuple[Rule, ...] = (RANGE, DHONDT, AVG, HYBRID, CLUSTER)
BITS = 4  # of each candidate in the code of a ballot: its points below the top score
MASK = (1 << BITS) - 1
MAX_CANDIDATES = 63 // BITS  # a ballot is a code in an int64
MAX_LEVELS = MASK + 1  # the points below the top score fit the bits of a candidate
DELTA = 0.8  # of DHONDT and HYBRID: each step to the largest g_k / (d_k + DELTA)
MU, KAPPA = 0.1, 2.0  # of CLUSTER: the cost of a split, and how much farther a cluster's neighbours are
POWER = 1.0  # of RANGE: the part of the way to this power; 1 is in proportion to the distance
ROWS = 8  # grid rows per task of the compiled loop (threads.py)


def distances(points, candidates) -> np.ndarray:
    """r[..., c] = |p - c| for points (..., 2) and candidates (C, 2)."""
    points = np.asarray(points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    return np.sqrt(((points[..., None, :] - candidates) ** 2).sum(axis=-1))


def _check(rule, delta, mu=MU, kappa=KAPPA, power=POWER):
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {', '.join(RULES)}")
    if not 0 < delta < math.inf:
        raise ValueError(f"delta must be a number above 0, got {delta!r}")
    if not 0 <= mu < math.inf:
        raise ValueError(f"mu must be a number of 0 or more, got {mu!r}")
    if not 0 < kappa < math.inf:
        raise ValueError(f"kappa must be a number above 0, got {kappa!r}")
    if not 0 < power < math.inf:
        raise ValueError(f"power must be a number above 0, got {power!r}")


def scored(points, candidates, levels: int, rule: Rule = RANGE, delta: float = DELTA, *, mu: float = MU,
           kappa: float = KAPPA, power: float = POWER) -> np.ndarray:
    """The score a voter at each of the points (..., 2) gives each candidate, int
    (..., C) from 0 to levels - 1: the ballot from its definition, one voter at a time.
    delta is the divisor of DHONDT and HYBRID, mu and kappa the cost of a split and the
    cohesion of CLUSTER, power that of RANGE; the other rules do not use them."""
    return from_distances(distances(points, candidates), levels, rule, delta, mu=mu, kappa=kappa, power=power)


def from_distances(r, levels: int, rule: Rule = RANGE, delta: float = DELTA, *, mu: float = MU,
                   kappa: float = KAPPA, power: float = POWER) -> np.ndarray:
    """scored() of voters at the distances r (..., C) from the candidates, wherever they
    are: the ballot depends on nothing else. docs/ballots.py gives it the distances of
    real voters."""
    _check(rule, delta, mu, kappa, power)
    r = np.asarray(r, dtype=np.float64)
    if rule == CLUSTER:
        return levels - 1 - _cluster_below(r, levels - 1, mu, kappa)
    if rule == DHONDT:
        return levels - 1 - _dhondt_below(r, levels - 1, delta)
    if rule == HYBRID:
        return levels - 1 - _hybrid_below(r, levels - 1, delta)
    if rule == AVG:
        return np.rint(_avg_part(r) * (levels - 1)).astype(np.int64)
    far = r.max(axis=-1, keepdims=True)
    span = far - r.min(axis=-1, keepdims=True)
    part = np.divide(far - r, span, out=np.ones_like(r), where=span > 0)
    return np.rint(part ** power * (levels - 1)).astype(np.int64)


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


def _hybrid_below(r, top, delta):
    """Points below the top score `top` of each candidate on a HYBRID ballot with the
    divisor delta, int (..., C), for the distances r (..., C): for a candidate closer than
    the midrange, the steps of the gaps above it of the upper ceil(top / 2); for the
    others, those the lower floor(top / 2) do not give."""
    order = np.argsort(r, axis=-1, kind="stable")  # closest first; ties: the lowest index
    s = np.take_along_axis(r, order, axis=-1)
    near, far = s[..., :1], s[..., -1:]
    mid = (near + far) / 2
    closer = s < mid  # the closest ones, a start of the order
    # the gaps between closer candidates, and from the last of them to the midrange
    gaps = np.where(closer[..., 1:], np.diff(s, axis=-1),
                    np.where(closer[..., :-1], mid - s[..., :-1], -np.inf))
    steps = np.zeros(gaps.shape, dtype=np.int64)
    for _ in range((top + 1) // 2):  # as DHONDT, among those gaps alone
        steps += (gaps / (steps + delta)).argmax(axis=-1)[..., None] == np.arange(gaps.shape[-1])
    below = np.concatenate([np.zeros_like(steps[..., :1]), np.cumsum(steps, axis=-1)], axis=-1)
    part = np.divide(far - s, far - mid, out=np.ones_like(s), where=far > mid)
    below = np.where(closer, below, top - np.rint(part * (top // 2)).astype(np.int64))
    below = np.where(far > near, below, 0)
    return np.take_along_axis(below, np.argsort(order, axis=-1, kind="stable"), axis=-1)


def _cluster_below(r, top, mu, kappa):
    """Points below the top score `top` of each candidate on a CLUSTER ballot, int
    (..., C), for the distances r (..., C)."""
    flat = np.ascontiguousarray(r.reshape(-1, r.shape[-1]))
    out = np.empty(flat.shape, dtype=np.int64)
    _cluster_ballots(flat, top, float(mu), float(kappa), out)
    return out.reshape(r.shape)

# ---------------------------------------------------------------- Compiled


@njit(cache=True, error_model="numpy")
def _cohesion(g, kappa, w):
    """How firmly each gap holds a cluster, into w (m,), for the gaps g (m,) between
    neighbouring distinct distances in order: the strongest run of distances i..j that
    gap lies in, 1 - kappa * (largest gap inside) / (smaller gap around), 0 at least.
    The run of all distances has no gap around it."""
    m = g.size
    w[:] = 0.0
    for i in range(m):
        left, inner = g[i - 1] if i > 0 else np.inf, 0.0
        for j in range(i + 1, m + 1 if i > 0 else m):
            inner = max(inner, g[j - 1])  # the largest of g[i:j]
            if kappa * inner >= left:
                break  # 0 or less for this run and the longer ones
            strength = 1.0 - kappa * inner / min(left, g[j] if j < m else np.inf)
            for k in range(i, j):
                w[k] = max(w[k], strength)


@njit(cache=True, error_model="numpy")
def _cluster(r, order, top, mu, kappa, work, back, below):
    """CLUSTER's ballot of the voter at the distances r (C,), with order (C,) the
    candidates closest first: the points below the top score `top` of each distinct
    distance, closest first, into below. work (4, max(C, top + 1)) and back (C, top + 1)
    are scratch. Dynamic programming over the distances in order: cost[v] is the least
    cost so far with the last score v, and back[k, v] the score before it, the same v if
    that costs no more, else the lowest of the cheapest."""
    values, g, w, cost = work[0], work[1], work[2], work[3]
    n = 0
    for k in range(r.size):
        if n == 0 or r[order[k]] > values[n - 1]:
            values[n] = r[order[k]]
            n += 1
    below[0] = 0
    if n == 1:
        return
    span = values[n - 1] - values[0]
    for k in range(n - 1):
        g[k] = (values[k + 1] - values[k]) / span
    _cohesion(g[:n - 1], kappa, w[:n - 1])
    cost[:] = np.inf
    cost[top] = 0.0
    for k in range(1, n):
        split = 0.0 if k == 1 or k == n - 1 else mu * w[k - 1] * max(0.0, 1.0 - top * g[k - 1])
        part = (values[n - 1] - values[k]) / span
        least, at = np.inf, -1  # of cost[u] + split over u > v, the lowest such u
        for v in range(top, -1, -1):
            stay = cost[v]
            cost[v] = min(stay, least) + (v / top - part) ** 2
            back[k, v] = at if least < stay else v
            if stay + split <= least:
                least, at = stay + split, v
    v = 0
    for k in range(n - 1, 0, -1):
        below[k] = top - v
        v = back[k, v]


@njit(cache=True, error_model="numpy")
def _cluster_ballots(r, top, mu, kappa, out):
    """_cluster_below() of the voters r (V, C), into out (V, C)."""
    n = r.shape[1]
    work, back = np.empty((4, max(n, top + 1))), np.empty((n, top + 1), dtype=np.int64)
    below = np.empty(n, dtype=np.int64)
    for i in range(r.shape[0]):
        order = np.argsort(r[i], kind="mergesort")
        _cluster(r[i], order, top, mu, kappa, work, back, below)
        d = 0
        for k in range(n):
            if k > 0 and r[i, order[k]] > r[i, order[k - 1]]:
                d += 1
            out[i, order[k]] = below[d]


@njit(inline="always", error_model="numpy")
def by_distance(x, y, candidates, r, order):
    """The distances r[c] = |(x, y) - c| and the candidates in order of distance,
    closest first, into r and order (C,); ties keep the lowest index first, like a
    stable argsort."""
    for c in range(candidates.shape[0]):  # insertion sort
        r[c] = math.hypot(x - candidates[c, 0], y - candidates[c, 1])
        k = c
        while k > 0 and r[order[k - 1]] > r[c]:
            order[k] = order[k - 1]
            k -= 1
        order[k] = c


@njit(inline="always", error_model="numpy")
def _ballot(x, y, candidates, top, rule, delta, mu, kappa, power, r, order, steps, work, back):
    """scored() of the voter at (x, y) as a code: BITS bits per candidate, holding the
    points it is below the top score `top`. `rule` is the index of the rule in RULES,
    delta the divisor of DHONDT and HYBRID, mu and kappa those of CLUSTER, power that of
    RANGE; r, order and
    steps (C,), work and back (those of _cluster) are scratch."""
    n = candidates.shape[0]
    code = 0
    if rule == 4:  # CLUSTER
        by_distance(x, y, candidates, r, order)
        _cluster(r, order, top, mu, kappa, work, back, steps)
        d = 0
        for k in range(n):
            if k > 0 and r[order[k]] > r[order[k - 1]]:
                d += 1
            code |= steps[d] << (BITS * order[k])
        return code
    if rule == 3:  # HYBRID
        by_distance(x, y, candidates, r, order)
        near, far = r[order[0]], r[order[n - 1]]
        if far == near:
            return code
        mid = (near + far) / 2
        closer = 1  # the closer candidates are the first `closer` of the order
        while r[order[closer]] < mid:
            closer += 1
        steps[:] = 0
        for _ in range((top + 1) // 2):
            gap, widest = 0, -1.0
            for k in range(closer):
                end = r[order[k + 1]] if k < closer - 1 else mid
                share = (end - r[order[k]]) / (steps[k] + delta)
                if share > widest:
                    widest, gap = share, k
            steps[gap] += 1
        below = 0  # the steps of the gaps above the candidate
        for k in range(closer):
            code |= below << (BITS * order[k])
            below += steps[k]
        for k in range(closer, n):
            c = order[k]
            code |= (top - np.int64(np.rint((far - r[c]) / (far - mid) * (top // 2)))) << (BITS * c)
        return code
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
            part = (far - r[c]) / span
            if power != 1.0:
                part = part ** power
            code |= (top - np.int64(np.rint(part * top))) << (BITS * c)
    return code


@njit(cache=True, nogil=True, error_model="numpy")
def _unscored(xs, ys, candidates, top, rule, delta, mu, kappa, power, sub, start, stop, out):
    """unscored() of the cells [xs[i], xs[i + 1]] with start <= i < stop, into
    out (C, X, Y)."""
    n = candidates.shape[0]
    r, points = np.empty(n), np.empty(n, dtype=np.int64)
    order, steps = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    work, back = np.empty((4, max(n, top + 1))), np.empty((n, top + 1), dtype=np.int64)
    left, right = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    for j in range(ys.size):
        left[j] = _ballot(xs[start], ys[j], candidates, top, rule, delta, mu, kappa, power, r, order, steps,
                          work, back)
    for i in range(start, stop):
        for j in range(ys.size):
            right[j] = _ballot(xs[i + 1], ys[j], candidates, top, rule, delta, mu, kappa, power,
                               r, order, steps, work, back)
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
                    code = _ballot(x, y, candidates, top, rule, delta, mu, kappa, power, r, order, steps,
                                   work, back)
                    for c in range(n):
                        points[c] += (code >> (BITS * c)) & MASK
            for c in range(n):
                out[c, i, j] = points[c] / (sub * sub)
        left, right = right, left


def unscored(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
             delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
             power: float = POWER) -> np.ndarray:
    """short[c, i, j] = mean points below the top score that the voters of the rectangle
    [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]] give candidate c, from 0 to levels - 1, shape
    (C, X, Y) for X + 1 and Y + 1 increasing grid lines; delta, mu, kappa and power as in
    scored(). A
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
    _check(rule, delta, mu, kappa, power)
    out = np.empty((len(candidates), len(xs) - 1, len(ys) - 1), dtype=np.float32)
    threads.in_chunks(
        lambda a, b: _unscored(xs, ys, candidates, levels - 1, RULES.index(rule), float(delta), float(mu),
                               float(kappa), float(power), sub, a, b, out),
        len(xs) - 1, ROWS)
    return out


@njit(cache=True, nogil=True, error_model="numpy")
def _comparisons(xs, ys, candidates, top, rule, delta, mu, kappa, power, sub, start, stop, out):
    """Strict score comparisons over cells [xs[i], xs[i + 1]], into out (C, C, X, Y)."""
    n = candidates.shape[0]
    r, order, steps = np.empty(n), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    work, back = np.empty((4, max(n, top + 1))), np.empty((n, top + 1), dtype=np.int64)
    left, right = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    counts = np.empty((n, n), dtype=np.int64)
    for j in range(ys.size):
        left[j] = _ballot(xs[start], ys[j], candidates, top, rule, delta, mu, kappa, power, r, order, steps,
                          work, back)
    for i in range(start, stop):
        for j in range(ys.size):
            right[j] = _ballot(xs[i + 1], ys[j], candidates, top, rule, delta, mu, kappa, power,
                               r, order, steps, work, back)
        for j in range(ys.size - 1):
            code = left[j]
            if code == left[j + 1] and code == right[j] and code == right[j + 1]:
                for c in range(n):
                    below_c = (code >> (BITS * c)) & MASK
                    for e in range(c + 1, n):
                        below_e = (code >> (BITS * e)) & MASK
                        difference = 1 if below_c < below_e else -1 if below_c > below_e else 0
                        out[c, e, i, j] = difference
                        out[e, c, i, j] = -difference
                continue
            counts[:, :] = 0
            for a in range(sub):
                x = xs[i] + (a + 0.5) / sub * (xs[i + 1] - xs[i])
                for b in range(sub):
                    y = ys[j] + (b + 0.5) / sub * (ys[j + 1] - ys[j])
                    code = _ballot(x, y, candidates, top, rule, delta, mu, kappa, power, r, order, steps,
                                   work, back)
                    for c in range(n):
                        below_c = (code >> (BITS * c)) & MASK
                        for e in range(c + 1, n):
                            below_e = (code >> (BITS * e)) & MASK
                            if below_c < below_e:
                                counts[c, e] += 1
                                counts[e, c] -= 1
                            elif below_c > below_e:
                                counts[c, e] -= 1
                                counts[e, c] += 1
            for c in range(n):
                for e in range(c + 1, n):
                    difference = counts[c, e] / (sub * sub)
                    out[c, e, i, j] = difference
                    out[e, c, i, j] = -difference
        left, right = right, left


def comparisons(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
                delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
                power: float = POWER) -> np.ndarray:
    """short[c, e, i, j] = share strictly scoring c above e minus e above c among
    voters in each rectangle [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]]. Voters who give
    the pair equal scores contribute 0. Curved score borders are sampled at sub x sub
    points in cells where the ballot changes; output shape is (C, C, X, Y)."""
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"score ballots need 2 to {MAX_CANDIDATES} candidates, got {len(candidates)}")
    if not 2 <= levels <= MAX_LEVELS:
        raise ValueError(f"score ballots need 2 to {MAX_LEVELS} levels, got {levels}")
    _check(rule, delta, mu, kappa, power)
    out = np.zeros((len(candidates), len(candidates), len(xs) - 1, len(ys) - 1), dtype=np.float32)
    threads.in_chunks(
        lambda a, b: _comparisons(xs, ys, candidates, levels - 1, RULES.index(rule), float(delta), float(mu),
                                 float(kappa), float(power), sub, a, b, out),
        len(xs) - 1, ROWS)
    return out
