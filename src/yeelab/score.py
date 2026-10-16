"""Score ballots of voters in the plane: the score a voter gives each candidate.

A score ballot has `levels` scores, 0 to levels - 1. A voter at v gives the closest
candidate the top score and the farthest 0. Which score the others get depends on the
distances r_i = |v - c_i|, in one of seven ways (`rule`):

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
    D21     Janeček's D21: plus votes, minus votes and none, the scores 2, 0 and 1 of
            levels = 3, or without minus votes plus votes and none, 1 and 0 of
            levels = 2. votes = (max_positive, max_negative, negative_after), VOTES by
            default. The voter starts from RANGE's ballot of three levels, at the
            power `power`: a plus vote to the candidates it gives the top score, the
            closest first, at most max_positive of them, and a minus vote to those it
            gives 0, the farthest first, at most max_negative, but only after
            negative_after plus votes or more. If that gives no minus vote, and RANGE's
            approval ballot (two levels) gives more plus votes, at most max_positive,
            the voter takes that instead: a plus vote to those it approves and a minus
            vote, on the same terms, to those it does not. Without minus votes that is
            always the ballot: RANGE's approval, of at most max_positive candidates.
            The closest always gets a plus vote, but the farthest a minus vote only if
            the voter gives one
    D21_ORDINAL  D21 by the order of distance alone: a plus vote to the max_positive
            closest candidates and a minus vote to the max_negative farthest of the
            others, every voter all of them; negative_after and the power do not count.
            With two levels it is the approval of the max_positive closest

Nothing changes when all distances are scaled. With two levels every rule is an approval
ballot: the voter approves the candidates with the top score, from the closest alone to
all but the farthest; D21's, without minus votes, is RANGE's of at most max_positive
candidates, and D21_ORDINAL's that of the max_positive closest. RANGE approves those
closer than halfway between the closest and the farthest, AVG those closer than the mean
distance (at it, the half is rounded to 0: not approved), DHONDT, whatever delta, those
above the largest gap (ties: the first) and HYBRID, whatever delta, those above the
largest gap among the candidates closer than halfway, the gap from the last of them to
halfway included (ties: the first). CLUSTER approves those RANGE approves, unless that
cut splits a cluster and another gap costs less. AVG's is the best ballot of a voter
with the utility -r_i who takes every pair of candidates to be as likely to tie (Weber):
approving c is worth sum_e (r_e - r_c), which is positive for these. For three
candidates RANGE, AVG and DHONDT are the same ballot, and HYBRID approves the middle one
only closer than a quarter of the way, r_(2) < (3 r_(1) + r_(3)) / 4; for more
candidates all four differ. The gaps are those of the distances, not of their squares: a
voter at the squared distances 0, 0.35, 0.45, 0.55 and 1 has the gaps 0.59, 0.08, 0.07
and 0.26, so DHONDT approves only the candidate the voter stands on.

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
half a score. A CLUSTER score changes where two choices of scores cost the same. A D21
vote changes where a RANGE score of three or two levels does, and where two candidates
with the same score are as far, if max_positive or max_negative cuts between them;
a D21_ORDINAL vote only on a bisector, like a ranking. Unlike the bisectors the others
are curves, so the mean score is not a sum over polygons. `ballot_table` gives, for a
grid of rectangles, the distinct ballots in each rectangle and the part of its voters
with each, and `tabulate` every mean over them:
the points below the top score of each candidate (`unscored`), the share of the voters
who give each candidate each score (`graded`, what its median reads), the strict
comparisons of each pair (`comparisons`), the points each candidate gets from the voters
who rank another candidate above it (`given`, what burying it takes away) and the points
below the top score from the voters who rank it above another but neither first
(`raised`, what compromising on it adds). margin/shares.py weighs them with the voters of
each rectangle.
"""

import math
from typing import Literal, NamedTuple

import numpy as np
from numba import njit

from yeelab import threads

RANGE = "range"
DHONDT = "dhondt"
AVG = "avg"
HYBRID = "hybrid"
CLUSTER = "cluster"
D21 = "d21"
D21_ORDINAL = "d21_ordinal"
Rule = Literal["range", "dhondt", "avg", "hybrid", "cluster", "d21", "d21_ordinal"]
RULES: tuple[Rule, ...] = (RANGE, DHONDT, AVG, HYBRID, CLUSTER, D21, D21_ORDINAL)
BITS = 4  # of each candidate in the code of a ballot: its points below the top score
MASK = (1 << BITS) - 1
MAX_CANDIDATES = 63 // BITS  # a ballot is a code in an int64
MAX_LEVELS = MASK + 1  # the points below the top score fit the bits of a candidate
DELTA = 0.8  # of DHONDT and HYBRID: each step to the largest g_k / (d_k + DELTA)
MU, KAPPA = 0.1, 2.0  # of CLUSTER: the cost of a split, and how much farther a cluster's neighbours are
POWER = 1.0  # of RANGE and D21: the part of the way to this power; 1 is in proportion to the distance
# of D21 and D21_ORDINAL: at most 2 plus votes and 1 minus vote, the minus vote only after
# 2 plus votes
VOTES = (2, 1, 2)  # (max_positive, max_negative, negative_after)
ROWS = 8  # grid rows per task of the compiled loop (threads.py)


def distances(points, candidates) -> np.ndarray:
    """r[..., c] = |p - c| for points (..., 2) and candidates (C, 2)."""
    points = np.asarray(points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    return np.sqrt(((points[..., None, :] - candidates) ** 2).sum(axis=-1))


def _check(rule, delta, mu=MU, kappa=KAPPA, power=POWER, votes=VOTES, levels=None):
    """ValueError unless the parameters make a ballot; for D21 and D21_ORDINAL levels
    must be 3 with minus votes and 2 without. negative_after may be above max_positive
    only without minus votes, where it does not count."""
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
    if (len(votes) != 3 or not all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in votes)
            or not (votes[0] >= 1 and votes[1] >= 0 and votes[2] >= 0)
            or (votes[1] and votes[2] > votes[0])):
        raise ValueError("votes must be whole numbers (max_positive, max_negative, negative_after) with "
                         f"max_positive >= 1, max_negative >= 0 and negative_after from 0 to max_positive, "
                         f"got {votes!r}")
    if rule in (D21, D21_ORDINAL) and levels is not None and levels != (3 if votes[1] else 2):
        raise ValueError(f"a D21 ballot has {3 if votes[1] else 2} levels with {votes[1]} minus votes, "
                         f"got {levels}")


def scored(points, candidates, levels: int, rule: Rule = RANGE, delta: float = DELTA, *, mu: float = MU,
           kappa: float = KAPPA, power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """The score a voter at each of the points (..., 2) gives each candidate, int
    (..., C) from 0 to levels - 1: the ballot from its definition, one voter at a time.
    delta is the divisor of DHONDT and HYBRID, mu and kappa the cost of a split and the
    cohesion of CLUSTER, power that of RANGE and D21, votes the numbers of votes of D21
    and D21_ORDINAL; the other rules do not use them."""
    return from_distances(distances(points, candidates), levels, rule, delta, mu=mu, kappa=kappa, power=power,
                          votes=votes)


def from_distances(r, levels: int, rule: Rule = RANGE, delta: float = DELTA, *, mu: float = MU,
                   kappa: float = KAPPA, power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """scored() of voters at the distances r (..., C) from the candidates, wherever they
    are: the ballot depends on nothing else. docs/ballots.py gives it the distances of
    real voters."""
    _check(rule, delta, mu, kappa, power, votes, levels)
    r = np.asarray(r, dtype=np.float64)
    if rule == CLUSTER:
        return levels - 1 - _cluster_below(r, levels - 1, mu, kappa)
    if rule == DHONDT:
        return levels - 1 - _dhondt_below(r, levels - 1, delta)
    if rule == HYBRID:
        return levels - 1 - _hybrid_below(r, levels - 1, delta)
    if rule == AVG:
        return np.rint(_avg_part(r) * (levels - 1)).astype(np.int64)
    if rule == D21:
        return levels - 1 - _d21_below(r, levels - 1, power, votes)
    if rule == D21_ORDINAL:
        n = r.shape[-1]
        up = np.full(r.shape[:-1], min(votes[0], n))
        return levels - 1 - _votes_below(r, levels - 1, up, np.minimum(votes[1], n - up))
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


def _d21_below(r, top, power, votes):
    """Points below the top score `top` of each candidate on a D21 ballot with the
    numbers of votes `votes`, int (..., C), for the distances r (..., C): 0 for a plus
    vote, 1 for none and 2 (top) for a minus vote. The scores of RANGE fall with the
    distance, so its plus votes go to a start of the order of distance and its minus
    votes to an end of it: only how many of each there are counts."""
    plus, minus, after = votes
    far = r.max(axis=-1, keepdims=True)
    span = far - r.min(axis=-1, keepdims=True)
    part = np.divide(far - r, span, out=np.ones_like(r), where=span > 0) ** power
    three, two = np.rint(part * 2), np.rint(part)  # RANGE's ballots of three and two levels
    up = np.minimum((three == 2).sum(axis=-1), plus)
    down = np.where(up >= after, np.minimum((three == 0).sum(axis=-1), minus), 0)
    approved = np.minimum((two == 1).sum(axis=-1), plus)
    switch = (down == 0) & (approved != up)  # no minus vote, and approval gives more plus votes
    down = np.where(switch, np.where(approved >= after, np.minimum((two == 0).sum(axis=-1), minus), 0), down)
    return _votes_below(r, top, np.where(switch, approved, up), down)


def _votes_below(r, top, up, down):
    """Points below the top score `top` of each candidate, int (..., C), for the
    distances r (..., C) of a voter who gives a plus vote to the `up` (...) closest
    candidates and a minus vote to the `down` (...) farthest: 0, 2 (top) and 1 for none
    (ties: the lowest index the closer, as in by_distance)."""
    n = r.shape[-1]
    position = np.argsort(np.argsort(r, axis=-1, kind="stable"), axis=-1, kind="stable")  # 0 the closest
    return np.where(position < up[..., None], 0, np.where(position >= n - down[..., None], top, 1))


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
def _ballot(x, y, candidates, top, rule, delta, mu, kappa, power, votes, r, order, steps, work, back):
    """scored() of the voter at (x, y) as a code: BITS bits per candidate, holding the
    points it is below the top score `top`. `rule` is the index of the rule in RULES,
    delta the divisor of DHONDT and HYBRID, mu and kappa those of CLUSTER, power that of
    RANGE and D21, votes the numbers of votes of D21 and D21_ORDINAL; r, order and
    steps (C,), work and back (those of _cluster) are scratch."""
    n = candidates.shape[0]
    code = 0
    if rule == 5 or rule == 6:  # D21 as _d21_below, D21_ORDINAL by the order alone
        by_distance(x, y, candidates, r, order)
        plus, minus, after = votes
        if rule == 6:
            up = min(plus, n)
            down = min(minus, n - up)
        else:
            near, far = r[order[0]], r[order[n - 1]]
            top3 = none3 = top2 = none2 = 0  # the top scores and the 0s of RANGE's three and two levels
            for k in range(n):
                part = (far - r[order[k]]) / (far - near) if far > near else 1.0
                if power != 1.0:
                    part = part ** power
                three = np.rint(part * 2)
                if three == 2:
                    top3 += 1
                elif three == 0:
                    none3 += 1
                if np.rint(part) == 1:
                    top2 += 1
                else:
                    none2 += 1
            up, approved = min(top3, plus), min(top2, plus)
            down = min(none3, minus) if up >= after else 0
            if down == 0 and approved != up:  # no minus vote, and approval gives more plus votes
                up, down = approved, (min(none2, minus) if approved >= after else 0)
        for k in range(n):
            code |= (0 if k < up else top if k >= n - down else 1) << (BITS * order[k])
        return code
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


@njit(inline="always", error_model="numpy")
def _order_code(r, order):
    """The candidates in order of the distances r (C,), closest first, into order (ties:
    the lowest index first, like by_distance), and that order as a code of BITS bits per
    position."""
    code = 0
    for c in range(r.size):  # insertion sort
        k = c
        while k > 0 and r[order[k - 1]] > r[c]:
            order[k] = order[k - 1]
            k -= 1
        order[k] = c
    for k in range(r.size):
        code |= order[k] << (BITS * k)
    return code


class BallotTable(NamedTuple):
    """The ballots of the voters of a grid of rectangles (ballot_table): for each
    rectangle k, in the order of its row and then its column, the distinct ballots of its
    voters, start[k] to start[k + 1] of these arrays.

        codes    the ballot as _ballot gives it: BITS bits per candidate, its points below
                 the top score
        orders   the candidates in order of distance: BITS bits per position, the closest
                 lowest (_order_code); only in a table by_order, else 0
        weights  the part of the rectangle's voters with that ballot, adding up to 1

    Every score share is a sum over the table (tabulate), so one pass over the grid,
    which is most of the work, gives them all."""

    shape: tuple[int, int]  # rectangles along x and y
    start: np.ndarray
    codes: np.ndarray
    orders: np.ndarray
    weights: np.ndarray
    candidates: int
    top: int  # levels - 1
    by_order: bool


@njit(cache=True, nogil=True, error_model="numpy")
def _table_rows(xs, ys, candidates, top, rule, delta, mu, kappa, power, votes, sub, by_order, start, stop):
    """The ballot table of the cells [xs[i], xs[i + 1]] with start <= i < stop: the
    number of distinct ballots of every cell (rows, Y), and their codes, orders and
    weights, cell after cell. A cell holds one ballot if its four corners have the same
    one, and with by_order also the same order of distance: the bisectors are straight,
    so then none crosses it. The others hold the distinct ballots of sub x sub points,
    each with how often it is there."""
    n, ny = candidates.shape[0], ys.size - 1
    r, order, steps = np.empty(n), np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    work, back = np.empty((4, max(n, top + 1))), np.empty((n, top + 1), dtype=np.int64)
    kinds = np.zeros((stop - start, ny), dtype=np.int64)
    size = 2 * (stop - start) * ny  # grows as needed
    codes, orders, weights = np.empty(size, dtype=np.int64), np.empty(size, dtype=np.int64), np.empty(size)
    used = 0
    point_codes, point_orders = np.empty(sub * sub, dtype=np.int64), np.empty(sub * sub, dtype=np.int64)
    counts = np.empty(sub * sub, dtype=np.int64)
    left, right = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    left_order, right_order = np.empty(ys.size, dtype=np.int64), np.empty(ys.size, dtype=np.int64)
    for j in range(ys.size):
        left[j] = _ballot(xs[start], ys[j], candidates, top, rule, delta, mu, kappa, power, votes, r, order,
                          steps, work, back)
        left_order[j] = _order_code(r, order) if by_order else 0
    for i in range(start, stop):
        for j in range(ys.size):
            right[j] = _ballot(xs[i + 1], ys[j], candidates, top, rule, delta, mu, kappa, power, votes,
                               r, order, steps, work, back)
            right_order[j] = _order_code(r, order) if by_order else 0
        for j in range(ny):
            code, ranked = left[j], left_order[j]
            whole = code == left[j + 1] and code == right[j] and code == right[j + 1]
            if whole and by_order:
                whole = ranked == left_order[j + 1] and ranked == right_order[j] and ranked == right_order[j + 1]
            if whole:
                found, point_codes[0], point_orders[0], counts[0] = 1, code, ranked, sub * sub
            else:
                found = 0
                for a in range(sub):
                    x = xs[i] + (a + 0.5) / sub * (xs[i + 1] - xs[i])
                    for b in range(sub):
                        y = ys[j] + (b + 0.5) / sub * (ys[j + 1] - ys[j])
                        code = _ballot(x, y, candidates, top, rule, delta, mu, kappa, power, votes, r, order,
                                       steps, work, back)
                        ranked = _order_code(r, order) if by_order else 0
                        k = 0
                        while k < found and (point_codes[k] != code or point_orders[k] != ranked):
                            k += 1
                        if k == found:
                            point_codes[k], point_orders[k], counts[k] = code, ranked, 0
                            found += 1
                        counts[k] += 1
            if used + found > size:  # twice as much room
                size = 2 * (used + found)
                codes, orders, weights = (np.concatenate((codes[:used], np.empty(size - used, dtype=np.int64))),
                                          np.concatenate((orders[:used], np.empty(size - used, dtype=np.int64))),
                                          np.concatenate((weights[:used], np.empty(size - used))))
            for k in range(found):
                codes[used], orders[used] = point_codes[k], point_orders[k]
                weights[used] = counts[k] / (sub * sub)
                used += 1
            kinds[i - start, j] = found
        left, right = right, left
        left_order, right_order = right_order, left_order
    return kinds, codes[:used], orders[:used], weights[:used]


def ballot_table(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
                 delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
                 power: float = POWER, votes: tuple[int, int, int] = VOTES, by_order: bool = False) -> BallotTable:
    """The distinct ballots of the voters of every rectangle [xs[i], xs[i + 1]] x
    [ys[j], ys[j + 1]], with the part of them that has each, for X + 1 and Y + 1
    increasing grid lines; delta, mu, kappa, power and votes as in scored(). A rectangle
    whose four corners have the same ballot counts as all of that ballot, and with by_order
    only if they also rank the candidates in the same order; the others, which a border
    crosses, hold the ballots of sub x sub points. given() and raised() need by_order:
    who ranks whom above whom is all they read. Chunks of the rows run in parallel
    threads."""
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    if not 2 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"score ballots need 2 to {MAX_CANDIDATES} candidates, got {len(candidates)}")
    if not 2 <= levels <= MAX_LEVELS:
        raise ValueError(f"score ballots need 2 to {MAX_LEVELS} levels, got {levels}")
    _check(rule, delta, mu, kappa, power, votes, levels)
    votes = tuple(int(v) for v in votes)
    parts = {}

    def rows(a, b):
        parts[a] = _table_rows(xs, ys, candidates, levels - 1, RULES.index(rule), float(delta), float(mu),
                               float(kappa), float(power), votes, sub, by_order, a, b)

    threads.in_chunks(rows, len(xs) - 1, ROWS)
    chunks = [parts[a] for a in sorted(parts)]
    kinds = np.concatenate([chunk[0] for chunk in chunks])
    start = np.concatenate([[0], np.cumsum(kinds.ravel())])
    codes, orders, weights = (np.concatenate([chunk[k] for chunk in chunks]) for k in (1, 2, 3))
    for array in (start, codes, orders, weights):
        array.setflags(write=False)
    return BallotTable(kinds.shape, start, codes, orders, weights, len(candidates), levels - 1, by_order)


# what tabulate() sums: the points below the top score of each candidate (C, X, Y), the
# strict comparisons of each pair (C, C, X, Y), what burying takes back and what
# compromising adds (both C, C, X, Y; given() and raised()), and the share of the voters
# who give each candidate each score (C, levels, X, Y; graded())
UNSCORED, COMPARISONS, GIVEN, RAISED, GRADES = range(5)
CELLS = 4096  # rectangles per task of the compiled sums (threads.py)


@njit(cache=True, nogil=True, error_model="numpy")
def _tabulate(start, codes, orders, weights, n, top, kind, tie_weight, first, stop, out):
    """tabulate() of the rectangles first <= k < stop into out (C, C * C or C * levels,
    X * Y)."""
    for k in range(first, stop):
        for row in range(out.shape[0]):
            out[row, k] = 0.0
        for t in range(start[k], start[k + 1]):
            code, ranked, weight = codes[t], orders[t], weights[t]
            if kind == UNSCORED:
                for c in range(n):
                    out[c, k] += weight * ((code >> (BITS * c)) & MASK)
            elif kind == GRADES:  # row c * levels + score
                for c in range(n):
                    out[c * (top + 1) + top - ((code >> (BITS * c)) & MASK), k] += weight
            elif kind == COMPARISONS:
                for c in range(n):
                    below_c = (code >> (BITS * c)) & MASK
                    for e in range(c + 1, n):
                        below_e = (code >> (BITS * e)) & MASK
                        if below_c != below_e:
                            difference = weight if below_c < below_e else -weight
                            out[c * n + e, k] += difference
                            out[e * n + c, k] -= difference
                        elif tie_weight:
                            position_c = position_e = 0
                            for position in range(n):
                                candidate = (ranked >> (BITS * position)) & MASK
                                if candidate == c:
                                    position_c = position
                                elif candidate == e:
                                    position_e = position
                            difference = tie_weight * weight * (1 if position_c < position_e else -1)
                            out[c * n + e, k] += difference
                            out[e * n + c, k] -= difference
            else:
                for above in range(1 if kind == RAISED else 0, n):  # the candidates in order
                    c = (ranked >> (BITS * above)) & MASK
                    for below in range(above + 1, n):
                        e = (ranked >> (BITS * below)) & MASK
                        if kind == RAISED:  # c, not the first choice, above e: the points c lacks
                            out[c * n + e, k] += weight * ((code >> (BITS * c)) & MASK)
                        else:  # c above e: the points e gets
                            out[e * n + c, k] += weight * (top - ((code >> (BITS * e)) & MASK))


def tabulate(table: BallotTable, kind: int, tie_weight: float = 0.0) -> np.ndarray:
    """The mean over the voters of every rectangle of the table (float32): for UNSCORED
    the points below the top score of each candidate (C, X, Y), for GRADES the share of
    them who give each candidate each score (C, levels, X, Y), for COMPARISONS the share
    strictly scoring c above e minus e above c (C, C, X, Y), for GIVEN given() and for
    RAISED raised(), which need a table by_order. Chunks of the rectangles run in
    parallel threads."""
    if (kind in (GIVEN, RAISED) or tie_weight > 0) and not table.by_order:
        raise ValueError("ordinal score comparisons need a ballot table by order")
    n = table.candidates
    cells = table.shape[0] * table.shape[1]
    rows = (n,) if kind == UNSCORED else (n, table.top + 1) if kind == GRADES else (n, n)
    out = np.empty((math.prod(rows), cells), dtype=np.float32)
    threads.in_chunks(lambda a, b: _tabulate(table.start, table.codes, table.orders, table.weights, n, table.top,
                                             kind, tie_weight, a, b, out), cells, CELLS)
    return out.reshape(rows + table.shape)


@njit(cache=True, nogil=True, error_model="numpy")
def _moved(start, codes, orders, weights, n, top, a, b, bury, compromise, tie_weight, first, stop, out):
    """moved_comparisons() of the rectangles first <= k < stop into out (C * C, X * Y)."""
    scores, moved = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    positions, moved_positions = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    kinds = (bury * (1 - compromise), (1 - bury) * compromise, bury * compromise)  # bury, compromise, both
    for k in range(first, stop):
        for row in range(out.shape[0]):
            out[row, k] = 0.0
        for t in range(start[k], start[k + 1]):
            code, ranked, weight = codes[t], orders[t], weights[t]
            position_a = position_b = 0
            for position in range(n):
                c = (ranked >> (BITS * position)) & MASK
                scores[c] = top - ((code >> (BITS * c)) & MASK)
                positions[c] = position
                if c == a:
                    position_a = position
                elif c == b:
                    position_b = position
            # a voter with a frontrunner first gives it the top score already: compromising
            # changes nothing there
            upper, lower = (a, b) if position_a < position_b else (b, a)
            for kind in range(3):
                share = kinds[kind]
                if share == 0:
                    continue
                for c in range(n):
                    moved[c] = scores[c]
                    moved_positions[c] = positions[c]
                if kind != 1:  # buries: the frontrunner ranked lower gets 0
                    moved[lower] = 0
                    for c in range(n):
                        if positions[c] > positions[lower]:
                            moved_positions[c] -= 1
                    moved_positions[lower] = n - 1
                if kind != 0:  # compromises: the other the top score
                    moved[upper] = top
                    if positions[upper] > 0:
                        for c in range(n):
                            if moved_positions[c] < positions[upper]:
                                moved_positions[c] += 1
                        moved_positions[upper] = 0
                for c in (a, b):  # the pairs with a or b, (a, b) once
                    for e in range(n):
                        if e == c or (c == b and e == a):
                            continue
                        before = (1 if scores[c] > scores[e] else -1 if scores[c] < scores[e]
                                  else tie_weight * (1 if positions[c] < positions[e] else -1))
                        after = (1 if moved[c] > moved[e] else -1 if moved[c] < moved[e]
                                 else tie_weight * (1 if moved_positions[c] < moved_positions[e] else -1))
                        if after != before:
                            change = weight * share * (after - before)
                            out[c * n + e, k] += change
                            out[e * n + c, k] -= change


def moved_comparisons(table: BallotTable, a: int, b: int, bury: float, compromise: float,
                      tie_weight: float = 0.0) -> np.ndarray:
    """How much the score comparisons of a table by order move, (C, C, X, Y) float32,
    where `bury` of the voters bury and `compromise` of them compromise against the
    frontrunners a and b, each whatever the other does (blocks.Burial, Compromise): a
    burying voter gives the frontrunner they rank lower 0, a compromising one gives the one
    they rank higher the top score if neither is their first choice, and bury * compromise
    of the voters do both. Only the comparisons of a or b with another candidate move. The
    comparisons do not add up over the strategies like the scores: a voter who does both
    compares the two frontrunners as neither change alone would, so each kind of voter is
    counted on its own."""
    if not table.by_order:
        raise ValueError("the comparisons of strategic voters need a ballot table by order")
    n = table.candidates
    cells = table.shape[0] * table.shape[1]
    out = np.empty((n * n, cells), dtype=np.float32)
    threads.in_chunks(lambda first, stop: _moved(table.start, table.codes, table.orders, table.weights, n, table.top,
                                             a, b, float(bury), float(compromise), float(tie_weight), first, stop, out),
                      cells, CELLS)
    return out.reshape((n, n) + table.shape)


@njit(cache=True, nogil=True, error_model="numpy")
def _moved_grades(start, codes, orders, weights, n, top, a, b, first, stop, out):
    """moved_grades() of the rectangles first <= k < stop into out (2 * 2 * levels, X * Y)."""
    levels = top + 1
    for k in range(first, stop):
        for row in range(out.shape[0]):
            out[row, k] = 0.0
        for t in range(start[k], start[k + 1]):
            code, ranked, weight = codes[t], orders[t], weights[t]
            position_a = position_b = 0
            for position in range(n):
                c = (ranked >> (BITS * position)) & MASK
                if c == a:
                    position_a = position
                elif c == b:
                    position_b = position
            # the frontrunner ranked lower is buried, the one ranked higher compromised on
            # if neither is first; side 0 is a, 1 is b
            lower, upper = (b, a) if position_a < position_b else (a, b)
            side = 1 if lower == b else 0
            out[side * levels + top - ((code >> (BITS * lower)) & MASK), k] += weight
            if min(position_a, position_b) > 0:
                out[(3 - side) * levels + top - ((code >> (BITS * upper)) & MASK), k] += weight


def moved_grades(table: BallotTable, a: int, b: int) -> np.ndarray:
    """The share of the voters of every rectangle of a table by order at each score of
    the frontrunners a and b, of those that a strategy against them moves, (2, 2,
    levels, X, Y) float32: [0, 0] the voters who rank b above a, at each score of a,
    which burying (blocks.Burial) turns to 0, and [0, 1] those who rank a above b, at
    each score of b; [1, 0] the voters who rank a above b, neither first, at each score
    of a, which compromising (blocks.Compromise) turns to the top score, and [1, 1]
    likewise for b. A voter's scores of the two are moved by one strategy each, so the
    shares of every score of burying and compromising voters add up (graded())."""
    if not table.by_order:
        raise ValueError("the shares of every score of strategic voters need a ballot table by order")
    levels = table.top + 1
    cells = table.shape[0] * table.shape[1]
    out = np.empty((4 * levels, cells), dtype=np.float32)
    threads.in_chunks(lambda first, stop: _moved_grades(table.start, table.codes, table.orders, table.weights,
                                                        table.candidates, table.top, a, b, first, stop, out),
                      cells, CELLS)
    return out.reshape((2, 2, levels) + table.shape)


def unscored(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
             delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
             power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """short[c, i, j] = mean points below the top score that the voters of the rectangle
    [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]] give candidate c, from 0 to levels - 1, shape
    (C, X, Y) for X + 1 and Y + 1 increasing grid lines; delta, mu, kappa, power and votes
    as in scored(). A rectangle whose four corners have the same ballot counts as all of
    that ballot; in the others, which a border crosses, the ballots at sub x sub points are
    averaged (ballot_table, tabulate)."""
    table = ballot_table(xs, ys, candidates, levels, sub, rule, delta, mu=mu, kappa=kappa, power=power, votes=votes)
    return tabulate(table, UNSCORED)


def graded(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
           delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
           power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """short[c, k, i, j] = share of the voters of the rectangle [xs[i], xs[i + 1]] x
    [ys[j], ys[j + 1]] who give candidate c the score k, from 0 to levels - 1, shape
    (C, levels, X, Y); the shares of a candidate add up to 1 in every rectangle.
    Rectangles as in unscored(), whose mean is sum_k (levels - 1 - k) short[c, k]."""
    table = ballot_table(xs, ys, candidates, levels, sub, rule, delta, mu=mu, kappa=kappa, power=power, votes=votes)
    return tabulate(table, GRADES)


def comparisons(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
                delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
                power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """short[c, e, i, j] = share strictly scoring c above e minus e above c among
    voters in each rectangle [xs[i], xs[i + 1]] x [ys[j], ys[j + 1]]. Voters who give
    the pair equal scores contribute 0. Rectangles as in unscored(); output shape is
    (C, C, X, Y)."""
    table = ballot_table(xs, ys, candidates, levels, sub, rule, delta, mu=mu, kappa=kappa, power=power, votes=votes)
    return tabulate(table, COMPARISONS)


def given(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
          delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
          power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """short[c, e, i, j] = mean points that the voters of the rectangle [xs[i], xs[i + 1]] x
    [ys[j], ys[j + 1]] who rank e above c give c, from 0 to levels - 1 (0 for e = c), shape
    (C, C, X, Y): what c loses if those voters bury it, giving it 0. A voter ranks e above c
    if e is closer (ties: the lower index first). Rectangles as in unscored(), but one
    holds one ballot only if its corners also have the same order of distance (by_order)."""
    table = ballot_table(xs, ys, candidates, levels, sub, rule, delta, mu=mu, kappa=kappa, power=power,
                         votes=votes, by_order=True)
    return tabulate(table, GIVEN)


def raised(xs, ys, candidates, levels: int, sub: int, rule: Rule = RANGE,
           delta: float = DELTA, *, mu: float = MU, kappa: float = KAPPA,
           power: float = POWER, votes: tuple[int, int, int] = VOTES) -> np.ndarray:
    """short[c, e, i, j] = mean points below the top score that the voters of the rectangle
    who rank c above e, but neither first, give c, from 0 to levels - 1 (0 for e = c), shape
    (C, C, X, Y): what c gains if those voters compromise on it, giving it the top score.
    Rectangles as in given()."""
    table = ballot_table(xs, ys, candidates, levels, sub, rule, delta, mu=mu, kappa=kappa, power=power,
                         votes=votes, by_order=True)
    return tabulate(table, RAISED)
