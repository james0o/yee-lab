"""Plain Python references of the methods, straight from their definitions, point by
point, and random profiles: what the built methods are checked against
(test_build.py, test_strategic.py, test_judgment.py)."""

from functools import cmp_to_key
from itertools import permutations

import numpy as np

from yeelab.build import Voters

TOLERANCE = 1e-6  # winners are compared where the margin is above (float32 profiles)


def voters_of(rankings, probs):
    """Every share a method may need, from a complete profile."""
    first = probs @ np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    position = np.argsort(rankings, axis=1)
    prefers = position[:, :, None] < position[:, None, :]
    d = probs @ prefers.reshape(len(rankings), -1)
    d = d.reshape(*probs.shape[:-1], rankings.shape[1], rankings.shape[1])
    return Voters(first, d, rankings, probs)


def ranking_weights(ballot, rankings, alive):
    """W[r, c] = points ranking r gives c: weight(position of c among the candidates in
    alive (C,), number of them), 0 for the others."""
    weights = np.zeros(rankings.shape)
    remaining = [[c for c in ranking if alive[c]] for ranking in rankings]
    for r, order in enumerate(remaining):
        for k, c in enumerate(order):
            weights[r, c] = ballot.weight(k, len(order))
    return weights


def elimination_reference(ballot, how, rankings, probs, until=1, weigh=ranking_weights):
    """Eliminate(Tally(ballot), how, until) point by point, straight from its
    definition: the candidates left (..., C) and the margin. With the `weigh` of weak
    orders (test_strategic.py), rankings are their levels."""
    n = rankings.shape[1]
    weights = {}  # by the remaining candidates
    left = np.empty(probs.shape[:-1] + (n,), dtype=bool)
    margin = np.empty(probs.shape[:-1])
    for point in np.ndindex(*probs.shape[:-1]):
        alive, gaps = np.ones(n, dtype=bool), [np.inf]
        while alive.sum() > until:
            key = alive.tobytes()
            if key not in weights:
                weights[key] = weigh(ballot, rankings, alive)
            remaining = np.flatnonzero(alive)
            totals = (probs[point].astype(np.float64) @ weights[key])[remaining]
            highest = np.argsort(-totals, kind="stable")  # ties: the lowest index first
            if how == "all":
                gaps.append(totals[highest[until - 1]] - totals[highest[until]])
                alive[remaining[highest[until:]]] = False
            elif how == "min":
                order = np.argsort(totals, kind="stable")  # ties: the lowest index goes
                gaps.append(totals[order[1]] - totals[order[0]])
                alive[remaining[order[0]]] = False
            else:
                mean = totals.mean()
                gaps.append(np.abs(totals - mean).min())
                out = (totals <= mean) | (totals == totals.min())  # min: in case of rounding
                if out.all():  # all totals equal: the lowest index stays
                    out[0] = False
                if (~out).sum() < until:  # too few would stay: the `until` highest do
                    gaps.append(totals[highest[until - 1]] - totals[highest[until]])
                    out[:] = True
                    out[highest[:until]] = False
                alive[remaining[out]] = False
        left[point], margin[point] = alive, min(gaps)
    return left, margin


def compositions(total):
    """Every way to write total as an ordered sum of whole numbers of at least 1."""
    if total == 0:
        yield ()
    for first in range(1, total + 1):
        for rest in compositions(total - first):
            yield (first, *rest)


def schedules_reference(ballot, rankings, probs, weigh=ranking_weights):
    """Schedules(Tally(ballot)) point by point, from its definition: for every way to
    split the C - 1 eliminations into rounds, each round drops that many of the lowest
    totals of the remaining candidates (ties: the lowest index stays), tallied again
    among them. The winners of all schedules (..., C) and the smallest gap of any round:
    the lowest total kept minus the highest dropped. With the `weigh` of weak orders
    (test_strategic.py), rankings are their levels."""
    n = rankings.shape[1]
    schedules = list(compositions(n - 1))
    assert len(schedules) == 2 ** (n - 2)
    weights = {}  # by the remaining candidates
    chosen = np.zeros(probs.shape[:-1] + (n,), dtype=bool)
    margin = np.empty(probs.shape[:-1])
    for point in np.ndindex(*probs.shape[:-1]):
        shares, gaps, tallied = probs[point].astype(np.float64), [np.inf], {}
        for schedule in schedules:
            alive = np.ones(n, dtype=bool)
            for drop in schedule:
                key = alive.tobytes()
                if key not in weights:
                    weights[key] = weigh(ballot, rankings, alive)
                if key not in tallied:  # the schedules share their first rounds
                    tallied[key] = shares @ weights[key]
                remaining = np.flatnonzero(alive)
                totals = tallied[key][remaining]
                highest = np.argsort(-totals, kind="stable")  # ties: the lowest index first
                keep = len(remaining) - drop
                gaps.append(totals[highest[keep - 1]] - totals[highest[keep]])
                alive[remaining[highest[keep:]]] = False
            chosen[point + (np.flatnonzero(alive)[0],)] = True
        margin[point] = min(gaps)
    return chosen, margin


def levels_of(rankings):
    """The rankings (R, C), best first, as weak orders: L[r, c] = position of c."""
    return np.argsort(rankings, axis=1)


def acquiescing(levels):
    """A[b, s]: the weak order b (levels L[b, c], lower better, levels_of for rankings)
    ranks no candidate outside the set s (a bit mask, 2^C of them) above a candidate in
    it, (B, 2^C)."""
    n = levels.shape[1]
    inside = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)  # [s, c]
    worst_inside = np.where(inside, levels[:, None, :], -1).max(axis=-1)
    best_outside = np.where(inside, n, levels[:, None, :]).min(axis=-1)
    return worst_inside <= best_outside


def dac_reference(levels, probs):
    """DescendingCoalitions point by point, from its definition: every set scores the
    share of the voters who acquiesce to it, and from the highest score down (ties: the
    lowest mask first) each set makes the candidates outside it ineligible, unless that
    would leave no one. The margin is the smallest gap of a set that changes the
    eligible candidates E: its score minus the highest score of a set that would change
    them too, but keep none of the candidates it keeps. levels are the weak orders of
    the voters (levels_of for rankings)."""
    n = levels.shape[1]
    masks = np.arange(1 << n)
    scores = probs.astype(np.float64) @ acquiescing(levels)
    winner = np.empty(probs.shape[:-1], dtype=np.int64)
    margin = np.empty(probs.shape[:-1])
    for point in np.ndindex(*probs.shape[:-1]):
        score, eligible, gap = scores[point], (1 << n) - 1, np.inf
        for s in np.argsort(-score, kind="stable"):
            kept = int(s) & eligible
            if kept in (0, eligible):  # no one would be left, or no one goes
                continue
            cuts = masks & eligible
            rivals = (cuts != 0) & (cuts != eligible) & ((cuts & kept) == 0)
            gap = min(gap, score[s] - score[rivals].max())
            eligible = kept
        winner[point], margin[point] = eligible.bit_length() - 1, gap
    return winner, margin


def random_profiles(n, count, seed=0):
    """Every ranking of n candidates (n!, n) and random shares of them (count, n!), most
    of the voters in a few rankings."""
    rankings = np.array(list(permutations(range(n))))
    rng = np.random.default_rng(seed + n)
    return rankings, rng.dirichlet(np.full(len(rankings), 0.3), size=count)


def majority_value(counts):
    """Balinski and Laraki's majority value of a candidate graded by explicit ballots,
    counts[k] of them with the grade k: the lower of the middle grades (the majority
    grade), taken out, and again on the rest, until none is left."""
    grades = sorted(grade for grade, count in enumerate(counts) for _ in range(int(count)))
    value = []
    while grades:
        value.append(grades.pop((len(grades) - 1) // 2))  # the lower middlemost
    return value


def judgment_order(counts):
    """The candidates of counts (C, L), every one graded by the same n ballots, in the
    order of majority judgment: by their majority values, the higher first at the first
    grade where they differ; equal values by the lower index."""
    values = [majority_value(row) for row in counts]
    return sorted(range(len(counts)), key=lambda c: values[c], reverse=True)  # stable: ties keep the index


def majority_totals_reference(grades, alive=None):
    """Tally(ballot, statistic="median") at one point straight from the definitions of
    build/judgment.py, with every cost of every threshold: grades (C, L) the shares (or
    counts) of each score, alive (C,). The order of the remaining candidates, and the
    totals: the first 0, each next one lower by the smallest gap of a candidate above it
    and one below, by one step of rounding where that is 0 and only the rule of the lower
    end puts it behind; -C for the others."""
    grades = np.asarray(grades, dtype=np.float64)
    n, levels = grades.shape
    alive = np.ones(n, dtype=bool) if alive is None else np.asarray(alive, dtype=bool)
    total = np.cumsum(grades, axis=-1)[:, -1:]  # one by one, as the kernel (sum adds pairwise)
    at_most = np.cumsum(grades[:, :-1], axis=-1)  # from the bottom
    above = np.cumsum(grades[:, :0:-1], axis=-1)[:, ::-1]  # from the top
    lower = above > at_most
    minority = np.where(lower, at_most, above) / total
    half = (above - at_most) / (2 * total)

    def duel(a, b):
        """(verdict, gap): who is ahead (1 a, -1 b, 0 a tie) and the smallest cost."""
        rows = []  # per threshold: (close, end, strength, side)
        for k in range(levels - 1):
            if lower[a, k] != lower[b, k]:
                rows.append((0.5, 1, abs(half[a, k] - half[b, k]), 1 if lower[a, k] else -1))
            else:
                ma, mb = minority[a, k], minority[b, k]
                ahead = mb - ma if lower[a, k] else ma - mb
                rows.append((max(ma, mb), int(lower[a, k]), abs(ma - mb), int(np.sign(ahead))))
        differing = [row for row in rows if row[3] != 0]
        if not differing:
            return 0, 0.0
        verdict = max(differing, key=lambda row: row[:2])[3]
        costs = []
        for k in rows:
            terms = [k[2] if k[3] == verdict else 0.0]
            terms += [min(j[2], j[0] - k[0]) for j in rows if j is not k and j[3] == verdict and j[:2] > k[:2]]
            costs.append(max(terms))
        return verdict, min(costs)

    verdict, gap = np.zeros((n, n), dtype=np.int64), np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            if alive[a] and alive[b]:
                v, g = duel(a, b)
                verdict[a, b], verdict[b, a], gap[a, b], gap[b, a] = v, -v, g, g
    order = sorted(np.flatnonzero(alive), key=cmp_to_key(lambda a, b: -verdict[a, b]))
    totals = np.full(n, -float(n))
    value = 0.0
    for place, c in enumerate(order):
        if place:
            step = min(gap[i, j] for i in order[:place] for j in order[place:])
            below = value - step
            if below == value and verdict[order[place - 1], c] > 0:
                below = np.nextafter(value, -np.inf)
            value = below
        totals[c] = value
    return order, totals
