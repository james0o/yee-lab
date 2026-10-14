"""The majority judgment of a score ballot, compiled: Tally(ballot, statistic="median")
(blocks.py) puts the candidates of every point in its order and gives Highest and
Eliminate totals in that order.

The scores 0 to levels - 1 of the ballot are the grades, and g[c, k] is the share of the
voters of a point who give candidate c the score k (Voters.graded). The majority grade of
c is its lower median: the smallest k with g[c, 0] + ... + g[c, k] >= 1/2.

Balinski and Laraki break a tie of majority grades by taking one majority grade out of
each tied candidate's grades and the median of the rest, again and again: the candidate
whose median first comes out higher is ahead. On n grades in order that takes them from
the middle out, the lower of the two middle ones first. Here the voters of a point are a
distribution, not n ballots, and one grade is no share of them; taking out a share t of
the voters is. The middle t of the voters, in the order of their grades, go, and the
medians of the rest are the grades at 1/2 - t/2 and 1/2 + t/2 of the way: the lower one
first, as the lower median. Two candidates are compared at the smallest t where one of
those two grades differs.

That is decided at the thresholds between the scores. Of the voters of c, the share
S(k) = g[c, k + 1] + ... gives more than k, and F(k) = 1 - S(k) at most k. Where a
majority gives more than k (S > F: k is below the majority grade), the lower grade drops
to k at t = S - F; elsewhere the upper grade rises above k at t = F - S. Two candidates
with different S(k) get different grades at that end from the smaller t of the two on,
from t = 0 if their majorities are on different sides of k (their majority grades
differ), and the one with the larger S(k) has the higher grade. So the threshold where
they differ first decides, the lower end first at the same t: the larger S(k) is ahead.
Two thresholds that disagree at the same t are only on borders, where the margin is 0.
Candidates with the same shares at every threshold have the same grades: they tie, and
Highest takes the lowest index.

For shares that are multiples of 1/n this is the procedure on n ballots, for every n:
there the lower end at t is its step n t and the upper end its step n t + 1. Other shares
are its limit as n grows. No number of voters is assumed. With two levels it is
approval voting, the most approvals ahead, by the margin of the mean.

A lexicographic order is no total of each candidate, so the totals of a point rank the
candidates instead: the first 0, each next one lower by the gap of its place, the
smallest gap of a candidate above it and one below. The gap of two candidates is

    min over thresholds k of the largest of
        |S_a(k) - S_b(k)|                                   if k agrees with the order
        min(|S_a(j) - S_b(j)|, (t_k - t_j) / 2)  for each threshold j before k that agrees

with t_k from where the two differ at k (thresholds where they do not differ count as
disagreeing, from where they would). It is 0 where their order changes and continuous
elsewhere. If no candidate's shares change by as much as the gap in all (the sum of |dg|
over its scores, the shares still adding up to 1), every S and F moves by less than half
of it and every term above by less than all of it, so the order stays. Where MJ's own
rule for the same t puts a candidate behind with a gap of 0, its total is one step of
rounding lower, so that Highest gives the candidate ahead.

The shares at most k are summed from the bottom and those above k from the top, and t is
compared through the smaller of the two, so that the few voters of a tail stay exact, as
in the part not given of the mean (blocks.Score).
"""

import numpy as np
from numba import njit

from yeelab import threads

CHUNK = 1024  # points per task of the compiled loop (threads.py)


@njit(cache=True, nogil=True, error_model="numpy")
def _sides(grades, lower, minority, half):
    """The thresholds of the shares grades (C, L) of each score: lower[c, k] where more of
    c's voters give it more than k than at most k, minority[c, k] the smaller of the two
    shares and half[c, k] = (more - at most) / 2, both as parts of c's own total. The
    shares at most k are summed from the bottom and the others from the top."""
    n, levels = grades.shape
    for c in range(n):
        total = 0.0
        for k in range(levels):
            total += grades[c, k]
        at_most = 0.0
        for k in range(levels - 1):
            at_most += grades[c, k]
            half[c, k] = at_most  # for now
        above = 0.0
        for k in range(levels - 2, -1, -1):
            above += grades[c, k + 1]
            at_most = half[c, k]
            lower[c, k] = above > at_most  # an even split is the upper end: the lower median
            minority[c, k] = (at_most if above > at_most else above) / total
            half[c, k] = (above - at_most) / (2 * total)


@njit(cache=True, nogil=True, error_model="numpy")
def _judge(grades, alive, out):
    """majority_totals for grades (P, C, L) and alive (P, C) into out (P, C).

    For each pair of remaining candidates a and b and each threshold k: close[k], where
    the two start to differ, as 1/2 - t/2 (the larger minority share; 1/2 where their
    majorities are on different sides, from t = 0), end[k] at which end (1 lower, 0
    upper), strength[k] = |S_a - S_b| and side[k] who is ahead there (1 a, -1 b, 0
    neither); then verdict[a, b], the side of the first threshold where they differ, and
    gap[a, b]. A threshold j comes before k from a larger close, or the lower end at the
    same one. (A function for a pair is several times slower in numba: the loops are
    here.)"""
    points, n, levels = grades.shape
    lower = np.empty((n, levels - 1), dtype=np.bool_)
    minority, half = np.empty((n, levels - 1)), np.empty((n, levels - 1))
    close, strength = np.empty(levels - 1), np.empty(levels - 1)
    end, side = np.empty(levels - 1, dtype=np.int64), np.empty(levels - 1, dtype=np.int64)
    verdict, gap = np.zeros((n, n), dtype=np.int64), np.zeros((n, n))
    order = np.empty(n, dtype=np.int64)
    for p in range(points):
        _sides(grades[p], lower, minority, half)
        for a in range(n):
            for b in range(a + 1, n):
                if not (alive[p, a] and alive[p, b]):
                    continue
                first = -1
                for k in range(levels - 1):
                    if lower[a, k] != lower[b, k]:  # different majority grades: from t = 0
                        close[k] = 0.5
                        end[k] = 1
                        strength[k] = abs(half[a, k] - half[b, k])
                        side[k] = 1 if lower[a, k] else -1
                    else:
                        ma = minority[a, k]
                        mb = minority[b, k]
                        close[k] = max(ma, mb)
                        end[k] = 1 if lower[a, k] else 0
                        strength[k] = abs(ma - mb)
                        # at the lower end the fewer at most k are ahead, at the upper end the more above it
                        ahead = mb - ma if lower[a, k] else ma - mb
                        side[k] = 1 if ahead > 0 else -1 if ahead < 0 else 0
                    if side[k] != 0 and (first < 0 or close[k] > close[first]
                                         or (close[k] == close[first] and end[k] > end[first])):
                        first = k
                if first < 0:  # the same shares at every threshold: a tie
                    verdict[a, b] = verdict[b, a] = 0
                    gap[a, b] = gap[b, a] = 0.0
                    continue
                ahead = side[first]
                least = strength[first]  # the cost of the first threshold, which none comes before
                for k in range(levels - 1):
                    # the first threshold's term of the cost of k is min(its strength,
                    # close[first] - close[k]): at least the gap where k is that far from it
                    reach = close[first] - close[k]
                    if k == first or reach >= least:
                        continue
                    cost = max(strength[k] if side[k] == ahead else 0.0, reach)
                    for j in range(levels - 1):
                        if cost >= least:  # the cost is the largest of its terms: no smaller gap from k
                            break
                        if j != k and j != first and side[j] == ahead and (
                                close[j] > close[k] or (close[j] == close[k] and end[j] > end[k])):
                            cost = max(cost, min(strength[j], close[j] - close[k]))
                    least = min(least, cost)
                verdict[a, b], verdict[b, a] = ahead, -ahead
                gap[a, b] = gap[b, a] = least
        count = 0  # the remaining candidates, the one ahead first; a tie keeps the lower index first
        for c in range(n):
            out[p, c] = -n  # below every remaining candidate
            if alive[p, c]:
                i = count
                while i > 0 and verdict[c, order[i - 1]] > 0:
                    order[i] = order[i - 1]
                    i -= 1
                order[i] = c
                count += 1
        total = 0.0
        for place in range(count):
            if place > 0:
                step = np.inf  # the smallest gap of a candidate above the place and one below
                for i in range(place):
                    for j in range(place, count):
                        step = min(step, gap[order[i], order[j]])
                below = total - step
                if below == total and verdict[order[place - 1], order[place]] > 0:
                    below = np.nextafter(total, -np.inf)  # behind by MJ's rule for the same t
                total = below
            out[p, order[place]] = total


def majority_totals(grades, alive=None) -> np.ndarray:
    """The totals of Tally(ballot, statistic="median") at every point, (..., C), for the
    shares grades (..., C, levels) of each score (Voters.graded), among the candidates
    marked in alive (..., C), all for None. The remaining candidates are in the order of
    the majority judgment: the first 0, each next one lower by the gap of its place. The
    others get -C, below all of them, as a score ballot is not marked again among the
    remaining candidates of an elimination. Chunks of the points run in parallel threads."""
    grades = np.ascontiguousarray(grades, dtype=np.float64)
    if grades.ndim < 2 or grades.shape[-1] < 2:
        raise ValueError(f"the shares of every score must be (..., C, levels) with at least 2 levels, "
                         f"got shape {grades.shape}")
    if not (np.isfinite(grades).all() and (grades >= 0).all() and (grades.sum(axis=-1) > 0).all()):
        raise ValueError("the shares of every score must be finite and 0 or more, and more than 0 in all for "
                         "every candidate")
    flat = grades.reshape(-1, *grades.shape[-2:])
    if alive is None:
        marked = np.ones(flat.shape[:2], dtype=bool)
    else:
        marked = np.ascontiguousarray(np.broadcast_to(alive, grades.shape[:-1]), dtype=bool).reshape(flat.shape[:2])
    out = np.empty(flat.shape[:2])
    threads.in_chunks(lambda a, b: _judge(flat[a:b], marked[a:b], out[a:b]), len(flat), CHUNK)
    return out.reshape(grades.shape[:-1])


def majority_grades(grades) -> np.ndarray:
    """The majority grade of every candidate, (..., C): its lower median, the smallest
    score k with at least as many of its voters at most k as above it, for the shares
    grades (..., C, levels) of each score. Summed from both ends, as majority_totals."""
    grades = np.asarray(grades, dtype=np.float64)
    at_most = np.cumsum(grades[..., :-1], axis=-1)
    above = np.cumsum(grades[..., :0:-1], axis=-1)[..., ::-1]
    return (above > at_most).sum(axis=-1)
