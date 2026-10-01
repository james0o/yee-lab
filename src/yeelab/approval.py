"""Approval ballots of voters in the plane: who a voter approves, and why the borders
of that are straight lines.

A voter at v values candidate c_i by the squared distance, u_i = -|v - c_i|^2, and
scales it so that the closest candidate a has 1 and the farthest b has 0:

    t_i = (u_i - u_b) / (u_a - u_b),    0 <= t_i <= 1.

The ballot is a cut of the candidates in that order, in one of two ways (`cut`):

    a threshold in (0, 1]   the candidates with t_i >= threshold. 1 approves only the
                            closest one (plurality), a threshold near 0 everyone but
                            the farthest (anti-plurality)
    GAP                     the candidates above the largest gap between two
                            neighbours in the order (ties: the first such gap)

Either way the closest candidate is approved and the farthest is not, and nothing
changes when all utilities are scaled or shifted. For three candidates the two are the
same ballot at the threshold 1/2: the middle candidate is above the larger gap exactly
when it is closer to the top than to the bottom.

Straight borders. u_i = -|v|^2 + 2 v . c_i - |c_i|^2, and -|v|^2 is the same for every
candidate, so it drops out of any combination whose weights sum to 0:

    sum_j w_j u_j >= 0   <=>   n . v <= o,    n = -2 sum_j w_j c_j,  o = -sum_j w_j |c_j|^2

(half_plane). The bisector of two candidates is w = e_c - e_e, and

    t_i >= threshold      u_i - threshold u_a - (1 - threshold) u_b >= 0
    gap k >= gap l        (u_k - u_k+1) - (u_l - u_l+1) >= 0, in the order of the voter

are of that kind too, where the closest and farthest candidate (or the whole order) do
not change. So the voters who approve a candidate are a union of convex polygons, and
their share comes from the same edge terms as the ranking cells (margin/shares.py). For
a threshold the lines of candidate i within one such cell all pass through the centre of
the circle through c_i, c_a and c_b, where the three utilities are equal: from the
bisector of c_i and c_b (threshold 0) to that of c_i and c_a (threshold 1).
"""

from typing import Literal

import numpy as np

GAP = "gap"
Cut = float | Literal["gap"]  # a threshold in (0, 1], or GAP


def utilities(points, candidates) -> np.ndarray:
    """u[..., c] = -|p - c|^2 for points (..., 2) and candidates (C, 2)."""
    points = np.asarray(points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    return -((points[..., None, :] - candidates) ** 2).sum(axis=-1)


def approved(points, candidates, cut: Cut) -> np.ndarray:
    """Whether a voter at each of the points (..., 2) approves each candidate, bool
    (..., C): the ballot from its definition, one voter at a time."""
    u = utilities(points, candidates)
    if cut == GAP:
        order = np.argsort(-u, axis=-1, kind="stable")  # best first
        best_first = np.take_along_axis(u, order, axis=-1)
        widest = (best_first[..., :-1] - best_first[..., 1:]).argmax(axis=-1)
        position = np.argsort(order, axis=-1, kind="stable")
        return position <= widest[..., None]
    best, worst = u.max(axis=-1, keepdims=True), u.min(axis=-1, keepdims=True)
    return u - worst >= cut * (best - worst)


def half_plane(candidates, weights) -> tuple[np.ndarray, float]:
    """(normal, offset) of the voters with sum_j weights[j] u_j >= 0, as normal . v <=
    offset; the weights (C,) must sum to 0."""
    candidates = np.asarray(candidates, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return -2 * weights @ candidates, -float(weights @ (candidates ** 2).sum(axis=-1))
