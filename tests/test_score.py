"""Checks of the score ballots (score.py) and their shares (margin/shares.py).

The ballot of one voter is checked from its definition: by every rule the closest
candidate gets the top score and the farthest 0. RANGE gives the others the score in
proportion to where their distance is between the two, rounded like np.round; with two
levels it approves the candidates closer than halfway. AVG puts the mean distance in the
middle of that scale; with two levels it approves the candidates closer than the mean,
which adds to the expected utility. DHONDT shares the steps out among the gaps by the
divisor method of its delta (D'Hondt at 1), which is checked by the condition every
apportionment of that method meets and no other does; with two levels it approves the
candidates above the largest gap, for any delta, and it does not become Borda with more
levels than candidates. A lower delta moves steps from the large gaps to the small ones.
HYBRID is DHONDT for the candidates closer than halfway, on the upper half of the
scale, and RANGE from halfway for the others, on the lower half; the two halves meet in
one score and never cross. With two levels AVG, DHONDT and HYBRID are checked against
approval ballots defined on their own (_approval); for three candidates RANGE, AVG and
DHONDT give the same ballot, and HYBRID approves the middle one only within a quarter of
the way. RANGE with a power raises its part of the way to it; with two levels it
approves the candidates beyond 2^(-1 / power) of the way. CLUSTER's ballot must have the least of the cost it minimizes, computed apart
from score.py over every ballot it may give; with mu = 0 it is RANGE. The compiled ballot
of the grid must be that of the definition.

The shares come from a grid of voters, so they are approximate. They are compared with
exact shares where there are some (two candidates get the top score from their first
choices and 0 from the others, and with two levels the closest candidate is approved
and the farthest is not) and with sampled voters elsewhere. They are computed as the
part of the top score not given, which must stay exact where it is tiny: the lead
between two candidates nearly all voters give the top score.
"""

from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pytest
from scipy.special import ndtr

from yeelab import ranking_cells
from yeelab.margin.shares import (GRID_CELLS, Model, _voter_grid, first_choice_shares, ranking_shares,
                                  unscored_shares)
from yeelab.score import (AVG, CLUSTER, DELTA, DHONDT, HYBRID, KAPPA, MAX_CANDIDATES, MAX_LEVELS, MU, POWER, RANGE,
                          RULES, distances, from_distances, scored, unscored)

FIVE = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
CANDIDATES = {2: np.random.default_rng(5).random((2, 2)), 3: np.random.default_rng(3).random((3, 2)),
              4: np.random.default_rng(6).random((4, 2)), 5: FIVE,
              6: np.random.default_rng(4).random((6, 2)), 7: np.random.default_rng(1).random((7, 2))}
LEVELS = [2, 3, 6, 11, 16]
DELTAS = [0.5, DELTA, 1.0]  # the range of the UI's slider and its default
MODELS = [Model("beta", 0.2, "rms"), Model("beta", 0.05, "rms"), Model("beta", 0.35, "mean_abs"),
          Model("normal", 0.2), Model("normal", 0.05)]
NODES = [(5, -9), (24, 24), (-5, 10), (0, 0)]  # of the nodes per axis, where the shares are sampled
SAMPLES = 400_000
# The grid shares against exact ones: 5e-4 seen for the narrowest voters, 2e-4 otherwise.
TOLERANCE = 2e-3


def _ids(value):
    return f"{value.distribution}-{value.deviation}" if isinstance(value, Model) else str(value)


def _at(distances_):
    """Candidates at the given distances from a voter at the origin, along the axes in
    turn: there the distances come back as given, to the last bit."""
    distances_ = np.asarray(distances_, dtype=np.float64)
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    return distances_[:, None] * axes[np.arange(len(distances_)) % len(axes)]


def _ballot(distances_, levels, rule=RANGE, delta=DELTA):
    return scored(np.zeros(2), _at(distances_), levels, rule, delta).tolist()


def _approval(points, candidates, cut):
    """Who a voter at each of the points approves, bool (..., C), by the approval ballot's
    own definition, apart from score.py: the closest candidates down to the largest gap
    between neighbours in the order of distance (cut "gap", ties: the first gap), down to
    the largest gap among the candidates closer than halfway, halfway itself counted as
    the next candidate ("hybrid", ties: the first gap) or those closer than the mean
    distance ("avg", ties: not approved)."""
    r = distances(points, candidates)
    order = np.argsort(r, axis=-1, kind="stable")  # closest first; ties: the lowest index
    if cut == "gap":
        count = np.diff(np.take_along_axis(r, order, axis=-1), axis=-1).argmax(axis=-1) + 1
    elif cut == "hybrid":
        s = np.take_along_axis(r, order, axis=-1)
        halfway = (s[..., :1] + s[..., -1:]) / 2
        closer = (s < halfway).sum(axis=-1, keepdims=True)
        # the closer ones, then halfway in place of every other: gaps of 0 past it
        ends = np.where(np.arange(s.shape[-1]) < closer, s, halfway)
        count = np.diff(ends, axis=-1).argmax(axis=-1) + 1
    else:
        count = np.maximum((r < r.mean(axis=-1, keepdims=True)).sum(axis=-1), 1)
    return np.argsort(order, axis=-1, kind="stable") < count[..., None]


def _top_shares(candidates, model, count):
    """Share of the voters with each candidate among their `count` closest: exact, from
    the cells of the ranking."""
    rankings, shares = ranking_shares(candidates, model)
    top = np.zeros((len(rankings), len(candidates)))
    np.put_along_axis(top, rankings[:, :count].astype(np.int64), 1.0, axis=1)
    return shares @ top

# ---------------------------------------------------------------- One voter


def test_three_close_candidates_and_a_far_one_get_the_top_score_and_none():
    """Distances 1, 2, 3 and 100: next to the fourth, the first three are as close."""
    assert _ballot([1, 2, 3, 100], 3) == [2, 2, 2, 0]
    assert _ballot([1, 2, 3, 100], 2) == [1, 1, 1, 0]
    assert _ballot([1, 2, 3, 100], 11) == [10, 10, 10, 0]
    assert _ballot([100, 3, 1, 2], 3) == [0, 2, 2, 2]  # in any order of the candidates


@pytest.mark.parametrize("levels", LEVELS)
def test_the_score_is_the_rounded_part_of_the_way(levels):
    rng = np.random.default_rng(levels)
    for n in range(2, 9):
        dist = rng.random((500, n)) + 0.01
        for row in dist:
            expected = np.round((row.max() - row) / (row.max() - row.min()) * (levels - 1))
            assert _ballot(row, levels) == expected.tolist()


@pytest.mark.parametrize("power", [0.5, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("levels", LEVELS)
def test_a_power_raises_the_part_of_the_way_to_it(levels, power):
    rng = np.random.default_rng(levels)
    dist = rng.random((2000, 6)) + 0.01
    part = (dist.max(axis=1, keepdims=True) - dist) / np.ptp(dist, axis=1, keepdims=True)
    np.testing.assert_array_equal(from_distances(dist, levels, RANGE, power=power),
                                  np.round(part ** power * (levels - 1)))
    assert (from_distances(dist, levels, RANGE, power=POWER) == from_distances(dist, levels, RANGE)).all()


def test_a_power_above_1_keeps_the_top_scores_for_the_near_candidates():
    """With two levels the voter approves those beyond 2^(-1 / power) of the way: 0.71 for
    power 2, so of distances 0, 2, 3, 6 and 10 (parts 1, 0.8, 0.7, 0.4, 0) the third is
    approved with power 1 but not 2. A higher power never gives a candidate more."""
    assert _ballot([0, 2, 3, 6, 10], 2) == [1, 1, 1, 0, 0]
    assert scored(np.zeros(2), _at([0, 2, 3, 6, 10]), 2, power=2.0).tolist() == [1, 1, 0, 0, 0]
    dist = np.random.default_rng(2).random((5000, 6))
    for levels in LEVELS:
        ballots = [from_distances(dist, levels, RANGE, power=power) for power in (0.5, 1, 1.5, 2, 3)]
        assert all((more <= less).all() for less, more in zip(ballots, ballots[1:]))


def test_evenly_spaced_candidates_get_evenly_spaced_scores():
    assert _ballot([1, 2, 3, 4, 5], 5) == [4, 3, 2, 1, 0]
    assert _ballot([1, 2, 3, 4, 5], 9) == [8, 6, 4, 2, 0]
    # halves, exact in binary, go to the even score as with np.round: 1.5 to 2 and 0.5 to 0
    assert _ballot([1, 2, 3, 4, 5], 3) == [2, 2, 1, 0, 0]
    assert _ballot([0, 1, 2, 3, 4, 5, 6, 7, 8], 5) == [4, 4, 3, 2, 2, 2, 1, 0, 0]
    assert _ballot([1, 2, 3], 2) == [1, 0, 0]  # halfway is a half: to 0


def test_a_voter_as_far_from_every_candidate_gives_them_all_the_top_score():
    assert _ballot([3, 3, 3], 6) == _ballot([3, 3, 3], 6, AVG) == [5, 5, 5]
    assert _ballot([0.5, 0.5], 2) == _ballot([0.5, 0.5], 2, AVG) == [1, 1]


def test_avg_puts_the_mean_distance_in_the_middle_of_the_scale():
    """The distances 0, 1, 3 and 8 have the mean 3, where AVG gives the middle score, and
    the midrange 4, where RANGE does. AVG's part of the way is 1, 0.83, 0.5 and 0, RANGE's
    1, 0.875, 0.625 and 0."""
    assert _ballot([0, 1, 3, 8], 5, AVG) == [4, 3, 2, 0]
    assert _ballot([0, 1, 3, 8], 5) == [4, 4, 2, 0]  # 3.5 and 2.5 to the even score
    assert _ballot([8, 3, 1, 0], 5, AVG) == [0, 2, 3, 4]  # in any order of the candidates
    # the mean 0.658 is beyond 0.65, the midrange 0.55 is not
    r = [0.1, 0.5, 0.65, 0.8, 0.9, 1]
    assert _ballot(r, 2, AVG) == [1, 1, 1, 0, 0, 0]
    assert _ballot(r, 2) == [1, 1, 0, 0, 0, 0]
    assert _ballot(r, 6, AVG) == [5, 3, 3, 1, 1, 0]
    assert _ballot(r, 6) == [5, 3, 2, 1, 1, 0]
    assert _ballot(r, 6, AVG, 0.5) == _ballot(r, 6, AVG)  # AVG has no delta


def test_avg_with_two_levels_is_approval_of_the_mean():
    """For any number of candidates; at the mean itself the half is rounded to 0, as the
    approval ballot does not approve a tie."""
    rng = np.random.default_rng(12)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((5000, 2))
        np.testing.assert_array_equal(scored(points, candidates, 2, AVG), _approval(points, candidates, "avg"))
    assert _ballot([0, 1, 2], 2, AVG) == [1, 0, 0]


def test_avg_with_two_levels_approves_what_adds_to_the_expected_utility():
    """With utility -r and every pair of candidates equally likely to tie, approving c
    is worth sum_e (r_e - r_c): AVG approves exactly the candidates where it is > 0."""
    rng = np.random.default_rng(7)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r = distances(points, candidates)
        worth = r.sum(axis=-1, keepdims=True) - n * r
        np.testing.assert_array_equal(scored(points, candidates, 2, AVG), worth > 0)


@pytest.mark.parametrize("rule", RULES)
def test_the_closest_gets_the_top_score_and_the_farthest_none(rule):
    rng = np.random.default_rng(1)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r, rows = distances(points, candidates), np.arange(2000)
        for levels in LEVELS:
            ballot = scored(points, candidates, levels, rule)
            assert ballot.dtype == np.int64 and ballot.shape == (2000, n)
            assert (ballot[rows, r.argmin(axis=-1)] == levels - 1).all()
            assert (ballot[rows, r.argmax(axis=-1)] == 0).all()
            # a closer candidate never gets a lower score
            by_distance = np.take_along_axis(ballot, r.argsort(axis=-1), axis=-1)
            assert (np.diff(by_distance, axis=-1) <= 0).all()


@pytest.mark.parametrize("rule", RULES)
def test_two_candidates_get_the_top_score_and_none_at_any_number_of_levels(rule):
    rng = np.random.default_rng(2)
    candidates, points = rng.random((2, 2)), rng.random((2000, 2))
    closest = distances(points, candidates).argmin(axis=-1)
    for levels in LEVELS:
        np.testing.assert_array_equal(scored(points, candidates, levels, rule),
                                      (levels - 1) * np.eye(2, dtype=np.int64)[closest])


def test_two_levels_approve_the_candidates_closer_than_halfway():
    rng = np.random.default_rng(3)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r = distances(points, candidates)
        halfway = (r.min(axis=-1, keepdims=True) + r.max(axis=-1, keepdims=True)) / 2
        clear = np.abs(r - halfway) > 1e-12  # not within rounding of halfway
        np.testing.assert_array_equal(scored(points, candidates, 2)[clear], (r < halfway)[clear])


def test_two_levels_are_one_approval_ballot_for_three_candidates_only():
    """The middle one of three is approved where it is closer to the closest than to
    the farthest, by RANGE, AVG and DHONDT: closer than halfway, closer than the mean,
    above the largest gap (HYBRID asks for less, see below). Four have four ballots."""
    rng = np.random.default_rng(4)
    points = rng.random((5000, 2))
    for n in (2, 3):
        candidates = rng.random((n, 2))
        for rule in (AVG, DHONDT):
            np.testing.assert_array_equal(scored(points, candidates, 2, rule), scored(points, candidates, 2))
    candidates = rng.random((4, 2))
    ballots = [scored(points, candidates, 2, rule) for rule in RULES]
    assert all((one != other).any() for one, other in combinations(ballots, 2))


@pytest.mark.parametrize("rule", RULES)
def test_the_ballot_does_not_change_with_the_scale(rule):
    rng = np.random.default_rng(5)
    candidates, points = rng.random((7, 2)), rng.random((3000, 2))
    for levels in (2, 6):
        scaled = scored(7 * points - 2, 7 * candidates - 2, levels, rule)
        assert (scaled == scored(points, candidates, levels, rule)).mean() > 0.9999  # but at a border, to rounding


def test_dhondt_shares_the_steps_out_among_the_gaps():
    """The distances 0, 0.59, 0.67, 0.74 and 1 of docs/math.pdf have the gaps 0.59,
    0.08, 0.07 and 0.26. By D'Hondt (delta = 1) two steps both go to the first gap
    (0.59 / 2 > 0.26), five split 4 to 1. RANGE rounds the part of the way instead, which
    is 1, 0.41, 0.33, 0.26 and 0 of the top score."""
    r = np.sqrt([0, 0.35, 0.45, 0.55, 1])
    assert _ballot(r, 2, DHONDT, 1) == [1, 0, 0, 0, 0]
    assert _ballot(r, 3, DHONDT, 1) == [2, 0, 0, 0, 0]
    assert _ballot(r, 6, DHONDT, 1) == [5, 1, 1, 1, 0]
    assert _ballot(r, 16, DHONDT, 1) == [15, 6, 5, 4, 0]
    assert _ballot(r, 6) == [5, 2, 2, 1, 0]
    assert _ballot(r[::-1], 6, DHONDT, 1) == [0, 1, 1, 1, 5]  # in any order of the candidates


def test_a_lower_delta_gives_the_small_gaps_more_steps():
    """The distances 0.1, 0.5, 0.65, 0.8, 0.9 and 1 have one large gap, 0.4, and small
    ones of 0.15 and 0.1; the large gap's share of the five steps is 2.2. D'Hondt and the
    default give it three, Sainte-Laguë (1/2) two; with delta near 0 every gap gets a step
    first (Borda), and with a large one every step goes to the largest gap (approval)."""
    r = [0.1, 0.5, 0.65, 0.8, 0.9, 1]
    assert _ballot(r, 6, DHONDT, 0.25) == [5, 4, 3, 2, 1, 0]
    assert _ballot(r, 6, DHONDT, 0.5) == [5, 3, 2, 1, 0, 0]
    assert _ballot(r, 6, DHONDT, DELTA) == _ballot(r, 6, DHONDT, 1) == [5, 2, 1, 0, 0, 0]
    assert _ballot(r, 6, DHONDT, 3) == [5, 0, 0, 0, 0, 0]
    assert _ballot(r, 6, RANGE, 0.5) == _ballot(r, 6) == [5, 3, 2, 1, 1, 0]  # RANGE has no delta


def test_dhondt_is_not_borda_with_more_levels_than_candidates():
    """Every gap with a step of its own is Borda. The default delta gives the small gaps
    none while one gap is much larger, however many levels there are; evenly spaced
    candidates get evenly spaced scores, which is Borda where the steps go round once."""
    assert _ballot([1, 2, 3, 100], 11, DHONDT) == [10, 10, 10, 0]
    assert _ballot([1, 2, 3, 100], 16, DHONDT) == [15, 15, 15, 0]
    assert _ballot([1, 2, 3, 4, 5], 5, DHONDT) == [4, 3, 2, 1, 0]
    assert _ballot([1, 2, 3, 4, 5], 9, DHONDT) == [8, 6, 4, 2, 0]
    assert _ballot([1, 2, 3, 4, 5], 3, DHONDT) == [2, 1, 0, 0, 0]  # tied gaps: the first ones


@pytest.mark.parametrize("delta", [*DELTAS, 3.0])
def test_dhondt_with_two_levels_is_approval_of_the_largest_gap(delta):
    """For any number of candidates and any delta, ties included: the first of equal
    gaps."""
    rng = np.random.default_rng(9)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((5000, 2))
        np.testing.assert_array_equal(scored(points, candidates, 2, DHONDT, delta),
                                      _approval(points, candidates, "gap"))
    assert _ballot([1, 2, 3], 2, DHONDT, delta) == [1, 0, 0]
    assert _ballot([3, 3, 3], 6, DHONDT, delta) == [5, 0, 0]  # all gaps tie: every step to the first


def _steps(points, candidates, levels, delta):
    """(gaps, steps) of the DHONDT ballots, (..., C - 1) each, in the order of distance."""
    r = distances(points, candidates)
    order = r.argsort(axis=-1, kind="stable")
    ballot = np.take_along_axis(scored(points, candidates, levels, DHONDT, delta), order, axis=-1)
    return np.diff(np.take_along_axis(r, order, axis=-1), axis=-1), -np.diff(ballot, axis=-1)


@pytest.mark.parametrize("delta", DELTAS)
def test_dhondt_meets_the_condition_of_its_apportionment(delta):
    """An apportionment d of levels - 1 steps to the gaps g is that of the divisor method
    of delta exactly where no gap would be worth its next step more than any gap is worth
    its last one: max_k g_k / (d_k + delta) <= min over d_k > 0 of g_k / (d_k - 1 + delta).
    Every gap is within C - 1 steps of its share (levels - 1) g_k / sum(g), and D'Hondt
    (delta = 1) meets the lower quota: no gap gets fewer steps than the share rounded
    down."""
    rng = np.random.default_rng(10)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((3000, 2))
        for levels in LEVELS:
            gaps, steps = _steps(points, candidates, levels, delta)
            assert (steps >= 0).all() and (steps.sum(axis=-1) == levels - 1).all()
            next_step = (gaps / (steps + delta)).max(axis=-1)
            last_step = np.where(steps > 0, gaps / np.maximum(steps - 1 + delta, delta), np.inf).min(axis=-1)
            assert (next_step <= last_step * (1 + 1e-12)).all()
            quota = (levels - 1) * gaps / gaps.sum(axis=-1, keepdims=True)
            assert (np.abs(steps - quota) <= n - 1).all()
            if delta == 1:
                assert (steps >= np.floor(quota - 1e-9)).all()


@pytest.mark.parametrize("delta", DELTAS)
def test_dhondt_more_levels_add_steps_and_take_none(delta):
    """Every divisor method is house monotone: one more level gives one gap one more
    step. Every candidate's score and its points below the top score grow or stay."""
    rng = np.random.default_rng(11)
    candidates, points = rng.random((6, 2)), rng.random((3000, 2))
    for levels in range(2, MAX_LEVELS):
        steps, more = (_steps(points, candidates, n, delta)[1] for n in (levels, levels + 1))
        assert ((more - steps).sum(axis=-1) == 1).all() and (more >= steps).all()


def test_hybrid_is_dhondt_for_the_closer_half_and_range_for_the_farther():
    """The distances 0.1, 0.5, 0.65, 0.8, 0.9 and 1 have the midrange 0.55: 0.1 and 0.5
    are closer, with the gap 0.4 between them and 0.05 from 0.5 to the midrange. With six
    levels the upper three steps all go to the gap 0.4, which leaves 0.5 with the lower
    two; the farther ones get 2 of the part of the way from 0.55 to 1, 0.78, 0.44 and
    0.22: 2, 1 and 0. RANGE gives 5, 3, 2, 1, 1, 0, DHONDT 5, 2, 1, 0, 0, 0."""
    r = [0.1, 0.5, 0.65, 0.8, 0.9, 1]
    assert _ballot(r, 6, HYBRID) == [5, 2, 2, 1, 0, 0]
    assert _ballot(r, 2, HYBRID) == [1, 0, 0, 0, 0, 0]
    assert _ballot(r, 3, HYBRID) == [2, 1, 1, 0, 0, 0]
    assert _ballot(r, 11, HYBRID) == [10, 5, 4, 2, 1, 0]
    assert _ballot(r[::-1], 6, HYBRID) == [0, 0, 1, 2, 2, 5]  # in any order of the candidates
    assert _ballot([3, 3, 3], 6, HYBRID) == [5, 5, 5]


def test_hybrid_shares_its_upper_steps_among_the_gaps_of_the_closer_half():
    """The distances 0, 6, 8, 9 and 20 have the midrange 10 and the gaps 6, 2 and 1 among
    the closer four, and 1 from 9 to 10. Of the three upper steps D'Hondt gives the gap 6
    all (6 / 3 ties 2 / 1: the first gap), Sainte-Laguë and the default two and one to
    the gap 2. With DHONDT the far gap 11 takes three of the five steps, at any of these
    delta, and 6, 8 and 9 all get 3."""
    r = [0, 6, 8, 9, 20]
    assert _ballot(r, 6, HYBRID, 1) == [5, 2, 2, 2, 0]
    assert _ballot(r, 6, HYBRID, 0.5) == _ballot(r, 6, HYBRID) == [5, 3, 2, 2, 0]
    for delta in DELTAS:
        assert _ballot(r, 6, DHONDT, delta) == [5, 3, 3, 3, 0]


@pytest.mark.parametrize("delta", [*DELTAS, 3.0])
def test_hybrid_with_two_levels_is_approval_of_the_largest_gap_of_the_closer_half(delta):
    """For any number of candidates and any delta, ties included: the first of equal gaps,
    so 0, 1 and 4 approve only the first (the gap to the midrange 2 ties the gap 1)."""
    rng = np.random.default_rng(13)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((5000, 2))
        np.testing.assert_array_equal(scored(points, candidates, 2, HYBRID, delta),
                                      _approval(points, candidates, "hybrid"))
    assert _ballot([0, 1, 4], 2, HYBRID, delta) == [1, 0, 0]
    assert _ballot([0, 1, 4], 2) == [1, 1, 0]


def test_hybrid_approves_the_middle_of_three_within_a_quarter_of_the_way():
    """Its gap from the closest is below its gap to halfway: r_2 < (3 r_1 + r_3) / 4."""
    rng = np.random.default_rng(14)
    candidates, points = rng.random((3, 2)), rng.random((5000, 2))
    r = np.sort(distances(points, candidates), axis=-1)
    quarter = (3 * r[:, 0] + r[:, 2]) / 4
    clear = np.abs(r[:, 1] - quarter) > 1e-12
    middle = np.take_along_axis(scored(points, candidates, 2, HYBRID),
                                distances(points, candidates).argsort(axis=-1), axis=-1)[:, 1]
    np.testing.assert_array_equal(middle[clear], (r[:, 1] < quarter)[clear])
    assert 0.1 < middle.mean() < 0.9


@pytest.mark.parametrize("levels", LEVELS)
def test_hybrid_gives_each_half_its_own_part_of_the_scale(levels):
    """The closer candidates get floor(T / 2) or more of T = levels - 1, the others that or
    less, which is the part of the way from the midrange to the farthest, rounded: for an
    even T the score of RANGE itself."""
    rng = np.random.default_rng(15)
    lower = (levels - 1) // 2
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r = distances(points, candidates)
        far, mid = r.max(axis=-1, keepdims=True), (r.min(axis=-1, keepdims=True) + r.max(axis=-1, keepdims=True)) / 2
        ballot, closer = scored(points, candidates, levels, HYBRID), r < mid
        assert (ballot[closer] >= lower).all() and (ballot[~closer] <= lower).all()
        part = (far - r) / (far - mid) * lower
        clear = ~closer & (np.abs(part % 1 - 0.5) > 1e-9)  # not within rounding of a half
        np.testing.assert_array_equal(ballot[clear], np.round(part[clear]))
        if (levels - 1) % 2 == 0:
            np.testing.assert_array_equal(ballot[clear], scored(points, candidates, levels)[clear])


def test_unknown_rules_are_rejected():
    for rule in ("gap", "dh", None):
        with pytest.raises(ValueError, match=f"unknown rule {rule!r}; choose from range, dhondt, avg, hybrid, cluster"):
            scored(np.zeros(2), FIVE, 3, rule)
        with pytest.raises(ValueError, match=f"unknown rule {rule!r}; choose from range, dhondt, avg, hybrid, cluster"):
            unscored(np.linspace(0, 1, 5), np.linspace(0, 1, 5), FIVE, 3, 4, rule)


def test_delta_must_be_above_zero():
    for delta in (0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match=f"delta must be a number above 0, got {delta!r}"):
            scored(np.zeros(2), FIVE, 3, DHONDT, delta)
        with pytest.raises(ValueError, match=f"delta must be a number above 0, got {delta!r}"):
            unscored(np.linspace(0, 1, 5), np.linspace(0, 1, 5), FIVE, 3, 4, DHONDT, delta)


def test_power_must_be_above_zero():
    lines = np.linspace(0, 1, 5)
    for power in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match=f"power must be a number above 0, got {power!r}"):
            scored(np.zeros(2), FIVE, 3, RANGE, power=power)
        with pytest.raises(ValueError, match=f"power must be a number above 0, got {power!r}"):
            unscored(lines, lines, FIVE, 3, 4, RANGE, power=power)


def test_mu_and_kappa_must_be_numbers_cluster_can_use():
    lines = np.linspace(0, 1, 5)
    for mu in (-0.1, np.inf, np.nan):
        with pytest.raises(ValueError, match=f"mu must be a number of 0 or more, got {mu!r}"):
            scored(np.zeros(2), FIVE, 3, CLUSTER, mu=mu)
        with pytest.raises(ValueError, match=f"mu must be a number of 0 or more, got {mu!r}"):
            unscored(lines, lines, FIVE, 3, 4, CLUSTER, mu=mu)
    for kappa in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match=f"kappa must be a number above 0, got {kappa!r}"):
            scored(np.zeros(2), FIVE, 3, CLUSTER, kappa=kappa)
        with pytest.raises(ValueError, match=f"kappa must be a number above 0, got {kappa!r}"):
            unscored(lines, lines, FIVE, 3, 4, CLUSTER, kappa=kappa)



def _cluster_cost(r, scores, levels, mu=MU, kappa=KAPPA):
    """The cost CLUSTER minimizes, from its definition, apart from score.py: for the
    scores (n,) of the distinct distances r (n,), closest first, the squared distances
    from RANGE's parts of the way, and mu * w_k * max(0, 1 - T g_k) for each split gap
    but the first and the last. w_k is the strongest run of distances gap k lies in:
    1 - kappa * its largest gap / the smaller gap around it, 0 at least."""
    top, g = levels - 1, np.diff(r) / (r[-1] - r[0])
    m = len(g)
    w = np.zeros(m)
    for i, j in combinations(range(m + 1), 2):  # the run of distances i..j
        if (i, j) != (0, m):
            around = min(g[i - 1] if i > 0 else np.inf, g[j] if j < m else np.inf)
            w[i:j] = np.maximum(w[i:j], 1 - kappa * g[i:j].max() / around)
    split = mu * w * np.maximum(0, 1 - top * g) * (np.diff(scores) != 0)
    return (((scores / top - (r[-1] - r) / (r[-1] - r[0])) ** 2).sum() + split[1:-1].sum())


@pytest.mark.parametrize(("mu", "kappa"), [(MU, KAPPA), (0.5, 1.0), (0.03, 4.0)])
@pytest.mark.parametrize("levels", [2, 3, 6, 11])
def test_cluster_gives_the_ballot_of_least_cost(levels, mu, kappa):
    """Among every ballot it may give, from T for the closest to 0 for the farthest and
    never more for a farther candidate, CLUSTER's costs the least, at the defaults of mu
    and kappa and away from them. Clusters of three are common enough among the random
    distances."""
    rng = np.random.default_rng(levels)
    top, splits = levels - 1, 0
    for n in range(2, 7):
        for row in rng.random((300 if n < 6 else 60, n)) ** 2:
            r = np.sort(row)
            ballot = from_distances(r, levels, CLUSTER, mu=mu, kappa=kappa)
            assert ballot[0] == top and ballot[-1] == 0 and (np.diff(ballot) <= 0).all()
            least = min(_cluster_cost(r, np.array([top, *inner, 0]), levels, mu, kappa)
                        for inner in combinations_with_replacement(range(top, -1, -1), n - 2))
            assert _cluster_cost(r, ballot, levels, mu, kappa) <= least + 1e-12
            splits += (ballot != from_distances(r, levels, RANGE)).any()
    assert splits > 0


def test_cluster_keeps_a_tight_cluster_with_few_levels_and_grades_it_with_many():
    """Two clusters: RANGE splits both with six levels, CLUSTER keeps each one score; with
    16 the gaps are a step or more and CLUSTER is RANGE. With two levels the cluster of
    0.44 and 0.54 is not split, which RANGE does with six."""
    r = [0, 0.04, 0.12, 0.88, 0.95, 1]
    assert _ballot(r, 6) == [5, 5, 4, 1, 0, 0]
    assert _ballot(r, 6, CLUSTER) == [5, 5, 5, 0, 0, 0]
    assert _ballot(r, 16, CLUSTER) == _ballot(r, 16) == [15, 14, 13, 2, 1, 0]
    assert _ballot(r[::-1], 6, CLUSTER) == [0, 0, 0, 5, 5, 5]  # in any order of the candidates
    assert _ballot([0, 0.44, 0.54, 1], 6) == [5, 3, 2, 0]
    assert _ballot([0, 0.44, 0.54, 1], 6, CLUSTER) == [5, 3, 3, 0]
    assert _ballot([0, 1, 2, 3, 21], 6, CLUSTER) == [5, 5, 5, 5, 0]
    assert _ballot([3, 3, 3], 6, CLUSTER) == [5, 5, 5]


@pytest.mark.parametrize("levels", LEVELS)
def test_cluster_without_a_cost_of_splitting_is_range(levels):
    r = np.random.default_rng(levels).random((20_000, 6))
    np.testing.assert_array_equal(from_distances(r, levels, CLUSTER, mu=0.0), from_distances(r, levels, RANGE))

# ---------------------------------------------------------------- Grid


@pytest.mark.parametrize(("rule", "delta", "mu", "kappa", "power"),
                         [(RANGE, DELTA, MU, KAPPA, power) for power in (POWER, 0.5, 1.5, 2.5)]
                         + [(AVG, DELTA, MU, KAPPA, POWER)]
                         + [(rule, delta, MU, KAPPA, POWER) for rule in (DHONDT, HYBRID) for delta in DELTAS]
                         + [(CLUSTER, DELTA, mu, kappa, POWER)
                            for mu, kappa in ((MU, KAPPA), (0.5, 1.0), (0.02, 5.0))])
@pytest.mark.parametrize("n", CANDIDATES)
def test_the_compiled_ballot_is_the_definition(n, rule, delta, mu, kappa, power):
    """A rectangle too small for a border to cross holds the ballot of the voter at it,
    as the points below the top score, for candidates and voters in and around the
    square."""
    rng = np.random.default_rng(n)
    points = 2 * rng.random((200, 2)) - 0.5
    for levels in LEVELS:
        short = np.stack([unscored([x, x + 1e-9], [y, y + 1e-9], CANDIDATES[n], levels, 2, rule, delta,
                                   mu=mu, kappa=kappa, power=power)[:, 0, 0] for x, y in points])
        np.testing.assert_array_equal(short, levels - 1 - scored(points, CANDIDATES[n], levels, rule, delta,
                                                                 mu=mu, kappa=kappa, power=power))


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("levels", [2, 4, 11])
def test_unscored_is_the_mean_of_each_rectangle(levels, rule):
    """A rectangle with one ballot at its four corners holds that ballot; the others the
    mean of the ballots at their sub x sub points. On lines that are not evenly spaced,
    and with different ones along x and y."""
    xs, ys, sub = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4
    short = np.moveaxis(unscored(xs, ys, FIVE, levels, sub, rule), 0, -1)
    assert short.shape == (40, 30, 5) and short.dtype == np.float32

    def below(points):
        return levels - 1 - scored(points, FIVE, levels, rule)

    corners = below(np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1))
    whole = ((corners[:-1, :-1] == corners[1:, :-1]) & (corners[:-1, :-1] == corners[:-1, 1:])
             & (corners[:-1, :-1] == corners[1:, 1:])).all(axis=-1)
    assert 0.1 < whole.mean() < 1
    np.testing.assert_array_equal(short[whole], corners[:-1, :-1][whole])

    at = (np.arange(sub) + 0.5) / sub
    inner_x = (xs[:-1, None] + at * np.diff(xs)[:, None]).ravel()
    inner_y = (ys[:-1, None] + at * np.diff(ys)[:, None]).ravel()
    ballots = below(np.stack(np.meshgrid(inner_x, inner_y, indexing="ij"), axis=-1))
    mean = ballots.reshape(40, sub, 30, sub, 5).mean(axis=(1, 3))
    np.testing.assert_allclose(short[~whole], mean[~whole], rtol=0, atol=1e-6)
    assert (short[~whole] % 1 != 0).any()


def test_unscored_rejects_what_it_cannot_do():
    lines = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="score ballots need 2 to 15 candidates, got 1"):
        unscored(lines, lines, FIVE[:1], 3, 4)
    with pytest.raises(ValueError, match="score ballots need 2 to 15 candidates, got 16"):
        unscored(lines, lines, np.random.default_rng(0).random((16, 2)), 3, 4)
    for levels in (1, 17):
        with pytest.raises(ValueError, match=f"score ballots need 2 to 16 levels, got {levels}"):
            unscored(lines, lines, FIVE, levels, 4)


@pytest.mark.parametrize("rule", RULES)
def test_the_most_candidates_and_levels_fit_a_ballot(rule):
    """15 candidates up to 15 points below the top score: the highest bits of the code."""
    assert (MAX_CANDIDATES, MAX_LEVELS) == (15, 16)
    rng = np.random.default_rng(6)
    candidates, points = rng.random((MAX_CANDIDATES, 2)), rng.random((100, 2))
    short = np.stack([unscored([x, x + 1e-9], [y, y + 1e-9], candidates, MAX_LEVELS, 2, rule)[:, 0, 0]
                      for x, y in points])
    np.testing.assert_array_equal(short, MAX_LEVELS - 1 - scored(points, candidates, MAX_LEVELS, rule))
    assert short.max() == MAX_LEVELS - 1

@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_the_voter_grid_holds_all_voters(model):
    """Equal cells across the square, each node's voters summing to 1 over the cells,
    and the cells of both models reaching where their voters are."""
    lines, mass = _voter_grid(model, model.medians)
    assert (np.diff(lines) > 0).all() and mass.shape == (model.nodes, len(lines) - 1)
    inside = lines[(lines > 0.01) & (lines < 0.99)]
    np.testing.assert_allclose(np.diff(inside), 1 / GRID_CELLS, atol=1e-12)
    np.testing.assert_allclose(mass.sum(axis=1), 1.0, atol=1e-12)
    assert (mass >= 0).all()
    if model.distribution == "beta":
        assert lines[0] == 0 and lines[-1] == 1 and lines[1] <= 1e-6
    else:
        assert lines[0] < -9 * model.sigma and lines[-1] > 1 + 9 * model.sigma


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_the_voter_grid_is_exact_in_both_tails(model):
    """The voters of the median 1 - m are those of m mirrored, so the cells above a
    median hold what the mirrored cells below the mirrored median do, down to the
    smallest shares. (Differences of a CDF near 1 would be 0 there.)"""
    medians = np.array([0.03, 0.4, 0.6, 0.97])
    lines, mass = _voter_grid(model, medians)
    np.testing.assert_allclose(lines, 1 - lines[::-1], rtol=0, atol=1e-15)
    assert (mass > 0).all()
    np.testing.assert_allclose(mass[::-1, ::-1], mass, rtol=1e-6, atol=0)
    if model.deviation == 0.05:
        assert mass.min() < 1e-30
    assert not _voter_grid(model, medians)[1].flags.writeable  # kept, so read-only

# ---------------------------------------------------------------- Shares


def _sample(model, node, rng):
    """Voters of one node, (SAMPLES, 2)."""
    i, j = node
    if model.distribution == "beta":
        params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
        return np.column_stack([rng.beta(*params[i], SAMPLES), rng.beta(*params[j], SAMPLES)])
    return rng.normal(model.medians[[i, j]], model.sigma, (SAMPLES, 2))


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_shares_match_sampled_voters(model, rule):
    rng = np.random.default_rng(7)
    for n, levels in ((5, 2), (5, 6), (7, 11)):
        shares = unscored_shares(CANDIDATES[n], model, levels, model.medians, rule)
        assert shares.shape == (model.nodes, model.nodes, n)
        assert (shares >= 0).all() and (shares <= 1 + 1e-12).all()
        for node in NODES:
            ballots = scored(_sample(model, node, rng), CANDIDATES[n], levels, rule)
            sampled = 1 - ballots.mean(axis=0) / (levels - 1)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=5e-3)  # 6 standard errors


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_candidates_are_scored_by_their_first_choices(model):
    """The closer one gets the top score and the other 0, at any number of levels and by
    every rule: the part of the top score not given is the share of the voters closer to
    the other, and the same share for all of them."""
    not_first = 1 - first_choice_shares(CANDIDATES[2], model)
    approval = unscored_shares(CANDIDATES[2], model, 2, model.medians)
    for levels, rule in product((2, 6, 16), RULES):
        unscored_ = unscored_shares(CANDIDATES[2], model, levels, model.medians, rule)
        np.testing.assert_allclose(unscored_, not_first, rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(unscored_, approval, rtol=1e-12, atol=0)  # the tiny ones too


@pytest.mark.parametrize("sigma_model", [Model("normal", 0.05), Model("normal", 0.2)], ids=_ids)
def test_the_part_not_given_is_exact_where_it_is_tiny(sigma_model):
    """Two candidates: a voter gives the farther one 0, and for normal voters the share
    closer to the other is Phi of the distance to their bisector. The grid share must
    follow it in relative terms, far below the rounding of 1 - share (6% seen at 1e-36:
    the border within a rectangle is only a share of the rectangle)."""
    medians = np.linspace(0.01, 0.99, 50)
    unscored_ = unscored_shares(CANDIDATES[2], sigma_model, 2, medians)
    diff = CANDIDATES[2][1] - CANDIDATES[2][0]
    threshold = (CANDIDATES[2][1] @ CANDIDATES[2][1] - CANDIDATES[2][0] @ CANDIDATES[2][0]) / 2
    mean = diff[0] * medians[:, None] + diff[1] * medians[None, :]
    z = (threshold - mean) / (sigma_model.sigma * np.linalg.norm(diff))  # to the bisector
    np.testing.assert_allclose(unscored_[..., 1], ndtr(z), rtol=0.1, atol=0)  # closer to 0
    np.testing.assert_allclose(unscored_[..., 0], ndtr(-z), rtol=0.1, atol=0)
    if sigma_model.deviation == 0.05:
        assert unscored_.min() < 1e-30


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_three_candidates_and_two_levels_have_one_share_by_range_avg_and_dhondt(model):
    for rule in (AVG, DHONDT):
        np.testing.assert_allclose(unscored_shares(CANDIDATES[3], model, 2, model.medians, rule),
                                   unscored_shares(CANDIDATES[3], model, 2, model.medians),
                                   rtol=1e-9, atol=0)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_levels_approve_from_the_closest_alone_to_all_but_the_farthest(model, rule):
    """With two levels a voter approves the closest candidate and not the farthest, so
    the share approving each is between the exact shares with it first and with it
    among the n - 1 closest."""
    for n in (5, 7):
        candidates = CANDIDATES[n]
        shares = 1 - unscored_shares(candidates, model, 2, model.medians, rule)
        assert (shares >= _top_shares(candidates, model, 1) - TOLERANCE).all()
        assert (shares <= _top_shares(candidates, model, n - 1) + TOLERANCE).all()
        assert (shares >= -1e-12).all() and (shares <= 1 + 1e-12).all()


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_more_levels_come_closer_to_the_part_of_the_way_itself(model):
    """Without rounding a voter gives c the part (r_max - r_c) / (r_max - r_min) of the top
    score. Rounding to `levels` scores moves a score by at most half a step, so the mean
    part not given is within 1 / (2 (levels - 1)) of the mean of the unrounded one, here
    taken from sampled voters."""
    rng = np.random.default_rng(8)
    candidates = CANDIDATES[5]
    for node in NODES[:2]:
        r = distances(_sample(model, node, rng), candidates)
        far = r.max(axis=-1, keepdims=True)
        exact = 1 - ((far - r) / (far - r.min(axis=-1, keepdims=True))).mean(axis=0)
        for levels in (3, 6, 16):
            shares = unscored_shares(candidates, model, levels, model.medians)[node]
            assert np.abs(shares - exact).max() <= 0.5 / (levels - 1) + 5e-3
        assert np.abs(shares - exact).max() < 0.02  # 16 levels
