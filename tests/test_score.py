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
from score.py over every ballot it may give; with mu = 0 it is RANGE. D21 gives the votes
worked out by hand for a few voters, never more votes than it may and a minus vote only
after enough plus votes; without minus votes it is RANGE's approval of at most
max_positive candidates (FPTP for one), and with its defaults one cut at halfway, both
defined apart from score.py. D21_ORDINAL gives the votes by the position in the order of
distance alone: with three candidates Borda's points or AntiPlurality's, and the share
approving with two levels is the exact share with the candidate among the max_positive
closest. The compiled ballot of the grid must be that of the definition.

The shares come from a grid of voters, so they are approximate. They are compared with
exact shares where there are some (two candidates get the top score from their first
choices and 0 from the others, and with two levels the closest candidate is approved
and the farthest is not) and with sampled voters elsewhere. They are computed as the
part of the top score not given, which must stay exact where it is tiny: the lead
between two candidates nearly all voters give the top score. What burying takes back,
the points a candidate gets from the voters who rank another above it, and what
compromising adds, the points below the top score from the voters who rank it above
another but neither first, are checked like the part not given: the mean of each
rectangle, against sampled voters, and against exact bounds (with two candidates the
one ranked lower gets nothing, and every voter has one of them first). All of them are
sums over one ballot table, whose weights in every rectangle add up to 1; the table is
kept, and one table by order gives what the separate functions give. How much burying
and compromising voters move the score comparisons is checked against the ballots moved
one voter at a time, in every rectangle and for sampled voters, and the score grid, which
gives these shares at the points of each pair of frontrunners, against the shares of all
points.
"""

from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pytest
from scipy.special import ndtr

from yeelab import ranking_cells
from yeelab.margin import shares as shares_module
from yeelab.margin.shares import (GRID_CELLS, GRID_SUB, Model, ScoreGrid, _grid_lines, _voter_grid, ballots,
                                  buried_shares, first_choice_shares, graded_shares, pairwise_shares, raised_shares,
                                  ranking_shares, score_comparison_shares, unscored_shares)
from yeelab.score import (AVG, BITS, BallotTable, CLUSTER, COMPARISONS, D21, D21_ORDINAL, DELTA, DHONDT, GIVEN,
                          GRADES, HYBRID, KAPPA, MAX_CANDIDATES, MAX_LEVELS, MU, POWER, RAISED, RANGE, RULES, UNSCORED, VOTES,
                          ballot_table, comparisons, distances, from_distances, given, graded, moved_comparisons,
                          moved_grades, raised, scored, tabulate, unscored)

FIVE = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
CANDIDATES = {2: np.random.default_rng(5).random((2, 2)), 3: np.random.default_rng(3).random((3, 2)),
              4: np.random.default_rng(6).random((4, 2)), 5: FIVE,
              6: np.random.default_rng(4).random((6, 2)), 7: np.random.default_rng(1).random((7, 2))}
LEVELS = [2, 3, 6, 11, 16]
# of any number of levels; D21's follow from its votes
SCALES = [rule for rule in RULES if rule not in (D21, D21_ORDINAL)]
DELTAS = [0.5, DELTA, 1.0]  # the range of the UI's slider and its default
MODELS = [Model("beta", 0.2, "rms"), Model("beta", 0.05, "rms"), Model("beta", 0.35, "mean_abs"),
          Model("normal", 0.2), Model("normal", 0.05)]
NODES = [(5, -9), (24, 24), (-5, 10), (0, 0)]  # of the nodes per axis, where the shares are sampled
SAMPLES = 100_000
SAMPLED = 1e-2  # 6 standard errors of a mean share of SAMPLES voters
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


@pytest.mark.parametrize("rule", SCALES)
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


@pytest.mark.parametrize("rule", SCALES)
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
    ballots = [scored(points, candidates, 2, rule) for rule in SCALES]
    assert all((one != other).any() for one, other in combinations(ballots, 2))


@pytest.mark.parametrize("rule", SCALES)
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
    choose = "choose from range, dhondt, avg, hybrid, cluster, d21, d21_ordinal"
    for rule in ("gap", "dh", None):
        with pytest.raises(ValueError, match=f"unknown rule {rule!r}; {choose}"):
            scored(np.zeros(2), FIVE, 3, rule)
        with pytest.raises(ValueError, match=f"unknown rule {rule!r}; {choose}"):
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
@pytest.mark.parametrize("levels", [2, 3, 6])
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


def _d21(parts, votes=VOTES, power=POWER, rule=D21):
    """The D21 ballot of a voter at the parts of the way (the closest 1, the farthest 0),
    as + for a plus vote, - for a minus vote and 0 for none."""
    levels = 3 if votes[1] else 2
    ballot = from_distances(1 - np.asarray(parts, dtype=np.float64), levels, rule, power=power, votes=votes)
    return "".join(("0+" if levels == 2 else "-0+")[score] for score in ballot)


def test_d21_gives_the_votes_worked_out_by_hand():
    """Score(3) gives Z two plus votes and so a minus vote; X only one, but Score(2)
    approves two, so X takes that and the minus vote with it; Y only one either way."""
    z, x, y = [1, 0.8, 0.4, 0], [1, 0.625, 0.375, 0], [1, 0.3, 0.2, 0]
    assert (_d21(z), _d21(x), _d21(y)) == ("++0-", "++0-", "+000")
    assert _d21(x, votes=(2, 1, 0)) == "+00-"  # a minus vote at once: no need to approve more
    assert (_d21(x, votes=(2, 0, 2)), _d21(y, votes=(2, 0, 2))) == ("++00", "+000")  # Score(2), at most 2
    assert _d21([1, 0.9, 0.8, 0.6, 0.1, 0], votes=(3, 2, 2)) == "+++0--"
    assert _d21([1, 0.9, 0.8, 0.6, 0.1, 0], votes=(2, 1, 2)) == "++000-"  # the closest and the farthest first
    assert _d21([1, 0.7, 0], power=2) == "+00"  # 0.49: not approved at the power 2, as by Score(2, power=2)
    assert _d21([1, 0.7, 0], power=1) == "++-"


def test_d21_never_gives_more_votes_than_it_may():
    """At most max_positive plus votes and max_negative minus votes, a minus vote only
    after negative_after plus votes, the closest always a plus vote, and a closer
    candidate never a lower score."""
    rng = np.random.default_rng(16)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r, rows = distances(points, candidates), np.arange(2000)
        for votes in ((2, 1, 2), (2, 1, 0), (3, 2, 2), (1, 1, 1), (2, 0, 2)):
            levels = 3 if votes[1] else 2
            ballot = scored(points, candidates, levels, D21, power=1.5, votes=votes)
            plus = (ballot == levels - 1).sum(axis=-1)
            minus = (ballot == 0).sum(axis=-1) if votes[1] else np.zeros_like(plus)
            assert (plus <= votes[0]).all() and (minus <= votes[1]).all()
            assert (plus[minus > 0] >= votes[2]).all()
            assert (ballot[rows, r.argmin(axis=-1)] == levels - 1).all()
            by_distance = np.take_along_axis(ballot, r.argsort(axis=-1), axis=-1)
            assert (np.diff(by_distance, axis=-1) <= 0).all()


@pytest.mark.parametrize("power", [POWER, 1.5])
def test_d21_without_minus_votes_is_approval_of_at_most_max_positive(power):
    """Score(2)'s approval, of the closest max_positive at most: FPTP for one, Score(2)
    itself for as many as there are candidates."""
    rng = np.random.default_rng(17)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r = distances(points, candidates)
        position = np.argsort(np.argsort(r, axis=-1, kind="stable"), axis=-1, kind="stable")
        approval = scored(points, candidates, 2, RANGE, power=power)
        for plus in range(1, n + 1):
            np.testing.assert_array_equal(scored(points, candidates, 2, D21, power=power, votes=(plus, 0, 0)),
                                          approval * (position < plus))
        np.testing.assert_array_equal(scored(points, candidates, 2, D21, power=power, votes=(1, 0, 0)),
                                      np.eye(n, dtype=np.int64)[r.argmin(axis=-1)])


@pytest.mark.parametrize("power", [POWER, 1.5])
def test_d21_with_its_defaults_is_one_cut_at_halfway(power):
    """With 2 plus votes and 1 minus vote after them, a voter whose Score(3) gives one
    plus vote approves as Score(2) does to get the minus vote, so every ballot is Score(2)'s
    approval of at most the 2 closest, with a minus vote for the farthest after 2."""
    rng = np.random.default_rng(18)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r = distances(points, candidates)
        position = np.argsort(np.argsort(r, axis=-1, kind="stable"), axis=-1, kind="stable")
        plus = scored(points, candidates, 2, RANGE, power=power) * (position < 2)
        minus = (plus.sum(axis=-1, keepdims=True) == 2) & (position == n - 1)
        np.testing.assert_array_equal(scored(points, candidates, 3, D21, power=power), 1 + plus - minus)


def test_d21_votes_and_levels_must_fit():
    """negative_after counts only with minus votes: above max_positive only without them."""
    for votes in ((0, 1, 0), (2, -1, 2), (2, 1, 3), (2, 1, -1), (2.0, 1, 2), (2, True, 2), (2, 1)):
        with pytest.raises(ValueError, match="votes must be whole numbers"):
            scored(np.zeros(2), FIVE, 3, D21, votes=votes)
    assert _d21([1, 0.6, 0], votes=(1, 0, 2)) == "+00"
    for rule in (D21, D21_ORDINAL):
        for levels, votes in ((2, (2, 1, 2)), (3, (2, 0, 2)), (6, (2, 1, 2))):
            with pytest.raises(ValueError, match=f"a D21 ballot has {3 if votes[1] else 2} levels"):
                scored(np.zeros(2), FIVE, levels, rule, votes=votes)
            with pytest.raises(ValueError, match=f"a D21 ballot has {3 if votes[1] else 2} levels"):
                unscored(np.linspace(0, 1, 5), np.linspace(0, 1, 5), FIVE, levels, 4, rule, votes=votes)


def test_d21_ordinal_gives_the_votes_by_the_order_alone():
    """Y, whom D21 gives one plus vote, still gives two and the minus vote; the votes go
    by position, all of them, the plus votes first where there are fewer candidates."""
    y = [1, 0.3, 0.2, 0]
    assert (_d21(y), _d21(y, rule=D21_ORDINAL)) == ("+000", "++0-")
    assert _d21(y, votes=(2, 0, 0), rule=D21_ORDINAL) == "++00"
    assert _d21([1, 0.9, 0.1, 0], votes=(1, 2, 0), rule=D21_ORDINAL) == "+0--"
    assert _d21([1, 0], rule=D21_ORDINAL) == "++" and _d21([1, 0.5, 0], votes=(2, 2, 0), rule=D21_ORDINAL) == "++-"
    rng = np.random.default_rng(19)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        position = np.argsort(np.argsort(distances(points, candidates), axis=-1, kind="stable"), axis=-1,
                              kind="stable")
        for plus, minus in ((2, 1), (1, 1), (3, 2), (2, 0)):
            levels, up = (3 if minus else 2), min(plus, n)
            down = min(minus, n - up)
            expected = np.where(position < up, levels - 1, np.where(position >= n - down, 0, levels - 2))
            np.testing.assert_array_equal(scored(points, candidates, levels, D21_ORDINAL, votes=(plus, minus, 0)),
                                          expected)
        if n == 3:  # Borda's points 2, 1, 0, and AntiPlurality's approval of all but the last
            np.testing.assert_array_equal(scored(points, candidates, 3, D21_ORDINAL, votes=(1, 1, 0)), 2 - position)
            np.testing.assert_array_equal(scored(points, candidates, 2, D21_ORDINAL, votes=(2, 0, 0)), position < 2)


def test_score_comparisons_count_strict_preferences_and_abstain_on_ties():
    tied_candidates = np.full((3, 2), [0.5, 0.5])
    tied = comparisons([0.4, 0.6], [0.4, 0.6], tied_candidates, 6, 1)
    np.testing.assert_array_equal(tied, 0.0)

    candidates = np.array([[0.0, 0.5], [1.0, 0.5]])
    strict = comparisons([0.2, 0.3], [0.45, 0.55], candidates, 2, 1)
    np.testing.assert_array_equal(strict[:, :, 0, 0], [[0.0, 1.0], [-1.0, 0.0]])


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


@pytest.mark.parametrize("votes", [VOTES, (2, 1, 0), (3, 2, 2), (2, 0, 2), (1, 0, 0)])
@pytest.mark.parametrize(("rule", "power"), [(D21, POWER), (D21, 1.5), (D21_ORDINAL, POWER)])
@pytest.mark.parametrize("n", CANDIDATES)
def test_the_compiled_d21_ballot_is_the_definition(n, rule, power, votes):
    """As for the other rules, with the levels of the votes."""
    rng = np.random.default_rng(n)
    points, levels = 2 * rng.random((200, 2)) - 0.5, 3 if votes[1] else 2
    short = np.stack([unscored([x, x + 1e-9], [y, y + 1e-9], CANDIDATES[n], levels, 2, rule, power=power,
                               votes=votes)[:, 0, 0] for x, y in points])
    np.testing.assert_array_equal(short, levels - 1 - scored(points, CANDIDATES[n], levels, rule, power=power,
                                                             votes=votes))


@pytest.mark.parametrize("rule", SCALES)
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


@pytest.mark.parametrize("rule", SCALES)
@pytest.mark.parametrize("levels", [2, 4, 11])
def test_graded_is_the_share_of_each_score_in_each_rectangle(levels, rule):
    """A rectangle with one ballot at its four corners holds all of its voters at that
    ballot's scores; the others the share of their sub x sub points at each score. A
    candidate's shares add up to 1 in every rectangle, and their mean points below the
    top score are those of unscored()."""
    xs, ys, sub = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4
    short = np.moveaxis(graded(xs, ys, FIVE, levels, sub, rule), (0, 1), (-2, -1))
    assert short.shape == (40, 30, 5, levels) and short.dtype == np.float32
    corners = scored(np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1), FIVE, levels, rule)
    whole = ((corners[:-1, :-1] == corners[1:, :-1]) & (corners[:-1, :-1] == corners[:-1, 1:])
             & (corners[:-1, :-1] == corners[1:, 1:])).all(axis=-1)
    assert 0.1 < whole.mean() < 1
    np.testing.assert_array_equal(short[whole], np.eye(levels)[corners[:-1, :-1][whole]])
    at = (np.arange(sub) + 0.5) / sub
    inner_x = (xs[:-1, None] + at * np.diff(xs)[:, None]).ravel()
    inner_y = (ys[:-1, None] + at * np.diff(ys)[:, None]).ravel()
    ballots_ = scored(np.stack(np.meshgrid(inner_x, inner_y, indexing="ij"), axis=-1), FIVE, levels, rule)
    mean = np.eye(levels)[ballots_].reshape(40, sub, 30, sub, 5, levels).mean(axis=(1, 3))
    np.testing.assert_allclose(short[~whole], mean[~whole], rtol=0, atol=1e-6)
    np.testing.assert_allclose(short.sum(axis=-1), 1, rtol=0, atol=1e-6)
    below = short.astype(np.float64) @ (levels - 1 - np.arange(levels))
    np.testing.assert_allclose(below, np.moveaxis(unscored(xs, ys, FIVE, levels, sub, rule), 0, -1), rtol=0, atol=1e-5)
    table = ballot_table(xs, ys, FIVE, levels, sub, rule, by_order=True)  # by order: finer, the same scores
    np.testing.assert_allclose(tabulate(table, GRADES).sum(axis=1), 1, rtol=0, atol=1e-6)


def _given_by(points, candidates, levels, rule=RANGE):
    """G[..., c, e] = points the voter at each of the points gives c if they rank e above it
    (closer, ties: the lower index first), else 0, from scored() and the order of
    distance."""
    rank = np.argsort(np.argsort(distances(points, candidates), axis=-1, kind="stable"), axis=-1)
    points_ = scored(points, candidates, levels, rule)
    return np.where(rank[..., None, :] < rank[..., :, None], points_[..., :, None], 0)


def _raised_by(points, candidates, levels, rule=RANGE):
    """G[..., c, e] = points below the top score the voter at each of the points gives c if
    they rank c above e but neither first, else 0."""
    rank = np.argsort(np.argsort(distances(points, candidates), axis=-1, kind="stable"), axis=-1)
    below = levels - 1 - scored(points, candidates, levels, rule)
    ranked = (rank[..., :, None] < rank[..., None, :]) & (rank[..., :, None] > 0)
    return np.where(ranked, below[..., :, None], 0)


def _moved_by(points, candidates, levels, a, b, bury, compromise, rule=RANGE):
    """M[..., c, e]: how much the strict comparison of c and e of the voter at each point
    moves, mixed over the kinds of strategic voters by their shares: burying (the
    frontrunner of a and b ranked lower gets 0), compromising (the other gets the top score
    if neither is first) and both."""
    rank = np.argsort(np.argsort(distances(points, candidates), axis=-1, kind="stable"), axis=-1)
    scores = scored(points, candidates, levels, rule).astype(np.int64)
    upper = np.where(rank[..., a] < rank[..., b], a, b)[..., None]
    lower = np.where(rank[..., a] < rank[..., b], b, a)[..., None]
    neither = (rank[..., a] > 0) & (rank[..., b] > 0)

    def compare(given_scores):
        return np.sign(given_scores[..., :, None] - given_scores[..., None, :])

    before, moved = compare(scores), 0.0
    for share, buries, compromises in ((bury * (1 - compromise), True, False),
                                       ((1 - bury) * compromise, False, True), (bury * compromise, True, True)):
        changed = scores.copy()
        if buries:
            np.put_along_axis(changed, lower, 0, axis=-1)
        if compromises:
            raised_ = np.where(neither[..., None], levels - 1, np.take_along_axis(changed, upper, axis=-1))
            np.put_along_axis(changed, upper, raised_, axis=-1)
        moved = moved + share * (compare(changed) - before)
    return moved


def _moved_grades_by(points, candidates, levels, a, b, rule=RANGE):
    """M[..., kind, side, k] = 1 where the voter at each point has the score k of a (side
    0) or b (side 1) and burying (kind 0: the frontrunner they rank lower) or
    compromising (kind 1: the one they rank higher, neither first) moves it, else 0."""
    rank = np.argsort(np.argsort(distances(points, candidates), axis=-1, kind="stable"), axis=-1)
    scores = np.eye(levels)[scored(points, candidates, levels, rule)]
    a_lower = (rank[..., a] > rank[..., b])[..., None]
    neither = ((rank[..., a] > 0) & (rank[..., b] > 0))[..., None]
    sides = [[a_lower, ~a_lower], [~a_lower & neither, a_lower & neither]]
    return np.stack([np.stack([np.where(sides[kind][side], scores[..., (a, b)[side], :], 0) for side in (0, 1)], -2)
                     for kind in (0, 1)], -3)


@pytest.mark.parametrize("rule", [RANGE, CLUSTER])
@pytest.mark.parametrize("levels", [2, 6])
def test_moved_grades_are_the_mean_of_each_rectangle(levels, rule):
    """In a rectangle with one ballot and one order at its corners, that voter's scores
    of the frontrunners that a strategy moves; in the others the mean of them at their
    sub x sub points. Those burying and compromising move are each at most all voters
    with that score."""
    xs, ys, sub, a, b = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4, 3, 0
    table = ballot_table(xs, ys, FIVE, levels, sub, rule, by_order=True)
    short = np.moveaxis(moved_grades(table, a, b), (-2, -1), (0, 1))
    assert short.shape == (40, 30, 2, 2, levels) and short.dtype == np.float32
    grid_points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    corners = np.concatenate([scored(grid_points, FIVE, levels, rule),
                              np.argsort(distances(grid_points, FIVE), axis=-1, kind="stable")], axis=-1)
    whole = ((corners[:-1, :-1] == corners[1:, :-1]) & (corners[:-1, :-1] == corners[:-1, 1:])
             & (corners[:-1, :-1] == corners[1:, 1:])).all(axis=-1)
    np.testing.assert_array_equal(short[whole], _moved_grades_by(grid_points, FIVE, levels, a, b, rule)[:-1, :-1][whole])
    at = (np.arange(sub) + 0.5) / sub
    inner_x = (xs[:-1, None] + at * np.diff(xs)[:, None]).ravel()
    inner_y = (ys[:-1, None] + at * np.diff(ys)[:, None]).ravel()
    inner = _moved_grades_by(np.stack(np.meshgrid(inner_x, inner_y, indexing="ij"), axis=-1), FIVE, levels, a, b, rule)
    mean = inner.reshape(40, sub, 30, sub, 2, 2, levels).mean(axis=(1, 3))
    np.testing.assert_allclose(short[~whole], mean[~whole], rtol=0, atol=1e-6)
    every = np.moveaxis(tabulate(table, GRADES), (0, 1), (-2, -1))[..., (a, b), :]
    assert (short.sum(axis=2) <= every + 1e-6).all() and short.max() == 1  # not vacuous
    with pytest.raises(ValueError, match="need a ballot table by order"):
        moved_grades(ballot_table(xs, ys, FIVE, levels, sub, rule), a, b)


@pytest.mark.parametrize("bury, compromise", [(0.4, 0.3), (1, 0), (0, 1)])
@pytest.mark.parametrize("rule", [RANGE, CLUSTER])
@pytest.mark.parametrize("levels", [2, 6])
def test_moved_comparisons_are_the_mean_of_each_rectangle(levels, rule, bury, compromise):
    """In a rectangle with one ballot and one order at its corners, the ballot moved; in the
    others the mean of the ballots moved at their sub x sub points. Only the pairs with a
    frontrunner move."""
    xs, ys, sub, a, b = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4, 3, 0
    table = ballot_table(xs, ys, FIVE, levels, sub, rule, by_order=True)
    short = np.moveaxis(moved_comparisons(table, a, b, bury, compromise), (0, 1), (-2, -1))
    grid_points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    corners = np.concatenate([scored(grid_points, FIVE, levels, rule),
                              np.argsort(distances(grid_points, FIVE), axis=-1, kind="stable")], axis=-1)
    whole = ((corners[:-1, :-1] == corners[1:, :-1]) & (corners[:-1, :-1] == corners[:-1, 1:])
             & (corners[:-1, :-1] == corners[1:, 1:])).all(axis=-1)
    np.testing.assert_allclose(short[whole], _moved_by(grid_points, FIVE, levels, a, b, bury, compromise,
                                                        rule)[:-1, :-1][whole], rtol=0, atol=1e-6)
    at = (np.arange(sub) + 0.5) / sub
    inner_x = (xs[:-1, None] + at * np.diff(xs)[:, None]).ravel()
    inner_y = (ys[:-1, None] + at * np.diff(ys)[:, None]).ravel()
    inner = _moved_by(np.stack(np.meshgrid(inner_x, inner_y, indexing="ij"), axis=-1), FIVE, levels, a, b, bury,
                      compromise, rule)
    mean = inner.reshape(40, sub, 30, sub, 5, 5).mean(axis=(1, 3))
    np.testing.assert_allclose(short[~whole], mean[~whole], rtol=0, atol=1e-6)
    others = [1, 2, 4]
    assert (short[..., np.ix_(others, others)[0], np.ix_(others, others)[1]] == 0).all()
    assert np.abs(short).max() > 0.1  # not vacuous


@pytest.mark.parametrize("rule", SCALES)
@pytest.mark.parametrize("levels", [2, 6])
@pytest.mark.parametrize("points, by", [(given, _given_by), (raised, _raised_by)], ids=["given", "raised"])
def test_given_and_raised_are_the_mean_of_each_rectangle(points, by, levels, rule):
    """A rectangle with one ballot and one order of distance at its four corners holds
    that ballot; the others the mean of the ballots at their sub x sub points."""
    xs, ys, sub = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4
    short = np.moveaxis(points(xs, ys, FIVE, levels, sub, rule), (0, 1), (-2, -1))
    assert short.shape == (40, 30, 5, 5) and short.dtype == np.float32
    grid_points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    ballots = scored(grid_points, FIVE, levels, rule)
    order = np.argsort(distances(grid_points, FIVE), axis=-1, kind="stable")
    corners = np.concatenate([ballots, order], axis=-1)
    whole = ((corners[:-1, :-1] == corners[1:, :-1]) & (corners[:-1, :-1] == corners[:-1, 1:])
             & (corners[:-1, :-1] == corners[1:, 1:])).all(axis=-1)
    assert 0.1 < whole.mean() < 1
    np.testing.assert_array_equal(short[whole], by(grid_points, FIVE, levels, rule)[:-1, :-1][whole])

    at = (np.arange(sub) + 0.5) / sub
    inner_x = (xs[:-1, None] + at * np.diff(xs)[:, None]).ravel()
    inner_y = (ys[:-1, None] + at * np.diff(ys)[:, None]).ravel()
    inner = by(np.stack(np.meshgrid(inner_x, inner_y, indexing="ij"), axis=-1), FIVE, levels, rule)
    mean = inner.reshape(40, sub, 30, sub, 5, 5).mean(axis=(1, 3))
    np.testing.assert_allclose(short[~whole], mean[~whole], rtol=0, atol=1e-6)
    assert (short[..., range(5), range(5)] == 0).all()


@pytest.mark.parametrize("by_order", [False, True])
@pytest.mark.parametrize("rule", [RANGE, CLUSTER])
def test_one_ballot_table_gives_every_share(rule, by_order):
    """The weights of every rectangle add up to 1, a table without the order holds at most
    one ballot where that by order does, and the sums over one table are what the
    separate functions give: the part not given and the comparisons from a table
    without the order, what burying takes back and compromising adds from one by order."""
    xs, ys = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31)
    table = ballot_table(xs, ys, FIVE, 6, 4, rule, by_order=by_order)
    assert table.shape == (40, 30) and table.start[-1] == len(table.codes) == len(table.weights)
    np.testing.assert_allclose(np.add.reduceat(table.weights, table.start[:-1]), 1, rtol=0, atol=1e-12)
    if by_order:
        coarse = ballot_table(xs, ys, FIVE, 6, 4, rule)
        assert (np.diff(table.start) >= np.diff(coarse.start)).all() and len(table.codes) > len(coarse.codes)
        np.testing.assert_array_equal(tabulate(table, GIVEN), given(xs, ys, FIVE, 6, 4, rule))
        np.testing.assert_array_equal(tabulate(table, RAISED), raised(xs, ys, FIVE, 6, 4, rule))
        np.testing.assert_allclose(tabulate(table, UNSCORED), unscored(xs, ys, FIVE, 6, 4, rule), rtol=0, atol=0.2)
    else:
        assert (table.orders == 0).all()
        np.testing.assert_array_equal(tabulate(table, UNSCORED), unscored(xs, ys, FIVE, 6, 4, rule))
        np.testing.assert_array_equal(tabulate(table, COMPARISONS), comparisons(xs, ys, FIVE, 6, 4, rule))
        for kind in (GIVEN, RAISED):
            with pytest.raises(ValueError, match="need a ballot table by order"):
                tabulate(table, kind)


def test_the_ballots_of_the_voter_grid_are_kept():
    """All score shares of a ballot, at any medians, come from one table, by order where
    burying or compromising read it."""
    model, candidates = Model("beta", 0.2, "rms"), CANDIDATES[5]
    shares_module._cached_ballots.cache_clear()
    for medians in (model.medians, np.linspace(0.1, 0.9, 7)):
        unscored_shares(candidates, model, 6, medians)
        score_comparison_shares(candidates, model, medians, 6)
    assert shares_module._cached_ballots.cache_info().misses == 1
    unscored_shares(candidates, model, 6, model.medians, by_order=True)
    buried_shares(candidates, model, model.medians, 6)
    raised_shares(candidates, model, model.medians, 6)
    assert shares_module._cached_ballots.cache_info().misses == 2
    lines = _grid_lines(model)
    np.testing.assert_array_equal(ballots(candidates, model, 6).codes,
                                  ballot_table(lines, lines, candidates, 6, GRID_SUB).codes)


def test_tie_weight_interpolates_ordinal_preferences_only_on_equal_scores():
    def encode(values):
        return sum(value << (BITS * candidate) for candidate, value in enumerate(values))

    def ranked(candidates):
        return sum(candidate << (BITS * position) for position, candidate in enumerate(candidates))

    table = BallotTable(
        (1, 1),
        np.array([0, 2]),
        np.array([encode((0, 0, 1)), encode((0, 1, 0))]),
        np.array([ranked((0, 1, 2)), ranked((2, 0, 1))]),
        np.array([0.25, 0.75]),
        3,
        1,
        True,
    )
    strict = tabulate(table, COMPARISONS)[..., 0, 0]
    half_weight = tabulate(table, COMPARISONS, 0.5)[..., 0, 0]
    full_weight = tabulate(table, COMPARISONS, 1.0)[..., 0, 0]
    np.testing.assert_allclose(strict, [[0, 0.75, 0.25], [-0.75, 0, -0.5], [-0.25, 0.5, 0]])
    np.testing.assert_allclose(half_weight, [[0, 0.875, -0.125], [-0.875, 0, -0.5], [0.125, 0.5, 0]])
    np.testing.assert_allclose(full_weight, [[0, 1, -0.5], [-1, 0, -0.5], [0.5, 0.5, 0]])
    unordered = BallotTable(table.shape, table.start, table.codes, table.orders, table.weights,
                            table.candidates, table.top, False)
    with pytest.raises(ValueError, match="ordinal score comparisons need a ballot table by order"):
        tabulate(unordered, COMPARISONS, 0.5)


def test_full_score_tie_weight_matches_ordinary_pairwise_margins():
    model = MODELS[0]
    candidates = CANDIDATES[5]
    actual = score_comparison_shares(candidates, model, model.medians, 6, power=1.5, tie_weight=1.0)
    pairwise = pairwise_shares(candidates, model)
    expected = pairwise - np.swapaxes(pairwise, -1, -2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-4)


def test_moved_comparisons_move_weighted_tie_preferences_with_strategies():
    code = 1 << BITS | 1 << (2 * BITS)
    order = 0 | 1 << BITS | 2 << (2 * BITS)
    table = BallotTable((1, 1), np.array([0, 1]), np.array([code]), np.array([order]), np.array([1.0]),
                        3, 2, True)
    moved = moved_comparisons(table, 1, 2, 0.4, 0.6, tie_weight=0.5)[..., 0, 0]
    np.testing.assert_allclose(moved[1, 2], 0.38)
    np.testing.assert_allclose(moved[1, 0], 0.9)
    np.testing.assert_allclose(moved[2, 0], 0.0)


def test_unscored_rejects_what_it_cannot_do():
    lines = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="score ballots need 2 to 15 candidates, got 1"):
        unscored(lines, lines, FIVE[:1], 3, 4)
    with pytest.raises(ValueError, match="score ballots need 2 to 15 candidates, got 16"):
        unscored(lines, lines, np.random.default_rng(0).random((16, 2)), 3, 4)
    for levels in (1, 17):
        with pytest.raises(ValueError, match=f"score ballots need 2 to 16 levels, got {levels}"):
            unscored(lines, lines, FIVE, levels, 4)


@pytest.mark.parametrize("rule", SCALES)
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


@pytest.mark.parametrize("rule", SCALES)
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
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=SAMPLED)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_graded_shares_match_sampled_voters(model):
    """The share of the voters who give each candidate each score, from the same grid:
    they add up to 1 and to the part not given, and match sampled voters. With two
    candidates only the top score and 0 are given, and the share of 0 is the part not
    given, down to the tiniest shares."""
    rng = np.random.default_rng(12)
    for n, levels in ((5, 2), (5, 6), (7, 11)):
        shares = graded_shares(CANDIDATES[n], model, levels, model.medians)
        assert shares.shape == (model.nodes, model.nodes, n, levels) and (shares >= 0).all()
        np.testing.assert_allclose(shares.sum(axis=-1), 1, rtol=0, atol=1e-9)
        unscored_ = unscored_shares(CANDIDATES[n], model, levels, model.medians)
        np.testing.assert_allclose(shares @ (levels - 1 - np.arange(levels)) / (levels - 1), unscored_,
                                   rtol=1e-9, atol=1e-300)
        for node in NODES:
            ballots_ = scored(_sample(model, node, rng), CANDIDATES[n], levels)
            sampled = (ballots_[..., None] == np.arange(levels)).mean(axis=0)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=SAMPLED)
    two = graded_shares(CANDIDATES[2], model, 6, model.medians)
    assert (two[..., 1:-1] == 0).all()
    np.testing.assert_allclose(two[..., 0], unscored_shares(CANDIDATES[2], model, 6, model.medians), rtol=1e-12, atol=0)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_candidates_are_scored_by_their_first_choices(model):
    """The closer one gets the top score and the other 0, at any number of levels and by
    every rule: the part of the top score not given is the share of the voters closer to
    the other, and the same share for all of them."""
    not_first = 1 - first_choice_shares(CANDIDATES[2], model)
    approval = unscored_shares(CANDIDATES[2], model, 2, model.medians)
    for levels, rule in product((2, 6, 16), SCALES):
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


@pytest.mark.parametrize("rule", SCALES)
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
def test_d21_ordinal_approval_is_the_exact_share_among_the_closest(model):
    """Without minus votes the share giving c a plus vote is that with c among the
    max_positive closest, exact from the cells of the ranking."""
    for n in (5, 7):
        candidates = CANDIDATES[n]
        for plus in (1, 2, n - 1):
            shares = 1 - unscored_shares(candidates, model, 2, model.medians, D21_ORDINAL, votes=(plus, 0, 0))
            np.testing.assert_allclose(shares, _top_shares(candidates, model, plus), rtol=0, atol=TOLERANCE)


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
            assert np.abs(shares - exact).max() <= 0.5 / (levels - 1) + SAMPLED
        assert np.abs(shares - exact).max() < 0.02 + SAMPLED / 2  # 16 levels


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_buried_shares_match_sampled_voters(model):
    """The part of the top score that the voters who rank e above c give c."""
    rng = np.random.default_rng(9)
    for n, levels in ((5, 2), (5, 6), (7, 11)):
        shares = buried_shares(CANDIDATES[n], model, model.medians, levels)
        assert shares.shape == (model.nodes, model.nodes, n, n)
        for node in NODES:
            sampled = _given_by(_sample(model, node, rng), CANDIDATES[n], levels).mean(axis=0) / (levels - 1)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=SAMPLED)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_burying_takes_back_at_most_what_was_given_by_those_who_rank_another_higher(model):
    """What c gets from the voters who rank e above it is at most their share, and at
    most all c gets; with two candidates the one ranked lower is the farthest and gets
    nothing, by every rule and at any number of levels."""
    for levels, rule in product((2, 6), SCALES):
        assert np.abs(buried_shares(CANDIDATES[2], model, model.medians, levels, rule)).max() < 1e-12
    candidates = CANDIDATES[5]
    above = pairwise_shares(candidates, model)  # [c, e]: c ranked above e
    for levels in (2, 6):
        shares = buried_shares(candidates, model, model.medians, levels)
        assert (shares >= 0).all()
        assert (shares <= np.swapaxes(above, -1, -2) + TOLERANCE).all()
        given_all = 1 - unscored_shares(candidates, model, levels, model.medians)
        assert (shares <= given_all[..., :, None] + TOLERANCE).all()


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_raised_shares_match_sampled_voters(model):
    """The part of the top score that the voters who rank c above e, but neither first, do
    not give c."""
    rng = np.random.default_rng(10)
    for n, levels in ((5, 2), (5, 6), (7, 11)):
        shares = raised_shares(CANDIDATES[n], model, model.medians, levels)
        assert shares.shape == (model.nodes, model.nodes, n, n)
        for node in NODES:
            sampled = _raised_by(_sample(model, node, rng), CANDIDATES[n], levels).mean(axis=0) / (levels - 1)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=SAMPLED)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_compromising_adds_at_most_what_was_not_given_by_those_who_rank_neither_first(model):
    """What compromising adds to c over e is at most the share of the voters who rank c
    above e, and at most what all voters do not give c; with two candidates every voter
    has one of them first, and it adds nothing."""
    for levels, rule in product((2, 6), SCALES):
        assert np.abs(raised_shares(CANDIDATES[2], model, model.medians, levels, rule)).max() < 1e-12
    candidates = CANDIDATES[5]
    above = pairwise_shares(candidates, model)
    for levels in (2, 6):
        shares = raised_shares(candidates, model, model.medians, levels)
        assert (shares >= 0).all()
        assert (shares <= above + TOLERANCE).all()
        not_given = unscored_shares(candidates, model, levels, model.medians)
        assert (shares <= not_given[..., :, None] + TOLERANCE).all()


@pytest.mark.parametrize("model", MODELS[:2] + MODELS[3:4], ids=_ids)
def test_the_score_grid_gives_the_shares_of_the_points_of_a_pair(model):
    """What burying takes back and compromising adds are those of buried_shares and
    raised_shares at the points asked for, from the rows and columns they need; the
    shares of every score that they move, at most those of all voters, and how much the
    comparisons move are those of sampled voters moved one at a time."""
    candidates, levels, a, b = CANDIDATES[5], 6, 3, 1
    grid = ScoreGrid(candidates, model, model.medians, levels)
    cells = np.array([(5, 9), (24, 24), (40, 3), (5, 3), (24, 9)])
    taken = buried_shares(candidates, model, model.medians, levels)[cells[:, 0], cells[:, 1]]
    added = raised_shares(candidates, model, model.medians, levels)[cells[:, 0], cells[:, 1]]
    np.testing.assert_allclose(grid.taken(a, b, cells), np.stack([taken[:, a, b], taken[:, b, a]], -1), atol=1e-12)
    np.testing.assert_allclose(grid.added(a, b, cells), np.stack([added[:, a, b], added[:, b, a]], -1), atol=1e-12)
    # of the same table by order, as the voters of a strategy have them (regions.voters)
    graded_ = graded_shares(candidates, model, levels, model.medians, by_order=True)[cells[:, 0], cells[:, 1]][:, (a, b)]
    moving = grid.moved_grades(a, b, cells)
    assert moving.shape == (2, 5, 2, levels) and (moving.sum(axis=0) <= graded_ + 1e-12).all()
    rng = np.random.default_rng(11)
    for k, node in enumerate(cells[:2]):
        sampled = _moved_grades_by(_sample(model, tuple(node), rng), candidates, levels, a, b).mean(axis=0)
        np.testing.assert_allclose(moving[:, k], sampled, rtol=0, atol=SAMPLED)
    moved = grid.moved(a, b, 0.4, 0.3, cells)
    assert moved.shape == (5, 5, 5)
    np.testing.assert_allclose(moved, -np.swapaxes(moved, -1, -2), atol=1e-12)
    for k, node in enumerate(cells[:2]):
        sampled = _moved_by(_sample(model, tuple(node), rng), candidates, levels, a, b, 0.4, 0.3)
        np.testing.assert_allclose(moved[k], sampled.mean(axis=0), rtol=0, atol=SAMPLED)
