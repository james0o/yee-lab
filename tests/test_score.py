"""Checks of the score ballots (score.py) and their shares (margin/shares.py).

The ballot of one voter is checked from its definition: by every rule the closest
candidate gets the top score and the farthest 0. RANGE gives the others the score in
proportion to where their distance is between the two, rounded like np.round; with two
levels it approves the candidates closer than halfway, which for three candidates is the
approval ballot. AVG puts the mean distance in the middle of that scale; with two levels
it is the approval ballot of the mean, for any number of candidates. DHONDT shares the steps out among the gaps by the divisor method of its
delta (D'Hondt at 1), which is checked by the condition every apportionment of that
method meets and no other does; with two levels it is the approval ballot of the largest
gap, for any number of candidates and any delta, and it does not become Borda with more
levels than candidates. A lower delta moves steps from the large gaps to the small ones.
The compiled ballot of the grid must be that of the definition.

The shares come from the voter grid of the approval shares (test_approval.py). They are
compared with exact shares where there are some (two candidates get the top score from
their first choices and 0 from the others; three candidates and two levels are the
approval shares, and so are any candidates and two levels of AVG and DHONDT) and with
sampled voters elsewhere.
"""

from itertools import product

import numpy as np
import pytest

from yeelab import ranking_cells
from yeelab.approval import AVG as MEAN_CUT, CUTS, GAP, approved, distances
from yeelab.margin.shares import Model, first_choice_shares, unapproved_shares, unscored_shares
from yeelab.score import AVG, DELTA, DHONDT, MAX_CANDIDATES, MAX_LEVELS, RANGE, RULES, scored, unscored

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
TOLERANCE = 2e-3  # the grid shares against exact ones, as in test_approval.py


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
    """For any number of candidates; at the mean itself the half is rounded to 0, as
    approval.AVG does not approve a tie."""
    rng = np.random.default_rng(12)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((5000, 2))
        np.testing.assert_array_equal(scored(points, candidates, 2, AVG), approved(points, candidates, MEAN_CUT))
    assert _ballot([0, 1, 2], 2, AVG) == [1, 0, 0]


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


@pytest.mark.parametrize("cut", CUTS)
def test_two_levels_are_the_approval_ballot_of_three_candidates_only(cut):
    """The middle one of three is closer than halfway where it is closer to the closest
    than to the farthest: the approval ballot at either cut. Four have other ballots."""
    rng = np.random.default_rng(4)
    points = rng.random((5000, 2))
    for n in (2, 3):
        candidates = rng.random((n, 2))
        np.testing.assert_array_equal(scored(points, candidates, 2), approved(points, candidates, cut))
    candidates = rng.random((4, 2))
    assert (scored(points, candidates, 2) != approved(points, candidates, cut)).any()


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
                                      approved(points, candidates, GAP))
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


def test_unknown_rules_are_rejected():
    for rule in ("gap", "dh", None):
        with pytest.raises(ValueError, match=f"unknown rule {rule!r}; choose from range, dhondt, avg"):
            scored(np.zeros(2), FIVE, 3, rule)
        with pytest.raises(ValueError, match=f"unknown rule {rule!r}; choose from range, dhondt, avg"):
            unscored(np.linspace(0, 1, 5), np.linspace(0, 1, 5), FIVE, 3, 4, rule)


def test_delta_must_be_above_zero():
    for delta in (0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match=f"delta must be a number above 0, got {delta!r}"):
            scored(np.zeros(2), FIVE, 3, DHONDT, delta)
        with pytest.raises(ValueError, match=f"delta must be a number above 0, got {delta!r}"):
            unscored(np.linspace(0, 1, 5), np.linspace(0, 1, 5), FIVE, 3, 4, DHONDT, delta)

# ---------------------------------------------------------------- Grid


@pytest.mark.parametrize(("rule", "delta"), [(RANGE, DELTA), (AVG, DELTA)] + [(DHONDT, delta) for delta in DELTAS])
@pytest.mark.parametrize("n", CANDIDATES)
def test_the_compiled_ballot_is_the_definition(n, rule, delta):
    """A rectangle too small for a border to cross holds the ballot of the voter at it,
    as the points below the top score, for candidates and voters in and around the
    square."""
    rng = np.random.default_rng(n)
    points = 2 * rng.random((200, 2)) - 0.5
    for levels in LEVELS:
        short = np.stack([unscored([x, x + 1e-9], [y, y + 1e-9], CANDIDATES[n], levels, 2, rule, delta)[:, 0, 0]
                          for x, y in points])
        np.testing.assert_array_equal(short, levels - 1 - scored(points, CANDIDATES[n], levels, rule, delta))


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
    """The closer one gets the top score and the other 0, at any number of levels: the
    part of the top score not given is the share of the voters closer to the other, and
    that of the approval ballots."""
    not_first = 1 - first_choice_shares(CANDIDATES[2], model)
    unapproved = unapproved_shares(CANDIDATES[2], model, "half", model.medians)
    for levels, rule in product((2, 6, 16), RULES):
        unscored_ = unscored_shares(CANDIDATES[2], model, levels, model.medians, rule)
        np.testing.assert_allclose(unscored_, not_first, rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(unscored_, unapproved, rtol=1e-12, atol=0)  # the tiny ones too


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_three_candidates_and_two_levels_have_the_approval_shares(model, cut):
    np.testing.assert_allclose(unscored_shares(CANDIDATES[3], model, 2, model.medians),
                               unapproved_shares(CANDIDATES[3], model, cut, model.medians),
                               rtol=1e-9, atol=0)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_levels_of_avg_have_the_shares_of_the_mean(model):
    for n in (3, 5, 7):
        np.testing.assert_allclose(unscored_shares(CANDIDATES[n], model, 2, model.medians, AVG),
                                   unapproved_shares(CANDIDATES[n], model, MEAN_CUT, model.medians),
                                   rtol=1e-9, atol=0)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_levels_of_dhondt_have_the_shares_of_the_largest_gap(model):
    for n in (3, 5, 7):
        np.testing.assert_allclose(unscored_shares(CANDIDATES[n], model, 2, model.medians, DHONDT),
                                   unapproved_shares(CANDIDATES[n], model, GAP, model.medians),
                                   rtol=1e-9, atol=0)


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
