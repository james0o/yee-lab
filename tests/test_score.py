"""Checks of the score ballots (score.py) and their shares (margin/shares.py).

The ballot of one voter is checked from its definition: the closest candidate gets the
top score and the farthest 0, the others the score in proportion to where their distance
is between the two, rounded like np.round. With two levels it approves the candidates
closer than halfway, which for three candidates is the approval ballot. The compiled
ballot of the grid must be that of the definition.

The shares come from the voter grid of the approval shares (test_approval.py). They are
compared with exact shares where there are some (two candidates get the top score from
their first choices and 0 from the others; three candidates and two levels are the
approval shares) and with sampled voters elsewhere.
"""

import numpy as np
import pytest

from yeelab import ranking_cells
from yeelab.approval import CUTS, approved, distances
from yeelab.margin.shares import Model, first_choice_shares, unapproved_shares, unscored_shares
from yeelab.score import MAX_CANDIDATES, MAX_LEVELS, scored, unscored

FIVE = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
CANDIDATES = {2: np.random.default_rng(5).random((2, 2)), 3: np.random.default_rng(3).random((3, 2)),
              4: np.random.default_rng(6).random((4, 2)), 5: FIVE,
              6: np.random.default_rng(4).random((6, 2)), 7: np.random.default_rng(1).random((7, 2))}
LEVELS = [2, 3, 6, 11, 16]
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


def _ballot(distances_, levels):
    return scored(np.zeros(2), _at(distances_), levels).tolist()

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
    assert _ballot([3, 3, 3], 6) == [5, 5, 5]
    assert _ballot([0.5, 0.5], 2) == [1, 1]


def test_the_closest_gets_the_top_score_and_the_farthest_none():
    rng = np.random.default_rng(1)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r, rows = distances(points, candidates), np.arange(2000)
        for levels in LEVELS:
            ballot = scored(points, candidates, levels)
            assert ballot.dtype == np.int64 and ballot.shape == (2000, n)
            assert (ballot[rows, r.argmin(axis=-1)] == levels - 1).all()
            assert (ballot[rows, r.argmax(axis=-1)] == 0).all()
            # a closer candidate never gets a lower score
            by_distance = np.take_along_axis(ballot, r.argsort(axis=-1), axis=-1)
            assert (np.diff(by_distance, axis=-1) <= 0).all()


def test_two_candidates_get_the_top_score_and_none_at_any_number_of_levels():
    rng = np.random.default_rng(2)
    candidates, points = rng.random((2, 2)), rng.random((2000, 2))
    closest = distances(points, candidates).argmin(axis=-1)
    for levels in LEVELS:
        np.testing.assert_array_equal(scored(points, candidates, levels),
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


def test_the_ballot_does_not_change_with_the_scale():
    rng = np.random.default_rng(5)
    candidates, points = rng.random((7, 2)), rng.random((3000, 2))
    for levels in (2, 6):
        scaled = scored(7 * points - 2, 7 * candidates - 2, levels)
        assert (scaled == scored(points, candidates, levels)).mean() > 0.9999  # but at a border, to rounding

# ---------------------------------------------------------------- Grid


@pytest.mark.parametrize("n", CANDIDATES)
def test_the_compiled_ballot_is_the_definition(n):
    """A rectangle too small for a border to cross holds the ballot of the voter at it,
    as the points below the top score, for candidates and voters in and around the
    square."""
    rng = np.random.default_rng(n)
    points = 2 * rng.random((200, 2)) - 0.5
    for levels in LEVELS:
        short = np.stack([unscored([x, x + 1e-9], [y, y + 1e-9], CANDIDATES[n], levels, 2)[:, 0, 0]
                          for x, y in points])
        np.testing.assert_array_equal(short, levels - 1 - scored(points, CANDIDATES[n], levels))


@pytest.mark.parametrize("levels", [2, 4, 11])
def test_unscored_is_the_mean_of_each_rectangle(levels):
    """A rectangle with one ballot at its four corners holds that ballot; the others the
    mean of the ballots at their sub x sub points. On lines that are not evenly spaced,
    and with different ones along x and y."""
    xs, ys, sub = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4
    short = np.moveaxis(unscored(xs, ys, FIVE, levels, sub), 0, -1)
    assert short.shape == (40, 30, 5) and short.dtype == np.float32

    def below(points):
        return levels - 1 - scored(points, FIVE, levels)

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


def test_the_most_candidates_and_levels_fit_a_ballot():
    """15 candidates up to 15 points below the top score: the highest bits of the code."""
    assert (MAX_CANDIDATES, MAX_LEVELS) == (15, 16)
    rng = np.random.default_rng(6)
    candidates, points = rng.random((MAX_CANDIDATES, 2)), rng.random((100, 2))
    short = np.stack([unscored([x, x + 1e-9], [y, y + 1e-9], candidates, MAX_LEVELS, 2)[:, 0, 0]
                      for x, y in points])
    np.testing.assert_array_equal(short, MAX_LEVELS - 1 - scored(points, candidates, MAX_LEVELS))
    assert short.max() == MAX_LEVELS - 1

# ---------------------------------------------------------------- Shares


def _sample(model, node, rng):
    """Voters of one node, (SAMPLES, 2)."""
    i, j = node
    if model.distribution == "beta":
        params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
        return np.column_stack([rng.beta(*params[i], SAMPLES), rng.beta(*params[j], SAMPLES)])
    return rng.normal(model.medians[[i, j]], model.sigma, (SAMPLES, 2))


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_shares_match_sampled_voters(model):
    rng = np.random.default_rng(7)
    for n, levels in ((5, 2), (5, 6), (7, 11)):
        shares = unscored_shares(CANDIDATES[n], model, levels, model.medians)
        assert shares.shape == (model.nodes, model.nodes, n)
        assert (shares >= 0).all() and (shares <= 1 + 1e-12).all()
        for node in NODES:
            ballots = scored(_sample(model, node, rng), CANDIDATES[n], levels)
            sampled = 1 - ballots.mean(axis=0) / (levels - 1)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=5e-3)  # 6 standard errors


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_candidates_are_scored_by_their_first_choices(model):
    """The closer one gets the top score and the other 0, at any number of levels: the
    part of the top score not given is the share of the voters closer to the other, and
    that of the approval ballots."""
    not_first = 1 - first_choice_shares(CANDIDATES[2], model)
    unapproved = unapproved_shares(CANDIDATES[2], model, "half", model.medians)
    for levels in (2, 6, 16):
        unscored_ = unscored_shares(CANDIDATES[2], model, levels, model.medians)
        np.testing.assert_allclose(unscored_, not_first, rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(unscored_, unapproved, rtol=1e-12, atol=0)  # the tiny ones too


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_three_candidates_and_two_levels_have_the_approval_shares(model, cut):
    np.testing.assert_allclose(unscored_shares(CANDIDATES[3], model, 2, model.medians),
                               unapproved_shares(CANDIDATES[3], model, cut, model.medians),
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
