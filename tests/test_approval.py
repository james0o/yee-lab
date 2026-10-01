"""Checks of the approval ballots (approval.py) and their shares (margin/shares.py).

The ballot of one voter is checked from its definition: the closest candidate is
approved and the farthest is not, HALF approves half of the candidates (the middle one
of an odd number by its two gaps), GAP those above the largest gap of the distances, and
for three candidates the two are the same ballot. The compiled ballot of the grid must
be that of the definition.

The shares come from a grid of voters, so they are approximate. They are compared with
exact shares where there are some (HALF with an even number of candidates is the top
half of the ranking, and two candidates are approved by their first choices) and with
sampled voters elsewhere. They are computed as the share who do not approve, which must
stay exact where it is tiny: the lead between two candidates nearly all voters approve.
"""

import numpy as np
import pytest
from scipy.special import ndtr

from yeelab import ranking_cells
from yeelab.approval import CUTS, GAP, HALF, approved, coverage, distances
from yeelab.margin.shares import (
    APPROVAL_CELLS,
    Model,
    _voter_grid,
    first_choice_shares,
    pairwise_shares,
    ranking_shares,
    unapproved_shares,
)

FIVE = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
CANDIDATES = {2: np.random.default_rng(5).random((2, 2)), 3: np.random.default_rng(3).random((3, 2)),
              4: np.random.default_rng(6).random((4, 2)), 5: FIVE,
              6: np.random.default_rng(4).random((6, 2)), 7: np.random.default_rng(1).random((7, 2))}
MODELS = [Model("beta", 0.2, "rms"), Model("beta", 0.05, "rms"), Model("beta", 0.35, "mean_abs"),
          Model("normal", 0.2), Model("normal", 0.05)]
NODES = [(5, -9), (24, 24), (-5, 10), (0, 0)]  # of the nodes per axis, where the shares are sampled
SAMPLES = 400_000
# The grid shares against exact ones: 8e-4 seen for the narrowest voters, 2e-4 otherwise.
TOLERANCE = 2e-3


def _ids(value):
    return f"{value.distribution}-{value.deviation}" if isinstance(value, Model) else str(value)


def _at(distances_):
    """Candidates at the given distances from a voter at the origin, each in its own
    direction."""
    distances_ = np.asarray(distances_, dtype=np.float64)
    angles = np.linspace(0, 5, len(distances_))
    return distances_[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])


def _ballot(distances_, cut):
    return approved(np.zeros(2), _at(distances_), cut).tolist()


def approval_shares(candidates, model, cut):
    """Share of the voters of each node who approve each candidate, (N, N, C)."""
    return 1 - unapproved_shares(candidates, model, cut, model.medians)


def _top_shares(candidates, model, count):
    """Share of the voters with each candidate among their `count` closest: exact, from
    the cells of the ranking."""
    rankings, shares = ranking_shares(candidates, model)
    top = np.zeros((len(rankings), len(candidates)))
    np.put_along_axis(top, rankings[:, :count].astype(np.int64), 1.0, axis=1)
    return shares @ top

# ---------------------------------------------------------------- One voter


def test_a_voter_on_a_candidate_approves_only_that_one_at_the_gap():
    """Squared distances 0, 0.35, 0.45, 0.55 and 1: the distances have the gaps 0.59,
    0.08, 0.07 and 0.26, so the largest is the first. Half of five is two, and the
    middle candidate is closer to the fourth (0.07) than to the second (0.08)."""
    by_distance = np.sqrt([0.0, 0.35, 0.45, 0.55, 1.0])
    assert _ballot(by_distance, GAP) == [True, False, False, False, False]
    assert _ballot(by_distance, HALF) == [True, True, False, False, False]


def test_one_close_candidate_and_four_far_ones():
    """Distances 0.5, 0.55, 0.41, 0.47 and 0.1, in order 0.1, 0.41, 0.47, 0.5, 0.55: the
    largest gap is after the closest. The middle one (0.47) is closer to the fourth
    (0.03) than to the second (0.06), so HALF approves two."""
    far = [0.5, 0.55, 0.41, 0.47, 0.1]
    assert _ballot(far, GAP) == [False, False, False, False, True]
    assert _ballot(far, HALF) == [False, False, True, False, True]


def test_the_middle_candidate_goes_with_the_closer_neighbour():
    assert _ballot([0.1, 0.2, 0.25, 0.5, 0.6], HALF) == [True, True, True, False, False]  # 0.05 < 0.25
    assert _ballot([0.1, 0.2, 0.45, 0.5, 0.6], HALF) == [True, True, False, False, False]  # 0.25 > 0.05
    assert _ballot([0.1, 0.2, 0.3, 0.4, 0.5], HALF)[:3] == [True, True, _ballot([0.1, 0.2, 0.3], GAP)[1]]
    assert _ballot([0.1, 0.3, 0.5], HALF) == [True, False, False]  # a tie: it is not approved
    assert _ballot([0.1, 0.3, 0.5], GAP) == [True, False, False]  # a tie: the first gap


@pytest.mark.parametrize("cut", CUTS)
def test_the_closest_is_approved_and_the_farthest_is_not(cut):
    rng = np.random.default_rng(0)
    for n in range(2, 9):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        r, ballot = distances(points, candidates), approved(points, candidates, cut)
        rows = np.arange(len(points))
        assert ballot[rows, r.argmin(axis=-1)].all()
        assert not ballot[rows, r.argmax(axis=-1)].any()
        # the approved candidates are the closest ones
        assert (np.where(ballot, r, -np.inf).max(axis=-1) < np.where(ballot, np.inf, r).min(axis=-1)).all()


def test_half_approves_half_of_the_candidates():
    rng = np.random.default_rng(1)
    for n in range(2, 9):
        count = approved(rng.random((2000, 2)), rng.random((n, 2)), HALF).sum(axis=-1)
        if n % 2 == 0:
            assert (count == n // 2).all()
        else:
            assert set(count.tolist()) == {n // 2, n // 2 + 1}


def test_gap_approves_from_one_candidate_to_all_but_one():
    rng = np.random.default_rng(2)
    count = approved(rng.random((20000, 2)), rng.random((5, 2)), GAP).sum(axis=-1)
    assert set(count.tolist()) == {1, 2, 3, 4}


def test_for_two_and_three_candidates_both_cuts_are_the_same_ballot():
    rng = np.random.default_rng(3)
    for n in (2, 3):
        candidates, points = rng.random((n, 2)), rng.random((5000, 2))
        np.testing.assert_array_equal(approved(points, candidates, GAP), approved(points, candidates, HALF))


@pytest.mark.parametrize("cut", CUTS)
def test_the_ballot_does_not_change_with_the_scale(cut):
    rng = np.random.default_rng(4)
    candidates, points = rng.random((6 if cut == GAP else 5, 2)), rng.random((3000, 2))
    np.testing.assert_array_equal(approved(7 * points - 2, 7 * candidates - 2, cut),
                                  approved(points, candidates, cut))

# ---------------------------------------------------------------- Grid


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("n", CANDIDATES)
def test_the_compiled_ballot_is_the_definition(n, cut):
    """A rectangle too small for a border to cross is covered by the ballot of the voter
    at it, for candidates and voters in and around the square."""
    rng = np.random.default_rng(n)
    points = 2 * rng.random((200, 2)) - 0.5
    cover = np.stack([coverage([x, x + 1e-9], [y, y + 1e-9], CANDIDATES[n], cut, 2)[:, 0, 0]
                      for x, y in points])
    np.testing.assert_array_equal(cover, approved(points, CANDIDATES[n], cut))


@pytest.mark.parametrize("cut", CUTS)
def test_coverage_is_the_share_of_each_rectangle(cut):
    """A rectangle with one ballot at its four corners is covered by that ballot; the
    others by the mean of the ballots at their sub x sub points. On lines that are not
    evenly spaced, and with different ones along x and y."""
    xs, ys, sub = np.linspace(0, 1, 41) ** 1.3, np.linspace(-0.2, 1.1, 31), 4
    cover = np.moveaxis(coverage(xs, ys, FIVE, cut, sub), 0, -1)
    assert cover.shape == (40, 30, 5) and cover.dtype == np.float32

    corners = approved(np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1), FIVE, cut)
    whole = ((corners[:-1, :-1] == corners[1:, :-1]) & (corners[:-1, :-1] == corners[:-1, 1:])
             & (corners[:-1, :-1] == corners[1:, 1:])).all(axis=-1)
    assert 0.5 < whole.mean() < 1
    np.testing.assert_array_equal(cover[whole], corners[:-1, :-1][whole])

    at = (np.arange(sub) + 0.5) / sub
    inner_x = (xs[:-1, None] + at * np.diff(xs)[:, None]).ravel()
    inner_y = (ys[:-1, None] + at * np.diff(ys)[:, None]).ravel()
    ballots = approved(np.stack(np.meshgrid(inner_x, inner_y, indexing="ij"), axis=-1), FIVE, cut)
    mean = ballots.reshape(40, sub, 30, sub, 5).mean(axis=(1, 3))
    np.testing.assert_allclose(cover[~whole], mean[~whole], rtol=0, atol=1e-7)
    assert ((cover[~whole] > 0) & (cover[~whole] < 1)).any()


def test_coverage_rejects_what_it_cannot_do():
    lines = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="unknown cut 0.5; choose from half, gap"):
        coverage(lines, lines, FIVE, 0.5, 4)
    with pytest.raises(ValueError, match="approval ballots need 2 to 62 candidates, got 1"):
        coverage(lines, lines, FIVE[:1], HALF, 4)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_the_voter_grid_holds_all_voters(model):
    """Equal cells across the square, each node's voters summing to 1 over the cells,
    and the cells of both models reaching where their voters are."""
    lines, mass = _voter_grid(model, model.medians)
    assert (np.diff(lines) > 0).all() and mass.shape == (model.nodes, len(lines) - 1)
    inside = lines[(lines > 0.01) & (lines < 0.99)]
    np.testing.assert_allclose(np.diff(inside), 1 / APPROVAL_CELLS, atol=1e-12)
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


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_shares_match_sampled_voters(model, cut):
    rng = np.random.default_rng(7)
    for n in (5, 7):
        shares = approval_shares(CANDIDATES[n], model, cut)
        assert shares.shape == (model.nodes, model.nodes, n)
        for node in NODES:
            sampled = approved(_sample(model, node, rng), CANDIDATES[n], cut).mean(axis=0)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=5e-3)  # 6 standard errors


@pytest.mark.parametrize("model", MODELS, ids=_ids)
@pytest.mark.parametrize("n", [2, 4, 6])
def test_half_of_an_even_number_is_the_top_half_of_the_ranking(n, model):
    np.testing.assert_allclose(approval_shares(CANDIDATES[n], model, HALF),
                               _top_shares(CANDIDATES[n], model, n // 2), rtol=0, atol=TOLERANCE)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_candidates_are_approved_by_their_first_choices(model):
    for cut in CUTS:
        np.testing.assert_allclose(approval_shares(CANDIDATES[2], model, cut),
                                   first_choice_shares(CANDIDATES[2], model), rtol=0, atol=TOLERANCE)


@pytest.mark.parametrize("sigma_model", [Model("normal", 0.05), Model("normal", 0.2)], ids=_ids)
def test_the_share_not_approving_is_exact_where_it_is_tiny(sigma_model):
    """Two candidates: a voter does not approve the farther one, and for normal voters
    the share closer to the other is Phi of the distance to their bisector. The grid
    share must follow it in relative terms, far below the rounding of 1 - share (6%
    seen at 1e-36: the border within a rectangle is only a share of the rectangle)."""
    medians = np.linspace(0.01, 0.99, 50)
    unapproved = unapproved_shares(CANDIDATES[2], sigma_model, HALF, medians)
    diff = CANDIDATES[2][1] - CANDIDATES[2][0]
    threshold = (CANDIDATES[2][1] @ CANDIDATES[2][1] - CANDIDATES[2][0] @ CANDIDATES[2][0]) / 2
    mean = diff[0] * medians[:, None] + diff[1] * medians[None, :]
    z = (threshold - mean) / (sigma_model.sigma * np.linalg.norm(diff))  # to the bisector
    np.testing.assert_allclose(unapproved[..., 1], ndtr(z), rtol=0.1, atol=0)  # closer to 0
    np.testing.assert_allclose(unapproved[..., 0], ndtr(-z), rtol=0.1, atol=0)
    if sigma_model.deviation == 0.05:
        assert unapproved.min() < 1e-30


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_three_candidates_have_the_same_shares_at_both_cuts(model):
    np.testing.assert_array_equal(approval_shares(CANDIDATES[3], model, GAP),
                                  approval_shares(CANDIDATES[3], model, HALF))


@pytest.mark.parametrize("model", MODELS, ids=_ids)
@pytest.mark.parametrize("n", [5, 7])
def test_shares_are_between_those_of_fewer_and_more_approved(n, model):
    """HALF approves the closest n // 2 and at most one more; GAP the closest one and
    at most all but the farthest. The shares add up to the mean number approved."""
    candidates, half = CANDIDATES[n], n // 2
    shares = approval_shares(candidates, model, HALF)
    assert (shares >= _top_shares(candidates, model, half) - TOLERANCE).all()
    assert (shares <= _top_shares(candidates, model, half + 1) + TOLERANCE).all()
    approvals = shares.sum(axis=-1)
    assert (approvals >= half - n * TOLERANCE).all() and (approvals <= half + 1 + n * TOLERANCE).all()

    shares = approval_shares(candidates, model, GAP)
    assert (shares >= _top_shares(candidates, model, 1) - TOLERANCE).all()
    assert (shares <= _top_shares(candidates, model, n - 1) + TOLERANCE).all()
    assert (shares >= -1e-12).all() and (shares <= 1 + 1e-12).all()
