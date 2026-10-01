"""Checks of the approval ballots (approval.py) and their shares (margin/shares.py).

The ballot of one voter is checked from its definition: the closest candidate is
approved and the farthest is not, a threshold of 1 is plurality and one near 0 all but
the farthest, and for three candidates the largest gap is the threshold 1/2.

The polygons the shares are summed over must hold exactly the voters who approve each
candidate: random points are in a polygon of a candidate if and only if the ballot at
that point approves it. The shares themselves are compared with sampled voters, and
with the first-choice and last-place shares they become at the two ends of the threshold.
"""

import numpy as np
import pytest

from yeelab import ranking_cells
from yeelab.approval import GAP, approved, half_plane, utilities
from yeelab.margin.shares import (
    Model,
    _voter_box,
    approval_polygons,
    approval_shares,
    first_choice_shares,
    ranking_shares,
)

FIVE = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
CANDIDATES = {2: np.random.default_rng(5).random((2, 2)), 3: np.random.default_rng(3).random((3, 2)),
              5: FIVE, 7: np.random.default_rng(1).random((7, 2))}
MODELS = [Model("beta", 0.2, "rms"), Model("beta", 0.35, "mean_abs"), Model("normal", 0.2)]
CUTS = [0.25, 0.5, 0.9, GAP]
NODES = [(5, 40), (24, 24), (44, 10)]  # of the 49 per axis, where the shares are sampled
SAMPLES = 400_000


def _ids(value):
    return f"{value.distribution}-{value.deviation}" if isinstance(value, Model) else str(value)


def _tolerance(model):
    """As in test_shares.py: Beta edges are cut differently (one quadrature each),
    normal edges are exact."""
    return 2e-4 if model.distribution == "beta" else 1e-9


def _last_place_shares(candidates, model):
    rankings, shares = ranking_shares(candidates, model)
    return shares @ np.eye(len(candidates))[rankings[:, -1]]

# ---------------------------------------------------------------- One voter


def test_one_close_candidate_is_the_only_one_approved():
    """Distances 0.5, 0.55, 0.41, 0.47 and 0.1: the jump is after the closest candidate,
    which is also the only one above the middle between the closest and the farthest."""
    distances = np.array([0.5, 0.55, 0.41, 0.47, 0.1])
    angles = np.linspace(0, 5, len(distances))
    candidates = distances[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    for cut in (0.5, GAP):
        np.testing.assert_array_equal(approved(np.zeros(2), candidates, cut),
                                      [False, False, False, False, True])


@pytest.mark.parametrize("cut", [0.01, *CUTS, 1.0], ids=str)
def test_the_closest_is_approved_and_the_farthest_is_not(cut):
    rng = np.random.default_rng(0)
    for n in (2, 3, 6):
        candidates, points = rng.random((n, 2)), rng.random((2000, 2))
        u, ballot = utilities(points, candidates), approved(points, candidates, cut)
        rows = np.arange(len(points))
        assert ballot[rows, u.argmax(axis=-1)].all()
        assert not ballot[rows, u.argmin(axis=-1)].any()


def test_thresholds_run_from_plurality_to_all_but_the_farthest():
    rng = np.random.default_rng(1)
    candidates, points = rng.random((6, 2)), rng.random((2000, 2))
    u = utilities(points, candidates)
    np.testing.assert_array_equal(approved(points, candidates, 1.0), u == u.max(axis=-1, keepdims=True))
    np.testing.assert_array_equal(approved(points, candidates, 1e-12), u != u.min(axis=-1, keepdims=True))
    wide, narrow = approved(points, candidates, 0.3), approved(points, candidates, 0.6)
    assert (wide | ~narrow).all() and (wide != narrow).any()  # a lower threshold approves more


def test_for_three_candidates_the_largest_gap_is_the_threshold_half():
    rng = np.random.default_rng(2)
    candidates, points = rng.random((3, 2)), rng.random((5000, 2))
    np.testing.assert_array_equal(approved(points, candidates, GAP), approved(points, candidates, 0.5))


def test_the_largest_gap_can_differ_from_the_threshold_half():
    """Distances with squares 0, 0.35, 0.45, 0.55 and 1: the middle is between the third
    and the fourth candidate, the largest gap between the fourth and the fifth."""
    candidates = np.sqrt([0.0, 0.35, 0.45, 0.55, 1.0])[:, None] * np.array([1.0, 0.0])
    np.testing.assert_array_equal(approved(np.zeros(2), candidates, 0.5), [True, True, True, False, False])
    np.testing.assert_array_equal(approved(np.zeros(2), candidates, GAP), [True, True, True, True, False])


def test_half_plane_is_the_sign_of_the_combination():
    """offset - normal . v is sum_j w_j u_j itself, for weights that sum to 0."""
    rng = np.random.default_rng(3)
    candidates, points = rng.random((5, 2)), rng.random((500, 2))
    for _ in range(10):
        weights = rng.normal(size=5)
        weights -= weights.mean()
        normal, offset = half_plane(candidates, weights)
        np.testing.assert_allclose(offset - points @ normal, utilities(points, candidates) @ weights,
                                   atol=1e-12)

# ---------------------------------------------------------------- Polygons


def _inside(polygon, points):
    """Whether each point is in the convex CCW polygon."""
    start, end = polygon, np.roll(polygon, -1, axis=0)
    cross = ((end[:, 0] - start[:, 0]) * (points[:, None, 1] - start[:, 1])
             - (end[:, 1] - start[:, 1]) * (points[:, None, 0] - start[:, 0]))
    return (cross >= 0).all(axis=1)


@pytest.mark.parametrize("cut", [*CUTS, 1.0], ids=str)
@pytest.mark.parametrize("model", [MODELS[0], MODELS[2]], ids=_ids)
@pytest.mark.parametrize("n", CANDIDATES)
def test_polygons_hold_the_voters_who_approve(n, model, cut):
    """Every voter is in exactly one polygon of each candidate it approves, and in none
    of the others."""
    candidates = CANDIDATES[n]
    box = _voter_box(model)
    points = box[0] + np.random.default_rng(n).random((5000, 2)) * (box[2] - box[0])
    count = np.zeros((len(points), n), dtype=int)
    for polygon, approves in zip(*approval_polygons(candidates, model, cut)):
        count[np.ix_(_inside(polygon, points), approves)] += 1
    np.testing.assert_array_equal(count, approved(points, candidates, cut))

# ---------------------------------------------------------------- Shares


def _sample(model, node, rng):
    """Voters of one node, (SAMPLES, 2)."""
    i, j = node
    if model.distribution == "beta":
        params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
        return np.column_stack([rng.beta(*params[i], SAMPLES), rng.beta(*params[j], SAMPLES)])
    return rng.normal(model.medians[[i, j]], model.sigma, (SAMPLES, 2))


@pytest.mark.parametrize("cut", CUTS, ids=str)
@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_shares_match_sampled_voters(model, cut):
    rng = np.random.default_rng(7)
    for n in (5, 7):
        shares = approval_shares(CANDIDATES[n], model, cut)
        for node in NODES:
            sampled = approved(_sample(model, node, rng), CANDIDATES[n], cut).mean(axis=0)
            np.testing.assert_allclose(shares[node], sampled, rtol=0, atol=5e-3)  # 6 standard errors


@pytest.mark.parametrize("model", MODELS, ids=_ids)
@pytest.mark.parametrize("n", CANDIDATES)
def test_threshold_one_gives_the_first_choice_shares(n, model):
    np.testing.assert_allclose(approval_shares(CANDIDATES[n], model, 1.0),
                               first_choice_shares(CANDIDATES[n], model), rtol=0, atol=_tolerance(model))


@pytest.mark.parametrize("model", MODELS, ids=_ids)
@pytest.mark.parametrize("n", CANDIDATES)
def test_a_threshold_near_zero_leaves_out_only_the_last_place(n, model):
    tolerance = max(_tolerance(model), 1e-8)  # the strips that a threshold of 1e-9 still leaves out
    np.testing.assert_allclose(approval_shares(CANDIDATES[n], model, 1e-9),
                               1 - _last_place_shares(CANDIDATES[n], model), rtol=0, atol=tolerance)


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_two_candidates_are_approved_by_their_first_choices(model):
    for cut in CUTS:
        np.testing.assert_allclose(approval_shares(CANDIDATES[2], model, cut),
                                   first_choice_shares(CANDIDATES[2], model), rtol=0, atol=_tolerance(model))


@pytest.mark.parametrize("model", MODELS, ids=_ids)
def test_three_candidates_share_the_gap_and_the_threshold_half(model):
    np.testing.assert_allclose(approval_shares(CANDIDATES[3], model, GAP),
                               approval_shares(CANDIDATES[3], model, 0.5), rtol=0, atol=_tolerance(model))


@pytest.mark.parametrize("model", MODELS, ids=_ids)
@pytest.mark.parametrize("n", [5, 7])
def test_shares_are_between_first_choices_and_all_but_last_place(n, model):
    """A lower threshold approves more; every cut approves the first choice and not the
    last place, so between one candidate and all but one per voter."""
    candidates, tolerance = CANDIDATES[n], _tolerance(model)
    first, last = first_choice_shares(candidates, model), _last_place_shares(candidates, model)
    by_threshold = [approval_shares(candidates, model, cut) for cut in (0.9, 0.5, 0.25)]
    for narrow, wide in zip(by_threshold, by_threshold[1:]):
        assert (wide >= narrow - tolerance).all()
    for shares in (*by_threshold, approval_shares(candidates, model, GAP)):
        assert (shares >= first - tolerance).all() and (shares <= 1 - last + tolerance).all()
        approvals = shares.sum(axis=-1)
        assert (approvals >= 1 - 2 * tolerance).all() and (approvals <= n - 1 + 2 * tolerance).all()
