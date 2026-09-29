"""Checks of the shares that single methods need (shares.py).

They are compared with the same shares aggregated from the complete ranking
probabilities (ranking_cells / normal compute_ranking_probabilities, exact), and the
edge cache is checked to recompute only what a dragged candidate moves.
"""

import numpy as np
import pytest

import normal
import ranking_cells
from methods import _pairwise_preferences
from shares import (
    Model,
    _edge_key,
    _half_plane,
    edge_cache,
    first_choice_shares,
    pairwise_shares,
    ranking_shares,
)

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
MODELS = [Model("beta", 0.2, "rms"), Model("beta", 0.35, "mean_abs"), Model("normal", 0.2)]


@pytest.fixture(scope="module", params=MODELS, ids=lambda m: f"{m.distribution}-{m.deviation}")
def exact(request):
    """(model, rankings, probabilities at the nodes) without tables or cache."""
    model = request.param
    if model.distribution == "beta":
        params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
        return model, *ranking_cells.compute_ranking_probabilities(CANDIDATES, params)
    return model, *normal.compute_ranking_probabilities(CANDIDATES, model.medians, model.deviation)


def _tolerance(model):
    """Beta: one quadrature across a whole bisector instead of its pieces (5e-5 seen);
    normal: both exact."""
    return 2e-4 if model.distribution == "beta" else 1e-10


def test_pairwise_shares_match_profile(exact):
    model, rankings, probs = exact
    np.testing.assert_allclose(pairwise_shares(CANDIDATES, model),
                               _pairwise_preferences(rankings, probs), rtol=0, atol=_tolerance(model))


def test_pairwise_shares_are_complementary(exact):
    model = exact[0]
    d = pairwise_shares(CANDIDATES, model)
    off = ~np.eye(len(CANDIDATES), dtype=bool)
    np.testing.assert_allclose((d + np.swapaxes(d, -1, -2))[..., off], 1.0, atol=1e-12)
    np.testing.assert_array_equal(d[..., ~off], 0.0)


def test_first_choice_shares_match_profile(exact):
    model, rankings, probs = exact
    first = probs @ np.eye(len(CANDIDATES))[rankings[:, 0]]
    tolerance = 1e-5 if model.distribution == "beta" else 1e-10  # Voronoi edges are cell edges
    np.testing.assert_allclose(first_choice_shares(CANDIDATES, model), first, rtol=0, atol=tolerance)


def test_ranking_shares_match_profile(exact):
    model, rankings, probs = exact
    got_rankings, got = ranking_shares(CANDIDATES, model)
    np.testing.assert_array_equal(got_rankings, rankings)
    np.testing.assert_allclose(got, probs, rtol=0, atol=1e-7)


def test_drag_recomputes_only_moved_bisectors():
    model = MODELS[0]
    edge_cache.clear()
    pairwise_shares(CANDIDATES, model)
    before = edge_cache.misses
    pairwise_shares(CANDIDATES, model)
    assert edge_cache.misses == before  # nothing new for the same candidates

    cached = set(edge_cache._items)
    moved = CANDIDATES.copy()
    moved[0] += [0.02, -0.01]
    pairwise_shares(moved, model)
    computed = set(edge_cache._items) - cached
    assert computed and edge_cache.misses - before == len(computed)
    # only edges of the half-planes of candidate 0, whose bisectors moved, are new
    allowed = set()
    for e in range(1, len(CANDIDATES)):
        poly = _half_plane(moved, 0, e)
        allowed |= {(model, _edge_key(s, t)[0]) for s, t in zip(poly, np.roll(poly, -1, axis=0))}
    assert computed <= allowed
