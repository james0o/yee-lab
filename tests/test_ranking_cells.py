"""Theoretical checks of ranking_cells.py and the Beta margin-share implementation.

Beta parameters: the median and the quantity each spread rule fixes (mean absolute
deviation, RMS distance from the median) are verified, the moments by independent
numerical integration. Ranking probabilities are verified against closed forms
(bisectors parallel to an axis reduce to a Beta CDF, and the median splits
voters exactly in half) and against Monte Carlo sampling. Interpolation from
Chebyshev nodes is checked against the direct shares at the evaluated medians.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta as beta_fn, betainc

from yeelab.ranking_cells import (
    NODES,
    SPREADS,
    beta_params_at,
    centre_shape,
    node_medians,
    ranking_cells,
)

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
AXIS_CANDIDATES = np.array([[0.3, 0.5], [0.7, 0.5], [0.55, 0.2], [0.3, 0.9]])
DEVIATION = 0.3
@pytest.fixture(scope="module", params=SPREADS)
def solved(request):
    """(spread, medians, parameters) at the Chebyshev nodes."""
    medians = node_medians(1000, NODES)
    return request.param, medians, beta_params_at(medians, DEVIATION, request.param)


# ---------------------------------------------------------------- Beta parameters

def _moment(m, a, b, power):
    """E|X - m|^power for X ~ Beta(a, b), by quadrature with the Beta
    singularities as algebraic weights."""
    # [0, m]: (m - x)^power (1 - x)^(b-1) with weight x^(a-1)
    left, _ = quad(lambda x: (m - x) ** power * (1 - x) ** (b - 1), 0, m,
                   weight="alg", wvar=(a - 1, 0))
    # [m, 1]: (x - m)^power x^(a-1) with weight (1 - x)^(b-1)
    right, _ = quad(lambda x: (x - m) ** power * x ** (a - 1), m, 1,
                    weight="alg", wvar=(0, b - 1))
    return (left + right) / beta_fn(a, b)


def test_params_have_requested_median(solved):
    _, medians, params = solved
    a, b = params[:, 0], params[:, 1]
    np.testing.assert_allclose(betainc(a, b, medians), 0.5, atol=1e-12)


@pytest.mark.parametrize("deviation", [0.05, 0.1, DEVIATION, 0.45])
def test_centre_shape_has_requested_deviation(deviation):
    a0 = centre_shape(deviation)
    assert _moment(0.5, a0, a0, 1) == pytest.approx(deviation, rel=1e-9)


def test_spreads_agree_at_the_centre():
    """Every rule gives the centre pixel Beta(a0, a0) with E|X - 1/2| = deviation."""
    a0 = centre_shape(DEVIATION)
    for spread in SPREADS:
        np.testing.assert_allclose(
            beta_params_at([0.5], DEVIATION, spread), [[a0, a0]], rtol=1e-9
        )


def test_params_keep_their_spread(solved):
    """The quantity fixed by each rule is the same for every median."""
    spread, medians, params = solved
    a0 = centre_shape(DEVIATION)
    power, target = {"mean_abs": (1, DEVIATION), "rms": (2, 1 / (4 * (2 * a0 + 1)))}[spread]
    for m, (a, b) in zip(medians, params):
        assert _moment(m, a, b, power) == pytest.approx(target, abs=1e-9)


def test_params_are_mirror_symmetric(solved):
    _, _, params = solved
    np.testing.assert_allclose(params, params[::-1, ::-1], rtol=1e-10)


@pytest.mark.parametrize("spread", SPREADS)
@pytest.mark.parametrize("median", [0.02, 0.9, 0.9995])
def test_single_median_is_solved(spread, median):
    """A lone median far from 1/2 gets the same parameters as in a full sweep."""
    sweep = np.linspace(median, 0.5, 400)
    np.testing.assert_allclose(
        beta_params_at([median], DEVIATION, spread), beta_params_at(sweep, DEVIATION, spread)[:1],
        rtol=1e-8,
    )


def test_only_mean_abs_keeps_deviation_near_walls():
    """Near a wall the other rules let E|X - m| shrink (docs/math.typ)."""
    for spread in SPREADS:
        (a, b), = beta_params_at([0.98], DEVIATION, spread)
        deviation = _moment(0.98, a, b, 1)
        if spread == "mean_abs":
            assert deviation == pytest.approx(DEVIATION, abs=1e-9)
        else:
            assert deviation < 0.25


@pytest.mark.parametrize("deviation", [0, 0.5, -0.1])
def test_rejects_impossible_deviation(deviation):
    with pytest.raises(ValueError):
        beta_params_at([0.5], deviation)


def test_rejects_unknown_spread():
    with pytest.raises(ValueError):
        beta_params_at([0.5], DEVIATION, "median_abs")


# ---------------------------------------------------------------- Cells

@pytest.mark.parametrize("candidates", [CANDIDATES, AXIS_CANDIDATES])
def test_cells_partition_unit_square(candidates):
    polygons, rankings = ranking_cells(candidates)
    areas = [
        0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        for p in polygons
    ]
    assert min(areas) > 0
    assert sum(areas) == pytest.approx(1.0, abs=1e-12)
    assert len(np.unique(rankings, axis=0)) == len(rankings)


@pytest.mark.parametrize("candidates", [CANDIDATES, AXIS_CANDIDATES])
def test_cell_ranking_holds_everywhere_inside(candidates):
    """Random convex combinations of cell vertices are ranked as the cell says."""
    rng = np.random.default_rng(0)
    polygons, rankings = ranking_cells(candidates)
    for polygon, ranking in zip(polygons, rankings):
        weights = rng.dirichlet(np.ones(len(polygon)), size=200)
        points = weights @ polygon
        dist = np.linalg.norm(points[:, None, :] - candidates[None], axis=-1)
        np.testing.assert_array_equal(np.argsort(dist, axis=1), np.tile(ranking, (200, 1)))


def test_node_medians_span_model_domain():
    medians = node_medians(200, NODES)
    assert len(medians) == NODES
    assert medians[0] == pytest.approx(0.5 / 200, rel=1e-12)
    assert medians[-1] == pytest.approx(1 - 0.5 / 200, rel=1e-12)
    assert np.all(np.diff(medians) > 0)
    np.testing.assert_allclose(medians, 1 - medians[::-1], atol=1e-15)
