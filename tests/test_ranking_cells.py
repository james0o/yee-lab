"""Theoretical checks of ranking_cells.

Beta parameters: median and mean absolute deviation are verified by independent
numerical integration. Ranking probabilities are verified against closed forms
(bisectors parallel to an axis reduce to a Beta CDF, and the median splits
voters exactly in half) and against Monte Carlo sampling.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta as beta_fn, betainc

from const import CANDIDATES
from ranking_cells import beta_params, compute_ranking_probabilities, ranking_cells

PIXELS = 21  # odd, so the middle pixel has median exactly 0.5
MIDDLE = PIXELS // 2
DEVIATION = 0.3
MEDIANS = (np.arange(PIXELS) + 0.5) / PIXELS

# A-B bisector is x = 0.5, A-D bisector is y = 0.7, C is in general position.
AXIS_CANDIDATES = np.array([[0.3, 0.5], [0.7, 0.5], [0.55, 0.2], [0.3, 0.9]])
A, B, C, D = range(4)


@pytest.fixture(scope="module")
def params():
    return beta_params(PIXELS, DEVIATION)


@pytest.fixture(scope="module")
def axis_profile(params):
    return compute_ranking_probabilities(AXIS_CANDIDATES, params)


@pytest.fixture(scope="module")
def default_profile(params):
    return compute_ranking_probabilities(CANDIDATES, params)


def _prefers(rankings, probs, first, second):
    """Share of voters ranking `first` above `second`, shape (pixels, pixels)."""
    position = np.argsort(rankings, axis=1)
    mask = position[:, first] < position[:, second]
    return probs[..., mask].sum(axis=-1)


# ---------------------------------------------------------------- Beta parameters

def test_params_have_requested_median(params):
    a, b = params[:, 0], params[:, 1]
    np.testing.assert_allclose(betainc(a, b, MEDIANS), 0.5, atol=1e-12)


def test_params_have_requested_deviation(params):
    """E|X - m| by quadrature with the Beta singularities as algebraic weights."""
    for m, (a, b) in zip(MEDIANS, params):
        norm = beta_fn(a, b)
        # [0, m]: (m - x) (1 - x)^(b-1) with weight x^(a-1)
        left, _ = quad(lambda x: (m - x) * (1 - x) ** (b - 1), 0, m,
                       weight="alg", wvar=(a - 1, 0))
        # [m, 1]: (x - m) x^(a-1) with weight (1 - x)^(b-1)
        right, _ = quad(lambda x: (x - m) * x ** (a - 1), m, 1,
                        weight="alg", wvar=(0, b - 1))
        assert (left + right) / norm == pytest.approx(DEVIATION, abs=1e-9)


def test_params_are_mirror_symmetric(params):
    np.testing.assert_allclose(params, params[::-1, ::-1], rtol=1e-10)


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


# ---------------------------------------------------------------- Probabilities

@pytest.mark.parametrize("profile", ["axis_profile", "default_profile"])
def test_probabilities_are_distribution(profile, request):
    _, probs = request.getfixturevalue(profile)
    assert probs.min() >= 0
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-12)


def test_vertical_bisector_matches_beta_cdf(axis_profile, params):
    """A beats B exactly when x < 0.5, so P = F_x(0.5) for every y."""
    rankings, probs = axis_profile
    expected = betainc(params[:, 0], params[:, 1], 0.5)[:, None]
    np.testing.assert_allclose(
        _prefers(rankings, probs, A, B), np.broadcast_to(expected, (PIXELS, PIXELS)),
        atol=1e-8,
    )


def test_horizontal_bisector_matches_beta_cdf(axis_profile, params):
    """D beats A exactly when y > 0.7, so P = 1 - G_y(0.7) for every x."""
    rankings, probs = axis_profile
    expected = 1 - betainc(params[:, 0], params[:, 1], 0.7)[None, :]
    np.testing.assert_allclose(
        _prefers(rankings, probs, D, A), np.broadcast_to(expected, (PIXELS, PIXELS)),
        atol=1e-8,
    )


def test_median_splits_voters_in_half(axis_profile):
    """In pixels with x median 0.5, exactly half the voters prefer A to B."""
    rankings, probs = axis_profile
    np.testing.assert_allclose(_prefers(rankings, probs, A, B)[MIDDLE], 0.5, atol=1e-8)


@pytest.mark.parametrize("profile", ["axis_profile", "default_profile"])
@pytest.mark.parametrize("pixel", [(0, 0), (MIDDLE, MIDDLE), (PIXELS - 1, 3), (4, 17)])
def test_matches_monte_carlo(profile, pixel, params, request):
    rankings, probs = request.getfixturevalue(profile)
    candidates = AXIS_CANDIDATES if profile == "axis_profile" else CANDIDATES
    i, j = pixel
    samples = 1_000_000
    rng = np.random.default_rng(i * PIXELS + j)
    x = rng.beta(params[i, 0], params[i, 1], samples)
    y = rng.beta(params[j, 0], params[j, 1], samples)
    dist = np.hypot(x[:, None] - candidates[:, 0], y[:, None] - candidates[:, 1])
    sampled = np.argsort(dist, axis=1)

    n = len(candidates)
    codes = rankings.astype(np.int64) @ n ** np.arange(n)
    sample_codes = sampled @ n ** np.arange(n)
    assert np.isin(sample_codes, codes).all(), "sampled ranking missing from cells"
    empirical = (sample_codes[:, None] == codes[None, :]).mean(axis=0)

    exact = probs[i, j]
    sigma = np.sqrt(exact * (1 - exact) / samples)
    assert np.all(np.abs(empirical - exact) <= 5 * sigma + 1e-6)
