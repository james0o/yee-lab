"""Theoretical checks of ranking_cells.

Beta parameters: the median and the quantity each spread rule fixes (mean absolute
deviation, RMS distance from the median, tapered a + b) are verified,
the moments by independent numerical integration. Ranking probabilities are verified against closed forms
(bisectors parallel to an axis reduce to a Beta CDF, and the median splits
voters exactly in half) and against Monte Carlo sampling. Interpolation from
Chebyshev nodes is checked against the exact probabilities at every pixel.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta as beta_fn, betainc

from ranking_cells import (
    NODES,
    SPREADS,
    TAPER,
    beta_params,
    beta_params_at,
    centre_shape,
    compute_ranking_probabilities,
    generate_ranking_probabilities,
    interpolate_to_pixels,
    node_medians,
    ranking_cells,
    rankings_path,
    read_cached_ranking_probabilities,
)

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
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


@pytest.fixture(scope="module", params=["pixels", "nodes"])
def medians(request):
    """The pixel medians, and the Chebyshev nodes of a large grid, which reach
    much closer to 0 and 1."""
    return MEDIANS if request.param == "pixels" else node_medians(1000, NODES)


@pytest.fixture(scope="module", params=SPREADS)
def solved(request, medians):
    """(spread, medians, params) for every spread rule."""
    return request.param, medians, beta_params_at(medians, DEVIATION, request.param)


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
    if spread == "tapered":
        np.testing.assert_allclose(
            params.sum(axis=1), 2 * a0 * (4 * medians * (1 - medians)) ** TAPER, rtol=1e-12
        )
        return
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
        beta_params_at(MEDIANS, deviation)


def test_rejects_unknown_spread():
    with pytest.raises(ValueError):
        beta_params_at(MEDIANS, DEVIATION, "median_abs")


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


@pytest.mark.parametrize("spread", [s for s in SPREADS if s != "mean_abs"])
def test_other_spreads_match_monte_carlo(spread):
    """Corner and edge pixels, where the new rules differ most from mean_abs."""
    params = beta_params(PIXELS, DEVIATION, spread)
    rankings, probs = compute_ranking_probabilities(CANDIDATES, params)
    n = len(CANDIDATES)
    codes = rankings.astype(np.int64) @ n ** np.arange(n)
    samples = 1_000_000
    for i, j in [(0, PIXELS - 1), (PIXELS - 1, MIDDLE), (4, 17)]:
        rng = np.random.default_rng(i * PIXELS + j)
        x = rng.beta(params[i, 0], params[i, 1], samples)
        y = rng.beta(params[j, 0], params[j, 1], samples)
        dist = np.hypot(x[:, None] - CANDIDATES[:, 0], y[:, None] - CANDIDATES[:, 1])
        sample_codes = np.argsort(dist, axis=1) @ n ** np.arange(n)
        empirical = (sample_codes[:, None] == codes[None, :]).mean(axis=0)
        exact = probs[i, j]
        sigma = np.sqrt(exact * (1 - exact) / samples)
        assert np.all(np.abs(empirical - exact) <= 5 * sigma + 1e-6)


# ---------------------------------------------------------------- Cache

def test_cache_keeps_spreads_apart(tmp_path):
    pixels, nodes = 12, 7
    paths = {rankings_path(pixels, DEVIATION, s, CANDIDATES, 24, nodes, tmp_path) for s in SPREADS}
    assert len(paths) == len(SPREADS)
    generated = generate_ranking_probabilities(
        CANDIDATES, pixels, DEVIATION, nodes, cache_root=tmp_path, spread="rms"
    )
    cached = read_cached_ranking_probabilities(
        CANDIDATES, pixels, DEVIATION, nodes, cache_root=tmp_path, spread="rms"
    )
    np.testing.assert_array_equal(cached[0], generated[0])
    np.testing.assert_array_equal(cached[1], generated[1])
    for spread in ("mean_abs", "tapered"):
        assert read_cached_ranking_probabilities(
            CANDIDATES, pixels, DEVIATION, nodes, cache_root=tmp_path, spread=spread
        ) is None


# ---------------------------------------------------------------- Interpolation

@pytest.mark.parametrize("spread", SPREADS)
def test_interpolation_matches_exact(spread):
    pixels = 60
    exact = compute_ranking_probabilities(CANDIDATES, beta_params(pixels, DEVIATION, spread))
    medians = node_medians(pixels, 33)
    rankings, probs = compute_ranking_probabilities(
        CANDIDATES, beta_params_at(medians, DEVIATION, spread)
    )
    np.testing.assert_array_equal(rankings, exact[0])
    interpolated = interpolate_to_pixels(probs, medians, pixels)
    np.testing.assert_allclose(interpolated, exact[1], atol=2e-5)
    assert interpolated.dtype == np.float32
    np.testing.assert_allclose(interpolated.sum(axis=-1), 1.0, atol=1e-6)


def test_node_medians_span_pixel_medians():
    medians = node_medians(200, NODES)
    assert len(medians) == NODES
    assert medians[0] == pytest.approx(0.5 / 200, rel=1e-12)
    assert medians[-1] == pytest.approx(1 - 0.5 / 200, rel=1e-12)
    assert np.all(np.diff(medians) > 0)
    np.testing.assert_array_equal(node_medians(PIXELS, 0), MEDIANS)
    np.testing.assert_array_equal(node_medians(PIXELS, PIXELS + 5), MEDIANS)
