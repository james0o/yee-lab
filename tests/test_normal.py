"""Theoretical checks of normal.py and its margin-share integration.

For voters N(m, sigma^2 I) the share preferring c_i to c_j has a closed form,
Phi(distance of m from their bisector / sigma), for bisectors in every direction
(the Beta model only has one for bisectors parallel to an axis). It follows that
Condorcet methods draw the Voronoi diagram. Also checked: the compiled Owen's T
against scipy's, the shares against Monte Carlo sampling.
"""

import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, owens_t
from scipy.stats import norm

from yeelab.normal import (
    BOX,
    NODES,
    node_medians,
    normal_cells,
    sigma_from_deviation,
    triangle_terms,
)
from yeelab.margin.regions import MARGINS, nearest
from yeelab.margin.shares import Model, pairwise_shares
from yeelab.ranking_cells import pixel_medians
from yeelab.build import Voters
from yeelab.voting import CYCLE

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
PIXELS = 20
DEVIATION = 0.3
MEDIANS = pixel_medians(PIXELS)
AXIS_CANDIDATES = np.array([[0.3, 0.5], [0.7, 0.5], [0.55, 0.2], [0.3, 0.9]])


@pytest.fixture(scope="module", params=[0.05, DEVIATION])
def profile(request):
    """(deviation, pairwise shares, interpolation-node means)."""
    model = Model("normal", request.param)
    return request.param, pairwise_shares(CANDIDATES, model), model.medians


def _grid_points(medians):
    return np.stack(np.meshgrid(medians, medians, indexing="ij"), axis=-1)


def test_sigma_gives_requested_deviation():
    sigma = sigma_from_deviation(DEVIATION)
    deviation, _ = quad(lambda x: abs(x) * norm.pdf(x, scale=sigma), -np.inf, np.inf)
    assert deviation == pytest.approx(DEVIATION, abs=1e-10)


@pytest.mark.parametrize("sigma", [0.063, 0.25, 0.5])
def test_triangles_match_scipy_owens_t(sigma):
    """The compiled Owen's T (quadrature and the a > 1 identity) against scipy's."""
    rng = np.random.default_rng(0)
    segments = rng.uniform(-1, 2, (30, 4))
    segments[:5, 1] = segments[:5, 3]  # horizontal
    segments[5:10, 0] = segments[5:10, 2]  # vertical
    segments[10:13, :2] = MEDIANS[[2, 9, 15], None]  # from a pixel centre (d = 0 there)
    medians = MEDIANS
    for (xs, ys, xe, ye), got in zip(segments, triangle_terms(segments, medians, sigma)):
        length = np.hypot(xe - xs, ye - ys)
        ux, uy = (xe - xs) / length, (ye - ys) / length
        x, y = medians[:, None], medians[None, :]
        d = (xs - x) * uy - (ys - y) * ux
        ts = (xs - x) * ux + (ys - y) * uy
        with np.errstate(divide="ignore", invalid="ignore"):
            def right(t):
                r = np.arctan(t / d) / (2 * np.pi) - owens_t(d / sigma, t / d)
                return np.where(d == 0, 0.0, r)
            expected = right(ts + length) - right(ts)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("candidates", [CANDIDATES, AXIS_CANDIDATES])
def test_cells_partition_box(candidates):
    sigma = sigma_from_deviation(DEVIATION)
    polygons, rankings = normal_cells(candidates, sigma)
    areas = [
        0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        for p in polygons
    ]
    assert min(areas) > 0
    assert sum(areas) == pytest.approx((1 + 2 * BOX * sigma) ** 2, rel=1e-12)
    assert len(np.unique(rankings, axis=0)) == len(rankings)


def test_pairwise_majority_matches_closed_form(profile):
    """Share preferring c_i to c_j is Phi(signed distance from bisector / sigma)."""
    deviation, pairwise, medians = profile
    sigma = sigma_from_deviation(deviation)
    centres = _grid_points(medians)
    for i, j in itertools.combinations(range(len(CANDIDATES)), 2):
        normal = CANDIDATES[j] - CANDIDATES[i]
        offset = (CANDIDATES[j] @ CANDIDATES[j] - CANDIDATES[i] @ CANDIDATES[i]) / 2
        distance = (offset - centres @ normal) / np.linalg.norm(normal)  # > 0 nearer c_i
        np.testing.assert_allclose(pairwise[..., i, j], ndtr(distance / sigma), atol=1e-12)


def test_condorcet_methods_draw_voronoi(profile):
    """Wherever a model node is not on a bisector, Schulze and the Condorcet
    winner are the nearest candidate, and there is no cycle."""
    deviation, pairwise, medians = profile
    centres = _grid_points(medians)
    distance = np.sort(np.linalg.norm(centres[..., None, :] - CANDIDATES, axis=-1), axis=-1)
    clear = distance[..., 1] - distance[..., 0] > 1e-9
    expected = nearest(CANDIDATES, centres)[0]
    voters = Voters(pairwise=pairwise)
    np.testing.assert_array_equal(MARGINS["schulze"].evaluate(voters)[0][clear], expected[clear])
    winners = MARGINS["condorcet"].evaluate(voters)[0]
    assert not (winners[clear] == CYCLE).any()
    np.testing.assert_array_equal(winners[clear], expected[clear])


@pytest.mark.parametrize("pixel", [(0, 0), (PIXELS // 2, PIXELS // 2), (PIXELS - 1, 3), (4, 17)])
def test_matches_monte_carlo(pixel):
    i, j = pixel
    model = Model("normal", DEVIATION)
    medians = model.medians
    samples = 200_000
    rng = np.random.default_rng(i * PIXELS + j)
    points = rng.normal((medians[i], medians[j]), sigma_from_deviation(DEVIATION), (samples, 2))
    rankings = np.argsort(np.linalg.norm(points[:, None] - CANDIDATES, axis=-1), axis=1)
    positions = np.argsort(rankings, axis=1)
    pair = (0, 1)
    empirical = np.mean(positions[:, pair[0]] < positions[:, pair[1]])
    expected = pairwise_shares(CANDIDATES, model)[i, j, *pair]
    standard_error = np.sqrt(expected * (1 - expected) / samples)
    assert abs(empirical - expected) <= 5 * standard_error + 1e-6


def test_node_medians_span_pixel_medians():
    medians = node_medians(200, NODES)
    assert len(medians) == NODES
    assert medians[0] == pytest.approx(0.5 / 200, rel=1e-12)
    assert medians[-1] == pytest.approx(1 - 0.5 / 200, rel=1e-12)
    assert np.all(np.diff(medians) > 0)
    np.testing.assert_allclose(medians, 1 - medians[::-1], atol=1e-15)
    np.testing.assert_array_equal(node_medians(PIXELS, 0), MEDIANS)
