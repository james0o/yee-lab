"""Theoretical checks of normal.py.

For voters N(m, sigma^2 I) the share preferring c_i to c_j has a closed form,
Phi(distance of m from their bisector / sigma), for bisectors in every direction
(the Beta model only has one for bisectors parallel to an axis). It follows that
Condorcet methods draw the Voronoi diagram. Also checked against Monte Carlo
sampling, and interpolation from Chebyshev nodes against exact probabilities.
"""

import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from methods import CYCLE, condorcet_cycle, schulze, voronoi
from normal import (
    BOX,
    NODES,
    compute_ranking_probabilities,
    interpolate_to_pixels,
    node_medians,
    normal_cells,
    sigma_from_deviation,
)
from ranking_cells import pixel_medians

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
PIXELS = 20
DEVIATION = 0.3
MEDIANS = pixel_medians(PIXELS)
AXIS_CANDIDATES = np.array([[0.3, 0.5], [0.7, 0.5], [0.55, 0.2], [0.3, 0.9]])


@pytest.fixture(scope="module", params=[0.05, DEVIATION])
def profile(request):
    """(deviation, rankings, probabilities) at the pixel centres."""
    return request.param, *compute_ranking_probabilities(CANDIDATES, MEDIANS, request.param)


def _pixel_centres(medians):
    return np.stack(np.meshgrid(medians, medians, indexing="ij"), axis=-1)


def test_sigma_gives_requested_deviation():
    sigma = sigma_from_deviation(DEVIATION)
    deviation, _ = quad(lambda x: abs(x) * norm.pdf(x, scale=sigma), -np.inf, np.inf)
    assert deviation == pytest.approx(DEVIATION, abs=1e-10)


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


def test_probabilities_are_distribution(profile):
    _, _, probs = profile
    assert probs.min() >= 0
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-12)


def test_pairwise_majority_matches_closed_form(profile):
    """Share preferring c_i to c_j is Phi(signed distance from bisector / sigma)."""
    deviation, rankings, probs = profile
    sigma = sigma_from_deviation(deviation)
    position = np.argsort(rankings, axis=1)
    centres = _pixel_centres(MEDIANS)
    for i, j in itertools.combinations(range(len(CANDIDATES)), 2):
        normal = CANDIDATES[j] - CANDIDATES[i]
        offset = (CANDIDATES[j] @ CANDIDATES[j] - CANDIDATES[i] @ CANDIDATES[i]) / 2
        distance = (offset - centres @ normal) / np.linalg.norm(normal)  # > 0 nearer c_i
        share = probs[..., position[:, i] < position[:, j]].sum(axis=-1)
        np.testing.assert_allclose(share, ndtr(distance / sigma), atol=1e-12)


def test_condorcet_methods_draw_voronoi(profile):
    """Wherever the pixel centre is not on a bisector, Schulze and the Condorcet
    winner are the nearest candidate, and there is no cycle."""
    _, rankings, probs = profile
    centres = _pixel_centres(MEDIANS)
    distance = np.sort(np.linalg.norm(centres[..., None, :] - CANDIDATES, axis=-1), axis=-1)
    clear = distance[..., 1] - distance[..., 0] > 1e-9
    nearest = voronoi(CANDIDATES, PIXELS)
    np.testing.assert_array_equal(schulze(rankings, probs)[clear], nearest[clear])
    winners = condorcet_cycle(rankings, probs)
    assert not (winners[clear] == CYCLE).any()
    np.testing.assert_array_equal(winners[clear], nearest[clear])


@pytest.mark.parametrize("pixel", [(0, 0), (PIXELS // 2, PIXELS // 2), (PIXELS - 1, 3), (4, 17)])
def test_matches_monte_carlo(pixel):
    rankings, probs = compute_ranking_probabilities(CANDIDATES, MEDIANS, DEVIATION)
    i, j = pixel
    samples = 1_000_000
    rng = np.random.default_rng(i * PIXELS + j)
    points = rng.normal((MEDIANS[i], MEDIANS[j]), sigma_from_deviation(DEVIATION), (samples, 2))
    sampled = np.argsort(np.linalg.norm(points[:, None] - CANDIDATES, axis=-1), axis=1)

    n = len(CANDIDATES)
    codes = rankings.astype(np.int64) @ n ** np.arange(n)
    sample_codes = sampled @ n ** np.arange(n)
    assert np.isin(sample_codes, codes).all(), "sampled ranking missing from cells"
    empirical = (sample_codes[:, None] == codes[None, :]).mean(axis=0)

    exact = probs[i, j]
    sigma = np.sqrt(exact * (1 - exact) / samples)
    assert np.all(np.abs(empirical - exact) <= 5 * sigma + 1e-6)


@pytest.mark.parametrize("deviation", [0.05, DEVIATION])
def test_interpolation_matches_exact(deviation):
    pixels = 60
    exact = compute_ranking_probabilities(CANDIDATES, pixel_medians(pixels), deviation)
    medians = node_medians(pixels, NODES)
    rankings, probs = compute_ranking_probabilities(CANDIDATES, medians, deviation)
    np.testing.assert_array_equal(rankings, exact[0])
    interpolated = interpolate_to_pixels(probs, medians, pixels)
    np.testing.assert_allclose(interpolated, exact[1], atol=2e-6)
    assert interpolated.dtype == np.float32


def test_node_medians_span_pixel_medians():
    medians = node_medians(200, NODES)
    assert len(medians) == NODES
    assert medians[0] == pytest.approx(0.5 / 200, rel=1e-12)
    assert medians[-1] == pytest.approx(1 - 0.5 / 200, rel=1e-12)
    assert np.all(np.diff(medians) > 0)
    np.testing.assert_allclose(medians, 1 - medians[::-1], atol=1e-15)
    np.testing.assert_array_equal(node_medians(PIXELS, 0), MEDIANS)
