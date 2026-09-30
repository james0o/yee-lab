"""Checks of the win regions as polygons (the methods of MARGINS, regions.py, /api/regions).

The methods must pick the same winners as the pixel methods (pixels.methods), and
their margins must vanish at every border, since the borders are traced as their zero
set. The polygons, filled even-odd, must reproduce the pixel diagram and tile the
square, with outer rings counter-clockwise and holes clockwise.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from matplotlib.path import Path as MplPath

from yeelab.build import Voters
from yeelab.margin.regions import MARGINS, grid, regions, winners
from yeelab.margin.shares import Model, first_choice_shares
from yeelab.pixels import beta as pixel_beta, methods as pixel_methods, normal as pixel_normal
from yeelab.voting import irv_rounds
from yeelab.web.app import DIAGRAMS, app

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
SEVEN = np.random.default_rng(1).random((7, 2))
MODELS = [Model("beta", 0.2, "rms"), Model("normal", 0.2)]
PIXELS = 150


def _voters(rankings, probs):
    """Every share a method may need, from a complete profile."""
    first = probs @ np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    d = pixel_methods._pairwise_preferences(rankings, probs)
    return Voters(first, d, rankings, probs)


def _pixel_winners(method, rankings, probs):
    """Winner of every pixel: pixels.methods, or, for the methods only yeelab.build has
    (checked in test_build.py), the method on the complete profile."""
    if method in pixel_methods.METHODS:
        return pixel_methods.METHODS[method](rankings, probs)
    return MARGINS[method].evaluate(_voters(rankings, probs))[0]


def _irv_reference(rankings, probs):
    """IRV with 0/1 transfer matrices, one matrix product per round and set of
    eliminated candidates: the numpy version the compiled kernel replaced."""
    n = rankings.shape[1]
    bits = 1 << np.arange(n)
    eliminated = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(bool)
    choice = rankings[np.arange(len(rankings)), (~eliminated[:, rankings]).argmax(axis=-1)]
    transfer = np.eye(n, dtype=probs.dtype)[choice]
    state = np.zeros(probs.shape[:2], dtype=np.int64)
    margin = np.full(probs.shape[:2], np.inf)
    for _ in range(n - 1):
        votes = np.empty((*probs.shape[:2], n), dtype=probs.dtype)
        for s in np.unique(state):
            votes[state == s] = probs[state == s] @ transfer[s]
        votes[(state[..., None] & bits) != 0] = np.inf
        lowest = np.partition(votes, 1, axis=-1)
        margin = np.minimum(margin, lowest[..., 1] - lowest[..., 0])
        state |= bits[votes.argmin(axis=-1)]
    return ((state[..., None] & bits) == 0).argmax(axis=-1), margin


@pytest.mark.parametrize("candidates", [CANDIDATES, SEVEN], ids=["five", "seven"])
def test_irv_kernel_matches_transfer_matrices(candidates):
    rankings, probs = pixel_beta.ranking_probabilities(candidates, 80, 0.2)
    winner, margin = irv_rounds(rankings, probs)
    expected_winner, expected_margin = _irv_reference(rankings, probs)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-6)  # float32 sums


@pytest.mark.parametrize("candidates", [CANDIDATES, SEVEN], ids=["five", "seven"])
def test_margin_variants_pick_the_same_winners(candidates):
    rankings, probs = pixel_beta.ranking_probabilities(candidates, 60, 0.2)
    voters = _voters(rankings, probs)
    for name in pixel_methods.METHODS:  # the others: test_build.py
        winner, margin = MARGINS[name].evaluate(voters)
        np.testing.assert_array_equal(winner, pixel_methods.METHODS[name](rankings, probs), err_msg=name)
        assert (margin >= 0).all(), name


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
@pytest.mark.parametrize("method", MARGINS)
def test_margins_vanish_at_borders(model, method):
    """Grid points next to a change of winner have margin O(grid step): the margin is
    continuous and 0 on the border (a jump would leave O(1) margins there)."""
    size = 200
    _, winner, margin = winners(method, SEVEN, model, size)
    border = np.zeros_like(winner, dtype=bool)
    for axis in (0, 1):
        change = np.diff(winner, axis=axis) != 0
        border[(slice(None),) * axis + (slice(1, None),)] |= change
        border[(slice(None),) * axis + (slice(None, -1),)] |= change
    assert border.any()
    assert margin[border].max() < 30 / size


def _rasterize(shapes, pixels):
    """Winner at the pixel centres from the polygons, filled even-odd; -2 = none."""
    centres = (np.arange(pixels) + 0.5) / pixels
    points = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1).reshape(-1, 2)
    out = np.full(len(points), -2)
    for region in shapes:
        inside = np.zeros(len(points), dtype=bool)
        for polygon in region["polygons"]:
            for ring in polygon:
                inside ^= MplPath(np.reshape(ring, (-1, 2))).contains_points(points)
        out[inside] = region["winner"]
    return out.reshape(pixels, pixels)


def _area(ring):
    x, y = np.reshape(ring, (-1, 2)).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


@pytest.fixture(scope="module", params=MODELS, ids=lambda m: m.distribution)
def profile(request):
    """(model, rankings, probabilities at the pixels) of the pixel pipeline."""
    model = request.param
    if model.distribution == "beta":
        return model, *pixel_beta.ranking_probabilities(CANDIDATES, PIXELS, 0.2, spread="rms")
    return model, *pixel_normal.ranking_probabilities(CANDIDATES, PIXELS, 0.2)


@pytest.mark.parametrize("method", MARGINS)
def test_regions_reproduce_pixel_diagram(profile, method):
    model, rankings, probs = profile
    shapes = regions(method, CANDIDATES, model, 300)
    got = _rasterize(shapes, PIXELS)
    # Pixels may differ only right at a border (and where it runs through pixel centres).
    assert (got == _pixel_winners(method, rankings, probs)).mean() > 0.995
    assert (got == -2).mean() < 1e-3
    rings = [(polygon[0], polygon[1:]) for region in shapes for polygon in region["polygons"]]
    assert all(_area(outer) > 0 for outer, _ in rings)
    assert all(_area(hole) < 0 for _, holes in rings for hole in holes)
    assert sum(_area(r) for outer, holes in rings for r in [outer, *holes]) == pytest.approx(1, abs=1e-3)


def test_voronoi_regions_are_exact():
    shapes = regions("voronoi", CANDIDATES, None, 100)
    assert [r["winner"] for r in shapes] == list(range(len(CANDIDATES)))
    assert sum(_area(r["polygons"][0][0]) for r in shapes) == pytest.approx(1, abs=1e-12)
    centres = (np.arange(PIXELS) + 0.5) / PIXELS
    points = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1)
    distance = np.sort(np.linalg.norm(points[..., None, :] - CANDIDATES, axis=-1), axis=-1)
    clear = distance[..., 1] - distance[..., 0] > 1e-9
    np.testing.assert_array_equal(_rasterize(shapes, PIXELS)[clear],
                                  pixel_methods.voronoi(CANDIDATES, PIXELS)[clear])


def test_grid_reaches_the_walls():
    coords, medians = grid(10, 300)
    assert coords[0] == 0 and coords[-1] == 1 and len(coords) == 12
    assert medians.min() == 0.5 / 300 and medians.max() == 1 - 0.5 / 300


def test_first_choice_regions_tile_even_with_normal_voters_outside():
    """Normal voters leave the square, but the first-choice shares of a node still
    sum to one (the cells cover the box around it)."""
    first = first_choice_shares(CANDIDATES, MODELS[1])
    np.testing.assert_allclose(first.sum(axis=-1), 1.0, atol=1e-9)


def test_config_lists_every_method():
    """The UI makes a button of each; a built method's tooltip ends with its expression."""
    config = TestClient(app).get("/api/config").json()
    methods = config["methods"]
    assert [m["name"] for m in methods] == list(MARGINS)
    assert [m["label"] for m in methods + config["ideals"]] == [
        "FPTP", "IRV", "Borda", "Baldwin", "Nanson", "Schulze", "Condorcet cycle", "Minimax",
        "Black", "Voronoi"]
    nanson = next(m for m in methods if m["name"] == "nanson")
    assert nanson["description"].endswith('\nEliminate(Tally(BordaCount()), how="mean")')


@pytest.mark.parametrize("distribution", ["beta", "normal"])
@pytest.mark.parametrize("method", DIAGRAMS)
def test_api_returns_regions(distribution, method):
    client = TestClient(app)
    response = client.post("/api/regions", json={
        "candidates": CANDIDATES.tolist(), "method": method, "distribution": distribution,
        "deviation": 0.2, "grid": 64,
    })
    assert response.status_code == 200
    shapes = response.json()["regions"]
    assert shapes and all(region["polygons"] for region in shapes)
