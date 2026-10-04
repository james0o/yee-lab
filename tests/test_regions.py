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

from yeelab.build import Approval, Approved, GapApproval, Highest, Mix, Score, Scored, Tally, Voters
from yeelab.margin.regions import MARGINS, grid, regions, winners
from yeelab.margin.shares import Model, first_choice_shares, unapproved_shares, unscored_shares
from yeelab.pixels import beta as pixel_beta, methods as pixel_methods, normal as pixel_normal
from yeelab.ranking_cells import pixel_medians
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


def _pixel_winners(method, model, rankings, probs):
    """Winner of every pixel: pixels.methods, or, for the methods only yeelab.build has
    (checked in test_build.py), the method on the complete profile. The approval and
    score shares are not in a profile of rankings (checked in test_approval.py and
    test_score.py): they are taken at the pixel centres."""
    if method in pixel_methods.METHODS:
        return pixel_methods.METHODS[method](rankings, probs)
    cuts = [share.cut for share in MARGINS[method].needs if isinstance(share, Approved)]
    levels = [share.levels for share in MARGINS[method].needs if isinstance(share, Scored)]
    if not cuts and not levels:
        return MARGINS[method].evaluate(_voters(rankings, probs))[0]
    centres = pixel_medians(probs.shape[0])
    unapproved = {cut: unapproved_shares(CANDIDATES, model, cut, centres) for cut in cuts}
    unscored = {n: unscored_shares(CANDIDATES, model, n, centres) for n in levels}
    return MARGINS[method].evaluate(Voters(unapproved=unapproved or None, unscored=unscored or None))[0]


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
    assert (got == _pixel_winners(method, model, rankings, probs)).mean() > 0.995
    assert (got == -2).mean() < 1e-3
    rings = [(polygon[0], polygon[1:]) for region in shapes for polygon in region["polygons"]]
    assert all(_area(outer) > 0 for outer, _ in rings)
    assert all(_area(hole) < 0 for _, holes in rings for hole in holes)
    assert sum(_area(r) for outer, holes in rings for r in [outer, *holes]) == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize("model", [Model("beta", 0.05, "rms"), Model("normal", 0.05)],
                         ids=lambda m: m.distribution)
@pytest.mark.parametrize("method", ["approval_gap", "approval_avg", "score"])
def test_approval_is_decided_where_nearly_all_voters_approve_the_same(model, method):
    """Narrow voters far from the candidates all approve the same ones, whose shares are
    1 to rounding. Their lead must survive: no ties (a margin of 0 is in no region) and
    no specks of rounding noise, but a few regions that tile the square. Likewise for
    the candidates nearly all voters give the top score."""
    candidates = [[0.5, 0.5], [0.32, 0.55], [0.18, 0.32], [0.16, 0.12]]
    _, winner, margin = winners(method, candidates, model, 160)
    assert (margin > 0).all()
    polygons = [polygon for region in regions(method, candidates, model, 160)
                for polygon in region["polygons"]]
    assert len(polygons) <= 6
    assert sum(_area(ring) for polygon in polygons for ring in polygon) == pytest.approx(1, abs=1e-3)


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
        "FPTP", "IRV", "Borda", "Baldwin", "Nanson", "Schulze", "Condorcet", "Minimax",
        "Black", "King of the hill", "King runoff", "Approval (gap)", "Approval (avg)", "Score", "Voronoi"]
    descriptions = {m["name"]: m["description"] for m in methods}
    assert descriptions["nanson"].endswith('\nEliminate(Tally(BordaCount()), how="mean")')
    # the slider: its method, and its default, which the method is listed with
    assert config["score"] == "score" and config["levels"] == 6 and config["max_levels"] == 11
    assert descriptions["score"].endswith("\nHighest(Tally(Score(6)))")


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


def _half():
    """Approval of half of the candidates: a block of yeelab.build, not a named method."""
    return Highest(Tally(Approval()))


def _mix(share):
    """Approval by both kinds of voters: `share` of them approve half of the candidates,
    the others down to their largest gap."""
    return Highest(Tally(Mix(GapApproval(), Approval(), share=share)))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_a_built_method_is_drawn_like_a_named_one(model):
    """winners() and regions() take a method of yeelab.build itself. The approval mix
    with all voters of one kind is the approval method of that kind."""
    for built, expected in ((Highest(Tally(GapApproval())), "approval_gap"), (_mix(0), "approval_gap"),
                            (_mix(1), _half())):
        for got, want in zip(winners(built, CANDIDATES, model, 64), winners(expected, CANDIDATES, model, 64)):
            np.testing.assert_array_equal(got, want)
        assert regions(built, CANDIDATES, model, 64) == regions(expected, CANDIDATES, model, 64)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_the_approval_mix_goes_from_one_approval_method_to_the_other(model):
    """Every total of the mix is between those of the two methods, so a candidate who
    wins a point with both kinds of voters wins it with any mix of them; elsewhere the
    winner changes with the share."""
    _, gap, _ = winners("approval_gap", CANDIDATES, model, 64)
    _, half, _ = winners(_half(), CANDIDATES, model, 64)
    assert (gap != half).any()
    changed = []
    for share in (0.25, 0.5, 0.75):
        _, winner, margin = winners(_mix(share), CANDIDATES, model, 64)
        assert (margin >= 0).all()
        np.testing.assert_array_equal(winner[gap == half], gap[gap == half])
        changed.append((winner != gap).mean())
    assert 0 < changed[0] < changed[1] < changed[2] < (half != gap).mean()


def _score(levels):
    return Highest(Tally(Score(levels)))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_with_two_levels_is_approval_of_three_candidates_only(model):
    """Two levels approve the candidates closer than halfway between the closest and the
    farthest. For three candidates that is the approval ballot at every cut: the same
    diagram. For five it is the diagram of none of the approval methods."""
    for name in (_half(), "approval_gap", "approval_avg"):
        _, winner, margin = winners(_score(2), CANDIDATES[:3], model, 64)
        _, approved, lead = winners(name, CANDIDATES[:3], model, 64)
        np.testing.assert_array_equal(winner, approved)
        np.testing.assert_allclose(margin, lead, rtol=1e-9, atol=0)
        assert (winners(_score(2), CANDIDATES, model, 64)[1] != winners(name, CANDIDATES, model, 64)[1]).any()


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_levels_change_the_diagram_less_and_less(model):
    """Every number of levels is a method of its own, but more levels round the same
    part of the way less: from 2 to 3 levels the diagram changes more than from 11 to 16."""
    seen = [winners(_score(levels), CANDIDATES, model, 64)[1] for levels in (2, 3, 6, 11, 16)]
    changed = [(one != other).mean() for one, other in zip(seen, seen[1:])]
    assert all(0 < change < 0.5 for change in changed)
    assert changed[0] > 5 * changed[-1]


def test_api_builds_the_method_of_the_slider():
    """levels is the number of scores of score. It defaults to what the method is listed
    with, and the other methods ignore it."""
    client = TestClient(app)

    def shapes(method, **settings):
        request = {"candidates": CANDIDATES.tolist(), "method": method, "grid": 64, **settings}
        response = client.post("/api/regions", json=request)
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes("approval_gap", levels=5) == shapes("approval_gap")

    assert shapes("score") == shapes("score", levels=6)
    seen = [shapes("score", levels=levels) for levels in (2, 3, 6, 11)]
    assert all(one != other for one, other in zip(seen, seen[1:]))
    assert seen[0] != shapes("approval_gap")

    for settings in ({"levels": 1}, {"levels": 12}, {"levels": 2.5}):
        request = {"candidates": CANDIDATES.tolist(), "method": "score", **settings}
        assert client.post("/api/regions", json=request).status_code == 422, settings


def test_api_offers_deviations_from_005():
    """Every offered deviation has voters; 0 (all voters at their pixel) is not offered."""
    client = TestClient(app)
    config = client.get("/api/config").json()
    assert config["deviations"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    assert [voters["deviation"] for voters in config["voters"]] == config["deviations"]
    request = {"candidates": CANDIDATES.tolist(), "method": "approval_gap", "grid": 64}
    assert client.post("/api/regions", json={**request, "deviation": 0.05}).status_code == 200
    assert client.post("/api/regions", json={**request, "deviation": 0}).status_code == 422
