"""Checks of the win regions as polygons (the methods of MARGINS, regions.py, /api/regions).

The methods must pick the same winners as the pixel methods (pixels.methods), and
their margins must vanish at every border, since the borders are traced as their zero
set. The polygons, filled even-odd, must reproduce the pixel diagram and tile the
square, with outer rings counter-clockwise and holes clockwise.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from matplotlib.path import Path as MplPath

from yeelab.build import Highest, Mix, Score, ScoreAvg, ScoreCluster, ScoreDH, ScoreHybrid, Scored, Tally, Voters
from yeelab.margin.regions import MARGINS, grid, regions, winners
from yeelab.margin.shares import Model, first_choice_shares, unscored_shares
from yeelab.pixels import beta as pixel_beta, methods as pixel_methods, normal as pixel_normal
from yeelab.ranking_cells import pixel_medians
from yeelab.voting import irv_rounds
from yeelab.web.app import DIAGRAMS, app

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
SEVEN = np.random.default_rng(1).random((7, 2))
MODELS = [Model("beta", 0.2, "rms"), Model("normal", 0.2)]
# approval: the score methods with two levels, which approve the candidates above the
# largest gap and those closer than the mean distance
APPROVE_GAP = Highest(Tally(ScoreDH(2)))
APPROVE_MEAN = Highest(Tally(ScoreAvg(2)))
PIXELS = 150


def _voters(rankings, probs):
    """Every share a method may need, from a complete profile."""
    first = probs @ np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    d = pixel_methods._pairwise_preferences(rankings, probs)
    return Voters(first, d, rankings, probs)


def _pixel_winners(method, model, rankings, probs):
    """Winner of every pixel: pixels.methods, or, for the methods only yeelab.build has
    (checked in test_build.py), the method on the complete profile. The score shares
    are not in a profile of rankings (checked in test_score.py): they are taken at the
    pixel centres."""
    if method in pixel_methods.METHODS:
        return pixel_methods.METHODS[method](rankings, probs)
    scores = [share for share in MARGINS[method].needs if isinstance(share, Scored)]
    if not scores:
        return MARGINS[method].evaluate(_voters(rankings, probs))[0]
    centres = pixel_medians(probs.shape[0])
    unscored = {share: unscored_shares(CANDIDATES, model, share.levels, centres, share.rule, share.delta,
                                       mu=share.mu, kappa=share.kappa, power=share.power)
                for share in scores}
    return MARGINS[method].evaluate(Voters(unscored=unscored))[0]


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
@pytest.mark.parametrize("method", ["score", Highest(Tally(Score(2))), Highest(Tally(Score(2, power=1.5))),
                                    Highest(Tally(Score(2, power=2))), APPROVE_MEAN, APPROVE_GAP,
                                    Highest(Tally(ScoreHybrid(2))), Highest(Tally(ScoreCluster(2)))], ids=str)
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
    """The UI makes a button of each; a built method's tooltip ends with its expression.
    Score is the one score method."""
    config = TestClient(app).get("/api/config").json()
    methods = config["methods"]
    assert [m["name"] for m in methods] == list(MARGINS)
    assert [m["label"] for m in methods + config["ideals"]] == [
        "FPTP", "IRV", "Borda", "Baldwin", "Nanson", "Schulze", "Condorcet", "Minimax",
        "Black", "King of the hill", "King runoff", "Score", "Voronoi"]
    descriptions = {m["name"]: m["description"] for m in methods}
    assert descriptions["nanson"].endswith('\nEliminate(Tally(BordaCount()), how="mean")')
    # the categories and power sliders: their method, their defaults, which it is listed
    # with, and their ranges
    assert config["scores"] == ["score"]
    assert config["levels"] == 6 and config["max_levels"] == 11
    assert descriptions["score"].endswith("\nHighest(Tally(Score(6, power=1.5)))")
    assert config["power"] == 1.5
    assert (config["min_power"], config["max_power"], config["power_step"]) == (1.0, 2.0, 0.05)
    for gone in ("deltas", "delta", "clusters", "mu", "kappa", "powers"):
        assert gone not in config


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


def _mix(share):
    """Approval by both kinds of voters: `share` of them approve the candidates closer
    than their mean distance, the others those above their largest gap."""
    return Highest(Tally(Mix(ScoreDH(2), ScoreAvg(2), share=share)))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_a_built_method_is_drawn_like_a_named_one(model):
    """winners() and regions() take a method of yeelab.build itself. The approval mix
    with all voters of one kind is the approval of that kind."""
    for built, expected in ((Highest(Tally(Score(6, power=1.5))), "score"), (_mix(0), APPROVE_GAP),
                            (_mix(1), APPROVE_MEAN)):
        for got, want in zip(winners(built, CANDIDATES, model, 64), winners(expected, CANDIDATES, model, 64)):
            np.testing.assert_array_equal(got, want)
        assert regions(built, CANDIDATES, model, 64) == regions(expected, CANDIDATES, model, 64)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_the_approval_mix_goes_from_one_kind_of_approval_to_the_other(model):
    """Every total of the mix is between those of the two kinds, so a candidate who wins
    a point with both kinds of voters wins it with any mix of them; elsewhere the winner
    changes with the share."""
    _, gap, _ = winners(APPROVE_GAP, CANDIDATES, model, 64)
    _, mean, _ = winners(APPROVE_MEAN, CANDIDATES, model, 64)
    assert (gap != mean).any()
    changed = []
    for share in (0.25, 0.5, 0.75):
        _, winner, margin = winners(_mix(share), CANDIDATES, model, 64)
        assert (margin >= 0).all()
        np.testing.assert_array_equal(winner[gap == mean], gap[gap == mean])
        changed.append((winner != gap).mean())
    assert 0 < changed[0] < changed[1] < changed[2] < (mean != gap).mean()


def _score(levels):
    return Highest(Tally(Score(levels)))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_two_levels_of_every_rule_are_one_diagram_for_three_candidates_only(model):
    """With two levels the middle one of three candidates is approved where it is closer
    to the closest than to the farthest, by every rule: the same diagram. For five the
    rules approve differently, and so do their diagrams."""
    _, winner, margin = winners(_score(2), CANDIDATES[:3], model, 64)
    for method in (APPROVE_MEAN, APPROVE_GAP):
        _, other, lead = winners(method, CANDIDATES[:3], model, 64)
        np.testing.assert_array_equal(other, winner)
        np.testing.assert_allclose(lead, margin, rtol=1e-9, atol=0)
    five = [winners(method, CANDIDATES, model, 64)[1] for method in (_score(2), APPROVE_MEAN, APPROVE_GAP)]
    assert all((one != other).any() for one, other in zip(five, five[1:] + five[:1]))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_levels_change_the_diagram_less_and_less(model):
    """Every number of levels is a method of its own, but more levels round the same
    part of the way less: from 2 to 3 levels the diagram changes more than from 11 to 16."""
    seen = [winners(_score(levels), CANDIDATES, model, 64)[1] for levels in (2, 3, 6, 11, 16)]
    changed = [(one != other).mean() for one, other in zip(seen, seen[1:])]
    assert all(0 < change < 0.5 for change in changed)
    assert changed[0] > 5 * changed[-1]


def _score_dh(levels):
    return Highest(Tally(ScoreDH(levels)))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_avg_is_a_method_apart_from_score(model):
    """The mean distance in the middle of the scale moves the scores, and so the winners."""
    avg = winners(Highest(Tally(ScoreAvg(6))), CANDIDATES, model, 64)[1]
    assert (avg != winners(_score(6), CANDIDATES, model, 64)[1]).any()


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_dh_with_more_levels_than_candidates_is_not_borda(model):
    """With as many steps as gaps or more, every gap could get one, and that is Borda.
    D'Hondt gives a small gap none, so the diagram stays apart from Borda's."""
    borda = winners("borda", CANDIDATES, model, 64)[1]
    for levels in (5, 11, 16):
        assert (winners(_score_dh(levels), CANDIDATES, model, 64)[1] != borda).mean() > 0.05


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_hybrid_is_a_method_apart_from_score_range_and_score_dh(model):
    """Its closer half is D'Hondt's and its farther half range's, and the diagram is
    neither of theirs, with six levels and with two."""
    for levels in (6, 2):
        hybrid = winners(Highest(Tally(ScoreHybrid(levels))), CANDIDATES, model, 64)[1]
        for other in (Score, ScoreDH):
            assert (hybrid != winners(Highest(Tally(other(levels))), CANDIDATES, model, 64)[1]).any()


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_score_cluster_is_a_method_apart_from_score_range(model):
    """It is Score's ballot unless that splits a cluster, and the diagram is not Score's,
    with six levels and with two."""
    for levels in (6, 2):
        cluster = winners(Highest(Tally(ScoreCluster(levels))), CANDIDATES, model, 64)[1]
        assert (cluster != winners(Highest(Tally(Score(levels))), CANDIDATES, model, 64)[1]).any()


def test_api_builds_score_from_levels_and_power():
    """levels is the number of scores of Score, 2 to 11, and power, 1 to 2, raises its part
    of the way to it. They default to what the method is listed with (6 and 1.5), the
    other methods ignore them, and settings the UI no longer has are not taken."""
    client = TestClient(app)

    def shapes(method, **settings):
        request = {"candidates": CANDIDATES.tolist(), "method": method, "grid": 64, **settings}
        response = client.post("/api/regions", json=request)
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes("borda", levels=5, power=2) == shapes("borda")
    assert shapes("score") == shapes("score", levels=6, power=1.5)
    seen = [shapes("score", levels=levels) for levels in (2, 3, 6, 11)]
    assert all(one != other for one, other in zip(seen, seen[1:]))
    for levels in (2, 6):
        seen = [shapes("score", levels=levels, power=power) for power in (1, 1.5, 2)]
        assert all(one != other for one, other in zip(seen, seen[1:]))
    # two levels are the approval ballot that cuts at 2^(-1/p) of the way
    for power in (1, 1.5, 2):
        approval = Highest(Tally(Score(2, power=power)))
        assert (shapes("score", levels=2, power=power)
                == json.loads(json.dumps(regions(approval, CANDIDATES.tolist(), MODELS[0], 64))))

    for settings in ({"levels": 1}, {"levels": 12}, {"levels": 2.5}, {"power": 0.95}, {"power": 2.05},
                     {"power": "x"}):
        request = {"candidates": CANDIDATES.tolist(), "method": "score", **settings}
        assert client.post("/api/regions", json=request).status_code == 422, settings
    for gone in ("score_range", "score_avg", "score_dh", "score_hybrid", "score_cluster"):
        request = {"candidates": CANDIDATES.tolist(), "method": gone}
        assert client.post("/api/regions", json=request).status_code == 422, gone


def test_api_offers_deviations_from_005():
    """Every offered deviation has voters; 0 (all voters at their pixel) is not offered."""
    client = TestClient(app)
    config = client.get("/api/config").json()
    assert config["deviations"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    assert [voters["deviation"] for voters in config["voters"]] == config["deviations"]
    request = {"candidates": CANDIDATES.tolist(), "method": "score", "grid": 64}
    assert client.post("/api/regions", json={**request, "deviation": 0.05}).status_code == 200
    assert client.post("/api/regions", json={**request, "deviation": 0}).status_code == 422
