"""Checks of the win regions as polygons (the methods of MARGINS, regions.py, /api/regions).

Their margins must vanish at every border, since the borders are traced as their zero
set. The polygons, filled even-odd, must reproduce the winner grid and tile the
square, with outer rings counter-clockwise and holes clockwise.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from matplotlib.path import Path as MplPath
from scipy.spatial import Voronoi

from yeelab.build import (D21, METHODS, AntiPlurality, Bullet, Burial, Conditional, Highest, Mix, Plurality, Score,
                          ScoreAvg, ScoreCluster, ScoreComparisons, ScoreDH, ScoreHybrid, Tally, Unbeaten)
from yeelab.margin.metrics import (FOCUS_REACH, FOCUS_ROUND, OUTLINE_STEPS, borders, convex_hull, focus_hull,
                                   focus_outline, hull_distance, voronoi_agreement)
from yeelab.margin.regions import MARGINS, grid, nearest, regions, voronoi_borders, winners
from yeelab.build.methods import FOUND
from yeelab.margin.shares import Model, first_choice_shares, ranking_shares, score_comparison_shares, unscored_shares
from yeelab.voting import CYCLE, irv_rounds
from yeelab.web.app import DIAGRAMS, MAX_TILT, TILT_STEP, _score_ballot_settings, _score_ballots, app

from references import TESTED_METHODS

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
SEVEN = np.random.default_rng(1).random((7, 2))
MODELS = [Model("beta", 0.2, "rms"), Model("normal", 0.2)]
# approval: the score methods with two levels, which approve the candidates above the
# largest gap and those closer than the mean distance
APPROVE_GAP = Highest(Tally(ScoreDH(2)))
APPROVE_MEAN = Highest(Tally(ScoreAvg(2)))
PIXELS = 150


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
    rankings, probs = ranking_shares(candidates, Model("beta", 0.2, "rms", 80))
    winner, margin = irv_rounds(rankings, probs)
    expected_winner, expected_margin = _irv_reference(rankings, probs)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-6)  # float32 sums


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


@pytest.mark.parametrize("method", MARGINS)
@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_regions_reproduce_margin_winners(model, method):
    shapes = regions(method, CANDIDATES, model, 300)
    got = _rasterize(shapes, PIXELS)
    _, expected, _ = winners(method, CANDIDATES, model, PIXELS)
    assert (got == expected[1:-1, 1:-1]).mean() > 0.995
    assert (got == -2).mean() < 1e-3
    rings = [(polygon[0], polygon[1:]) for region in shapes for polygon in region["polygons"]]
    assert all(_area(outer) > 0 for outer, _ in rings)
    assert all(_area(hole) < 0 for _, holes in rings for hole in holes)
    assert sum(_area(r) for outer, holes in rings for r in [outer, *holes]) == pytest.approx(1, abs=1e-3)


def test_plot_cli_defaults_to_png_and_can_export_svg(tmp_path, monkeypatch):
    from yeelab.plot import main

    monkeypatch.chdir(tmp_path)
    png_path = tmp_path / "voronoi.png"
    main(["--method", "voronoi"])
    assert png_path.read_bytes().startswith(b"\x89PNG")

    svg_path = tmp_path / "voronoi.svg"
    main(["--method", "voronoi", "--svg", "--output", str(svg_path)])
    assert svg_path.read_bytes().lstrip().startswith(b"<?xml")
    assert b"<svg" in svg_path.read_bytes()


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
    expected = nearest(CANDIDATES, points)[0]
    np.testing.assert_array_equal(_rasterize(shapes, PIXELS)[clear],
                                  expected[clear])


def _grid_points(size):
    coordinates = (np.arange(size) + 0.5) / size
    return np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)


def test_metric_is_zero_for_the_voronoi_diagram():
    points = _grid_points(100)
    for result in voronoi_agreement(CANDIDATES, points, nearest(CANDIDATES, points)[0], 0.2).values():
        assert result["error"] == pytest.approx(0) and result["relative"] == pytest.approx(0)
        assert result["random"] > 0


def test_loss_is_the_regret_over_regret_plus_the_median_gap():
    """Two candidates 0.4 apart (L = 0.4), every point given to the left one: the loss of a
    point is r / (r + L) with r its extra distance, averaged over the area of the grid."""
    from yeelab.margin.metrics import _area_weights

    candidates = np.array([[0.3, 0.5], [0.7, 0.5]])
    points = _grid_points(60)
    distances = np.linalg.norm(points[..., None, :] - candidates, axis=-1)
    regret = distances[..., 0] - distances.min(axis=-1)
    result = voronoi_agreement(candidates, points, np.zeros(points.shape[:2], dtype=int), 0.2)["diagram"]
    assert result["error"] == pytest.approx(100 * np.average(regret / (regret + 0.4), weights=_area_weights(points)))


def test_random_winner_is_the_mean_over_the_candidates():
    """The error of a random winner is the mean of the errors of giving every point to one
    candidate, whatever the method; relative is the square root of the error over it."""
    points = _grid_points(50)
    every = [voronoi_agreement(CANDIDATES, points, np.full(points.shape[:2], c), 0.2)["diagram"]["error"]
             for c in range(len(CANDIDATES))]
    winner = (nearest(CANDIDATES, points)[0] + 1) % len(CANDIDATES)
    result = voronoi_agreement(CANDIDATES, points, winner, 0.2)["diagram"]
    assert result["random"] == pytest.approx(np.mean(every))
    assert result["relative"] == pytest.approx(100 * np.sqrt(result["error"] / result["random"]))


def test_cycles_are_left_out():
    """A point without a winner (a Condorcet cycle, Undecided) is no result: both errors are
    area means over the decided points only, and undefined where nothing is decided."""
    from yeelab.margin.metrics import _area_weights

    points = _grid_points(40)
    winner = (nearest(CANDIDATES, points)[0] + 1) % len(CANDIDATES)
    winner[:10] = CYCLE
    distances = np.linalg.norm(points[..., None, :] - CANDIDATES, axis=-1)
    regret = distances - distances.min(axis=-1, keepdims=True)
    gaps = np.linalg.norm(CANDIDATES[:, None] - CANDIDATES[None], axis=-1)
    loss = regret / (regret + np.median(gaps[np.triu_indices(len(CANDIDATES), 1)]))
    decided, area = winner >= 0, _area_weights(points)
    chosen = np.take_along_axis(loss, np.maximum(winner, 0)[..., None], -1)[..., 0]
    result = voronoi_agreement(CANDIDATES, points, winner, 0.2)["diagram"]
    assert result["error"] == pytest.approx(100 * np.average(chosen[decided], weights=area[decided]))
    assert result["random"] == pytest.approx(100 * np.average(loss.mean(-1)[decided], weights=area[decided]))
    assert voronoi_agreement(CANDIDATES, points, np.full(points.shape[:2], CYCLE), 0.2)["diagram"]["relative"] is None


def test_cycles_can_be_the_worst_result():
    """With cycles="worst" a point without a winner counts with the loss of the farthest
    candidate, and the random winner's error is over all points."""
    from yeelab.margin.metrics import _area_weights

    points = _grid_points(40)
    winner = (nearest(CANDIDATES, points)[0] + 1) % len(CANDIDATES)
    winner[:10] = CYCLE
    distances = np.linalg.norm(points[..., None, :] - CANDIDATES, axis=-1)
    regret = distances - distances.min(axis=-1, keepdims=True)
    gaps = np.linalg.norm(CANDIDATES[:, None] - CANDIDATES[None], axis=-1)
    loss = regret / (regret + np.median(gaps[np.triu_indices(len(CANDIDATES), 1)]))
    chosen = np.where(winner >= 0, np.take_along_axis(loss, np.maximum(winner, 0)[..., None], -1)[..., 0],
                      loss.max(axis=-1))
    area = _area_weights(points)
    result = voronoi_agreement(CANDIDATES, points, winner, 0.2, cycles="worst")["diagram"]
    assert result["error"] == pytest.approx(100 * np.average(chosen, weights=area))
    assert result["random"] == pytest.approx(100 * np.average(loss.mean(-1), weights=area))
    every = voronoi_agreement(CANDIDATES, points, np.full(points.shape[:2], CYCLE), 0.2, cycles="worst")
    assert every["diagram"]["error"] == pytest.approx(100 * np.average(loss.max(-1), weights=area))
    with pytest.raises(ValueError, match="cycles"):
        voronoi_agreement(CANDIDATES, points, winner, 0.2, cycles="drop")


def _to_segments(points, segments):
    """Distance of each point to the nearest of the segments (S, 2, 2)."""
    starts, directions = segments[:, 0], segments[:, 1] - segments[:, 0]
    t = np.clip(((points[:, None] - starts) * directions).sum(-1) / (directions ** 2).sum(-1), 0, 1)
    return np.linalg.norm(points[:, None] - (starts + t[..., None] * directions), axis=-1).min(axis=1)


def _along(segments, step=0.005):
    """Points along the segments, about `step` apart."""
    return np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / step))) for a, b in segments])


@pytest.mark.parametrize("pixel_median, shift", [("marginal", 1.0), ("geometric", 1.0), ("geometric", 0.5)])
def test_exact_voronoi_borders_are_where_the_traced_ones_are(pixel_median, shift):
    """voronoi_borders are the borders regions() traces for the Voronoi diagram, with pixels
    at geometric medians only within the moved grid too. They differ only next to a point
    where three Voronoi cells meet, by less than a grid cell: there the traced ones stop
    short, and leave the bisector where a grid point's margin is to a third candidate."""
    size = 160
    exact = voronoi_borders(CANDIDATES, MODELS[0], size, pixel_median, shift)
    traced = borders(regions("voronoi", CANDIDATES, MODELS[0], size, pixel_median, shift))
    assert {tuple(sorted(pair)) for pair in exact[1]} == {tuple(sorted(pair)) for pair in traced[1]}
    exact_segments, traced_segments = np.asarray(exact[0], dtype=float), np.asarray(traced[0], dtype=float)
    off = _to_segments(_along(traced_segments), exact_segments)
    assert np.mean(off < 1e-6) > 0.95  # on the bisectors, to the six decimals of the traced vertices
    assert off.max() < 2 ** 0.5 / size
    assert _to_segments(_along(exact_segments), traced_segments).max() < 2 ** 0.5 / size


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
    Score and STAR use the score settings."""
    config = TestClient(app).get("/api/config").json()
    methods = config["methods"] + config["found"]
    assert [m["name"] for m in config["found"]] == list(FOUND)
    assert [m["name"] for m in config["methods"]] == [name for name in MARGINS if name not in FOUND]
    assert {m["name"] for m in methods + config["ideals"]} == {*MARGINS, "voronoi"}
    assert config["ideals"][0]["label"] == "Voronoi"
    assert all({"name", "label", "description"} <= method.keys() for method in methods)
    assert all(method["description"].endswith(repr(MARGINS[method["name"]])) for method in methods)
    descriptions = {m["name"]: m["description"] for m in methods}
    assert descriptions["nanson"].endswith('\nEliminate(Tally(BordaCount()), how="mean")')
    # the categories and power sliders: their method, their defaults, which it is listed
    # with, and their ranges
    assert {"score", "star"} <= set(config["scores"])
    assert "system321" in config["scores"]
    assert "system321" not in config["score_levels"]
    assert config["fixed_score_levels"]["system321"] == [3]
    assert [name for name in config["score_comparisons"] if name in TESTED_METHODS] == ["star", "system321"]
    assert config["tie_weight"] == 0 and config["tie_weight_step"] == 0.05
    assert config["spread_power"] == 2.0
    assert config["spread_powers"] == [2.0]  # only rms: the UI shows no slider and no power
    assert config["spread"]["label"] == "RMS distance from median" and "P" not in config["spread"]["tex"]
    assert config["levels"] == 6 and config["max_levels"] == 11
    assert descriptions["score"].endswith("\nHighest(Tally(Score(6, power=1.5)))")
    assert descriptions["star"].endswith(
        "\nUnbeaten(ScoreComparisons(Score(6, power=1.5)), among=Highest(Tally(Score(6, power=1.5)), n=2))")
    assert config["power"] == 1.5
    assert (config["min_power"], config["max_power"], config["power_step"]) == (1.0, 2.0, 0.05)
    for gone in ("deltas", "delta", "clusters", "mu", "kappa", "powers"):
        assert gone not in config


def test_regions_offer_only_the_default_spread_power():
    """The API takes the power the UI offers, rms, and no other (SPREAD_POWERS)."""
    client = TestClient(app)
    request = {
        "candidates": CANDIDATES.tolist(), "method": "fptp", "spread_power": 2,
        "grid": 32, "metrics": False, "overlay": False,
    }
    response = client.post("/api/regions", json=request)
    assert response.status_code == 200
    expected = regions("fptp", CANDIDATES, Model("beta", 0.2, "rms"), 32)
    assert response.json()["regions"] == json.loads(json.dumps(expected))
    for other in (16, 8):
        assert client.post("/api/regions", json={**request, "spread_power": other}).status_code == 422


def test_voter_density_endpoint_returns_parameters_for_selected_power():
    client = TestClient(app)
    response = client.get("/api/voters", params={"deviation": 0.2, "spread_power": 2})
    assert response.status_code == 200
    voters = response.json()
    assert voters["deviation"] == 0.2
    assert len(voters["beta_params"]) == 300
    for other in (64, 8):
        assert client.get("/api/voters", params={"deviation": 0.2, "spread_power": other}).status_code == 422


def test_runoff_tie_weight_is_applied_to_nested_score_comparisons():
    client = TestClient(app)
    request = {
        "candidates": CANDIDATES.tolist(), "method": "star", "levels": 2, "power": 2,
        "grid": 64,
    }
    default = client.post("/api/regions", json=request)
    explicit_zero = client.post("/api/regions", json={**request, "tie_weight": 0})
    weighted = client.post("/api/regions", json={**request, "tie_weight": 0.5})
    assert default.status_code == explicit_zero.status_code == weighted.status_code == 200
    assert default.json()["regions"] == explicit_zero.json()["regions"]

    ballot = Score(2, power=2)
    expected = Unbeaten(
        ScoreComparisons(ballot, tie_weight=0.5),
        among=Highest(Tally(ballot), n=2),
    )
    assert weighted.json()["regions"] == json.loads(
        json.dumps(regions(expected, CANDIDATES.tolist(), MODELS[0], 64))
    )
    assert client.post("/api/regions", json={**request, "tie_weight": 1.1}).status_code == 422


def test_fixed_score_levels_ignore_category_slider_but_keep_power_slider():
    """A fixed Score ballot keeps its categories while the UI power slider still applies."""
    client = TestClient(app)

    def shapes(levels, power):
        response = client.post("/api/regions", json={
            "candidates": CANDIDATES.tolist(), "method": "system321",
            "levels": levels, "power": power, "grid": 64,
        })
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes(2, 2) == shapes(11, 2)

    ballot = Score(3, power=2, fixed=True)
    expected = Unbeaten(
        ScoreComparisons(ballot),
        among=Highest(Tally(AntiPlurality()), n=2,
                      among=Highest(Tally(Plurality()), n=3)),
    )
    assert shapes(2, 2) == json.loads(
        json.dumps(regions(expected, CANDIDATES.tolist(), MODELS[0], 64))
    )


def test_d21_takes_the_power_slider_but_not_the_categories():
    """D21's levels follow from its votes; its power is Score's."""
    client = TestClient(app)
    config = client.get("/api/config").json()
    assert "d21" in config["scores"] and "d21" not in config["score_levels"]

    def shapes(levels, power):
        response = client.post("/api/regions", json={
            "candidates": CANDIDATES.tolist(), "method": "d21", "levels": levels, "power": power, "grid": 64,
        })
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes(2, 2) == shapes(11, 2) != shapes(2, 1)
    expected = Highest(Tally(D21(max_positive=2, max_negative=1, negative_after=2, power=2)))
    assert shapes(2, 2) == json.loads(json.dumps(regions(expected, CANDIDATES.tolist(), MODELS[0], 64)))
    # an ordinal D21 has no power: neither slider applies
    ordinal = Highest(Tally(D21(ordinal=True)))
    assert _score_ballots(ordinal) == [] and _score_ballot_settings(ordinal, 2, 2) == ordinal


def test_hull_distance_is_zero_inside_and_the_distance_to_the_border_outside():
    square = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float))
    assert len(square) == 4
    points = np.array([[0.5, 0.5], [0.0, 0.3], [1.5, 0.5], [2.0, 2.0], [0.5, -0.25]])
    np.testing.assert_allclose(hull_distance(points, square), [0, 0, 0.5, np.sqrt(2), 0.25], atol=1e-12)
    segment = convex_hull(np.array([[0.2, 0.2], [0.6, 0.2], [0.4, 0.2]]))
    assert len(segment) == 2
    np.testing.assert_allclose(hull_distance(np.array([[0.4, 0.5], [0.0, 0.2], [0.8, 0.6]]), segment),
                               [0.3, 0.2, np.hypot(0.2, 0.4)], atol=1e-12)
    np.testing.assert_allclose(hull_distance(np.array([[0.5, 0.6]]), convex_hull(np.array([[0.2, 0.2]]))), [0.5])


def test_focus_hull_takes_in_the_voronoi_vertices_near_the_candidates():
    """Four candidates with one inside the triangle of the others: the three Voronoi vertices
    around it lie outside the hull of the candidates, near it, and are added to it."""
    candidates = np.array([[0.562, 0.704], [0.297, 0.561], [0.567, 0.399], [0.472, 0.552]])
    hull = focus_hull(candidates, 0.2)
    assert len(convex_hull(candidates)) == 3 and len(hull) == 6
    assert (hull_distance(candidates, hull) == 0).all()
    assert hull_distance(np.array([[0.2, 0.2]]), hull)[0] > 0


def test_focus_hull_leaves_out_far_vertices_and_those_outside_the_square():
    flat = np.array([[0.2, 0.5], [0.5, 0.59], [0.8, 0.5]])  # one vertex, (0.5, 0.045), 0.455 below them
    np.testing.assert_array_equal(focus_hull(flat, 0.2), convex_hull(flat))
    assert len(focus_hull(flat, 0.25)) == 4
    flatter = np.array([[0.2, 0.5], [0.5, 0.51], [0.8, 0.5]])  # its vertex is below the square
    assert len(focus_hull(flatter, 100.0)) == 3
    triangle = np.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.65]])  # acute: its vertex is inside the hull
    np.testing.assert_array_equal(focus_hull(triangle, 0.2), convex_hull(triangle))
    # no vertices: two candidates, or all on one line
    for line in (np.array([[0.2, 0.2], [0.6, 0.7]]), np.array([[0.1, 0.1], [0.4, 0.4], [0.8, 0.8]])):
        assert len(focus_hull(line, 0.2)) == 2


def test_focus_hull_catches_a_vertex_a_deviation_and_a_third_away():
    """Three candidates whose Voronoi vertex, (0.437, 0.725), is 0.263 from their hull: 1.3
    deviations at D = 0.2, within the reach, so the region takes it in."""
    candidates = np.array([[0.212, 0.406], [0.774, 0.530], [0.425, 0.336]])
    hull = focus_hull(candidates, 0.2)
    assert len(hull) == 4
    assert hull_distance(np.array([[0.437, 0.725]]), hull)[0] == pytest.approx(0, abs=1e-3)


def test_focus_hull_reach_is_in_deviations():
    """A vertex farther than FOCUS_REACH deviations from the hull is left out, a nearer one kept."""
    candidates = np.array([[0.3, 0.5], [0.7, 0.5], [0.5, 0.6]])
    vertex = Voronoi(candidates).vertices[0]
    away = hull_distance(vertex[None], convex_hull(candidates))[0]
    assert away > 0
    assert len(focus_hull(candidates, away / FOCUS_REACH * 1.01)) == 4
    assert len(focus_hull(candidates, away / FOCUS_REACH * 0.99)) == 3


def test_focus_outline_is_the_focus_hull_rounded_by_a_quarter_of_the_deviation():
    candidates = np.array([[0.3, 0.3], [0.7, 0.35], [0.5, 0.7], [0.45, 0.45]])
    deviation = 0.2
    outline, hull = focus_outline(candidates, deviation), focus_hull(candidates, deviation)
    radius = FOCUS_ROUND * deviation
    assert (hull_distance(hull, outline) == 0).all()  # the hull is inside the outline
    distances = hull_distance(outline, hull)
    assert distances.max() == pytest.approx(radius) and distances.min() > radius * np.cos(np.pi / OUTLINE_STEPS) - 1e-12


def test_one_metric_averages_over_the_focus_region_the_other_over_the_diagram():
    """Two candidates, the right one winning everywhere above y = 0.8, far from both: with
    deviation 0.2 (the region is within 0.05 of the segment) that error is not in the metric of
    the region, but in that of the diagram. With a region that covers the diagram, the two are
    the same: one formula."""
    candidates = np.array([[0.3, 0.5], [0.7, 0.5]])
    points = _grid_points(100)
    winner = nearest(candidates, points)[0].copy()
    winner[points[..., 1] > 0.8] = 1
    near = voronoi_agreement(candidates, points, winner, 0.2)
    assert near["region"]["error"] == pytest.approx(0)
    assert near["diagram"]["error"] > 0
    assert voronoi_agreement(candidates, points, winner, 1.5)["region"]["error"] > 0  # it reaches y = 0.875
    assert voronoi_agreement(candidates, points, winner, 10.0)["region"] == near["diagram"]


def test_metric_is_none_when_the_region_has_no_decided_point():
    points = _grid_points(40)
    winner = np.full(points.shape[:2], CYCLE)
    nothing = {"error": None, "random": None, "relative": None}
    assert voronoi_agreement(CANDIDATES, points, winner, 0.2) == {"diagram": nothing, "region": nothing}


@pytest.mark.parametrize("distribution", ["beta", "normal"])
@pytest.mark.parametrize("method", [name for name in DIAGRAMS if name in TESTED_METHODS or name not in MARGINS])
def test_api_returns_regions(distribution, method):
    client = TestClient(app)
    response = client.post("/api/regions", json={
        "candidates": CANDIDATES.tolist(), "method": method, "distribution": distribution,
        "deviation": 0.2, "grid": 64,
    })
    assert response.status_code == 200
    payload = response.json()
    shapes = payload["regions"]
    assert shapes and all(region["polygons"] for region in shapes)
    assert {"diagram", "region"} == set(payload["metrics"])
    for metrics in payload["metrics"].values():
        assert {"error", "random", "relative"} == set(metrics)
        assert 0 <= metrics["error"] <= 100 and 0 < metrics["random"] <= 100 and metrics["relative"] >= 0
    overlays = payload["overlays"]
    assert overlays["voronoi_boundaries"]
    assert all(len(segment) == 4
               for segment in overlays["voronoi_boundaries"])


def test_api_reports_perfect_metrics_for_voronoi():
    response = TestClient(app).post("/api/regions", json={
        "candidates": CANDIDATES.tolist(), "method": "voronoi", "grid": 64,
    })

    assert response.status_code == 200
    for metrics in response.json()["metrics"].values():
        assert metrics["error"] == pytest.approx(0) and metrics["relative"] == pytest.approx(0)


@pytest.mark.parametrize("pixel_median", ["marginal", "geometric"])
def test_api_compares_with_voronoi_only_when_asked(pixel_median):
    """The UI drags without the metrics and the Voronoi borders: the regions stay the same."""
    client = TestClient(app)
    request = {"candidates": CANDIDATES.tolist(), "method": "irv", "grid": 64, "pixel_median": pixel_median}
    full = client.post("/api/regions", json=request).json()
    bare = client.post("/api/regions", json={**request, "metrics": False, "overlay": False}).json()
    overlay = client.post("/api/regions", json={**request, "metrics": False}).json()

    assert bare["regions"] == overlay["regions"] == full["regions"]
    assert bare["metrics"] is None and set(bare["overlays"]) == {"focus"}
    assert bare["overlays"]["focus"] == full["overlays"]["focus"]
    assert overlay["metrics"] is None
    assert overlay["overlays"]["voronoi_boundaries"] == full["overlays"]["voronoi_boundaries"]
    assert client.post("/api/regions", json={**request, "overlay": False}).json()["metrics"] == full["metrics"]


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
    of the way to it; the other score methods take them too. They default to what the method is listed with (6 and 1.5), the
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


def test_api_builds_majority_judgment_from_levels_and_power():
    """Majority judgment takes the score sliders like Score, on the same ballot, and
    differs from it; with two levels it is approval voting, Score's winners. Strategic
    voters vote in it, and change it."""
    client = TestClient(app)

    def shapes(method="mj", **settings):
        request = {"candidates": CANDIDATES.tolist(), "method": method, "grid": 64, **settings}
        response = client.post("/api/regions", json=request)
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes() == shapes(levels=6, power=1.5) != shapes("score")
    expected = Highest(Tally(Score(4, power=2), statistic="median"))
    assert shapes(levels=4, power=2) == json.loads(json.dumps(regions(expected, CANDIDATES.tolist(), MODELS[0], 64)))
    assert shapes(levels=4, power=2) != shapes(levels=6, power=2)
    for power in (1, 2):
        _, median, margin = winners(Highest(Tally(Score(2, power=power), statistic="median")), CANDIDATES, MODELS[0], 64)
        _, mean, _ = winners(Highest(Tally(Score(2, power=power))), CANDIDATES, MODELS[0], 64)
        np.testing.assert_array_equal(median[margin > 1e-9], mean[margin > 1e-9])
    config = client.get("/api/config").json()
    assert "mj" in config["scores"] and "mj" in config["score_levels"] and "mj" not in config["score_comparisons"]
    assert not any("mj" in config["unsupported"][name] for name in ("bullet", "burial", "compromise"))
    method = next(option for option in config["methods"] if option["name"] == "mj")
    assert method["label"] == "Majority judgment"
    assert method["description"].endswith('\nHighest(Tally(Score(6, power=1.5), statistic="median"))')
    for strategy in ("bullet", "burial", "compromise"):
        assert shapes(**{strategy: 0.5}) != shapes(), strategy


def test_api_offers_deviations_from_005():
    """Every offered deviation has voters; 0 (all voters at their pixel) is not offered."""
    client = TestClient(app)
    config = client.get("/api/config").json()
    assert config["deviations"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    assert [voters["deviation"] for voters in config["voters"]] == config["deviations"]
    request = {"candidates": CANDIDATES.tolist(), "method": "score", "grid": 64}
    assert client.post("/api/regions", json={**request, "deviation": 0.05}).status_code == 200
    assert client.post("/api/regions", json={**request, "deviation": 0}).status_code == 422


def test_api_tilts_every_strategy_to_the_left():
    """The tilt slider tilts every strategy to the left by its strength, tilt=(-l, 0): the
    regions of the method built so; 0 is the strategies as they are, and without strategic
    voters it changes nothing. Its strength runs from 0 to MAX_TILT."""
    client = TestClient(app)

    def shapes(method="borda", **settings):
        request = {"candidates": CANDIDATES.tolist(), "method": method, "grid": 32, **settings}
        response = client.post("/api/regions", json=request)
        assert response.status_code == 200
        return response.json()["regions"]

    config = client.get("/api/config").json()["tilt"]
    assert config["max"] == MAX_TILT and config["step"] == TILT_STEP and config["label"]
    built = Bullet(Burial(METHODS["borda"], 0.5, tilt=(-3.0, 0.0)), 0.2, tilt=(-3.0, 0.0))
    assert shapes(burial=0.5, bullet=0.2, tilt=3) == json.loads(json.dumps(regions(built, CANDIDATES.tolist(),
                                                                                   MODELS[0], 32)))
    assert shapes(burial=0.5, tilt=3) != shapes(burial=0.5) == shapes(burial=0.5, tilt=0)
    assert shapes(tilt=3) == shapes()
    for tilt in (-0.5, MAX_TILT + 0.5, "x"):
        request = {"candidates": CANDIDATES.tolist(), "method": "borda", "burial": 0.5, "tilt": tilt}
        assert client.post("/api/regions", json=request).status_code == 422, tilt
