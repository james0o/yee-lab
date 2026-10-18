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

from yeelab.build import (D21, AntiPlurality, Conditional, Highest, Mix, Plurality, Score, ScoreAvg, ScoreCluster,
                          ScoreComparisons, ScoreDH, ScoreHybrid, Tally, Unbeaten)
from yeelab.margin.metrics import borders, voronoi_agreement
from yeelab.margin.regions import MARGINS, grid, nearest, regions, voronoi_borders, winners
from yeelab.margin.shares import Model, first_choice_shares, ranking_shares, score_comparison_shares, unscored_shares
from yeelab.voting import irv_rounds
from yeelab.web.app import DIAGRAMS, _score_ballot_settings, _score_ballots, app

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


def test_voronoi_agreement_is_perfect_for_the_reference_diagram():
    coordinates = (np.arange(100) + 0.5) / 100
    points = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
    winner, _ = nearest(CANDIDATES, points)
    shapes = regions("voronoi", CANDIDATES, None, 100)

    result = voronoi_agreement(CANDIDATES, points, winner, borders(shapes), borders(shapes))

    assert result["assignment"] == pytest.approx(100)
    assert result["inside_hull"] == pytest.approx(100)
    assert result["boundary_mean"] == pytest.approx(0)


def test_voronoi_agreement_has_no_inside_score_for_collinear_candidates():
    candidates = np.array([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]])
    coordinates = (np.arange(20) + 0.5) / 20
    points = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
    winner, _ = nearest(candidates, points)
    shapes = regions("voronoi", candidates, None, 20)

    assert voronoi_agreement(candidates, points, winner, borders(shapes), borders(shapes))["inside_hull"] is None


def test_wrong_assignments_reduce_regret_and_boundary_agreement():
    coordinates = (np.arange(80) + 0.5) / 80
    points = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
    winner, _ = nearest(CANDIDATES, points)
    incorrect = winner.copy()
    incorrect[:40] = (incorrect[:40] + 1) % len(CANDIDATES)
    shapes = regions("voronoi", CANDIDATES, None, 80)

    result = voronoi_agreement(CANDIDATES, points, incorrect, borders(shapes), borders(shapes))

    assert result["assignment"] < 100
    assert result["inside_hull"] < 100


def test_boundary_distance_is_relative_to_half_the_candidate_gap():
    from yeelab.margin.metrics import boundary_distance

    def cells(split):
        return [
            {"winner": 0, "polygons": [[
                [0, 0, split, 0, split, 1, 0, 1],
            ]]},
            {"winner": 1, "polygons": [[
                [split, 0, 1, 0, 1, 1, split, 1],
            ]]},
        ]

    # the boundaries are 0.1 apart everywhere: half of the gap of 0.2 between the candidates
    candidates = np.array([[0.3, 0.5], [0.7, 0.5]])
    assert boundary_distance(borders(cells(0.45)), borders(cells(0.55)), candidates, 0.4) == pytest.approx(0.5, abs=1e-6)
    # beyond a candidate a boundary cannot be any worse
    candidates = np.array([[0.45, 0.5], [0.55, 0.5]])
    assert boundary_distance(borders(cells(0.45)), borders(cells(0.55)), candidates, 0.1) == pytest.approx(1)


def test_boundary_distance_uses_perpendicular_projection_onto_segments():
    from yeelab.margin.metrics import _point_to_segments

    distance = _point_to_segments(
        np.array([[0.5, 0.3]]),
        np.array([[[0.2, 0.5], [0.8, 0.5]]]),
    )

    assert distance[0] == pytest.approx(0.2)


def test_point_to_segments_goes_to_the_nearer_end_beyond_a_segment():
    """Beyond its ends the nearest point of a segment is the nearer end, segments of no
    length are left out, and the distance is the smallest over all segments."""
    from yeelab.margin.metrics import _point_to_segments

    segments = np.array([[[0.0, 0.0], [1.0, 0.0]], [[2.0, 2.0], [2.0, 2.0]]])
    points = np.array([[0.5, 0.3], [-0.3, 0.4], [1.6, -0.8], [2.0, 2.1]])
    assert _point_to_segments(points, segments) == pytest.approx([0.3, 0.5, 1.0, 5.41 ** 0.5])

    rng = np.random.default_rng(0)
    points, segments = rng.random((50, 2)), rng.random((40, 2, 2))
    starts, directions = segments[:, 0], segments[:, 1] - segments[:, 0]
    t = np.clip(((points[:, None] - starts) * directions).sum(-1) / (directions ** 2).sum(-1), 0, 1)
    nearest = np.linalg.norm(points[:, None] - (starts + t[..., None] * directions), axis=-1).min(axis=1)
    assert _point_to_segments(points, segments) == pytest.approx(nearest, abs=1e-15)


@pytest.mark.parametrize("pixel_median, shift", [("marginal", 1.0), ("geometric", 1.0), ("geometric", 0.5)])
def test_exact_voronoi_borders_are_where_the_traced_ones_are(pixel_median, shift):
    """voronoi_borders are the borders regions() traces for the Voronoi diagram, with pixels
    at geometric medians only within the moved grid too. They differ only next to a point
    where three Voronoi cells meet, by less than a grid cell: there the traced ones stop
    short, and leave the bisector where a grid point's margin is to a third candidate."""
    from yeelab.margin.metrics import _point_to_segments, _sample_segments

    size = 160
    exact = voronoi_borders(CANDIDATES, MODELS[0], size, pixel_median, shift)
    traced = borders(regions("voronoi", CANDIDATES, MODELS[0], size, pixel_median, shift))
    assert {tuple(sorted(pair)) for pair in exact[1]} == {tuple(sorted(pair)) for pair in traced[1]}
    on_traced, on_exact = _sample_segments(traced[0])[0], _sample_segments(exact[0])[0]
    off = _point_to_segments(on_traced, exact[0])
    assert np.mean(off < 1e-6) > 0.95  # on the bisectors, to the six decimals of the traced vertices
    assert off.max() < 2 ** 0.5 / size
    assert _point_to_segments(on_exact, traced[0]).max() < 2 ** 0.5 / size


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
    methods = config["methods"]
    assert [m["name"] for m in methods] == list(MARGINS)
    assert [m["name"] for m in methods + config["ideals"]] == [*MARGINS, "voronoi"]
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
    assert {"assignment", "inside_hull", "boundary_mean"} == set(payload["metrics"])
    assert 0 <= payload["metrics"]["assignment"] <= 100
    assert payload["metrics"]["inside_hull"] is None or 0 <= payload["metrics"]["inside_hull"] <= 100
    boundary_mean = payload["metrics"]["boundary_mean"]
    assert boundary_mean is None or 0 <= boundary_mean <= 2 ** 0.5
    overlays = payload["overlays"]
    assert overlays["voronoi_boundaries"]
    assert all(len(segment) == 4
               for segment in overlays["voronoi_boundaries"])


def test_api_reports_perfect_metrics_for_voronoi():
    response = TestClient(app).post("/api/regions", json={
        "candidates": CANDIDATES.tolist(), "method": "voronoi", "grid": 64,
    })

    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert metrics["assignment"] == pytest.approx(100)
    assert metrics["inside_hull"] == pytest.approx(100)
    assert metrics["boundary_mean"] == pytest.approx(0)


@pytest.mark.parametrize("pixel_median", ["marginal", "geometric"])
def test_api_compares_with_voronoi_only_when_asked(pixel_median):
    """The UI drags without the metrics and the Voronoi borders: the regions stay the same."""
    client = TestClient(app)
    request = {"candidates": CANDIDATES.tolist(), "method": "irv", "grid": 64, "pixel_median": pixel_median}
    full = client.post("/api/regions", json=request).json()
    bare = client.post("/api/regions", json={**request, "metrics": False, "overlay": False}).json()
    overlay = client.post("/api/regions", json={**request, "metrics": False}).json()

    assert bare["regions"] == overlay["regions"] == full["regions"]
    assert bare["metrics"] is None and bare["overlays"] == {}
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
