"""Checks of the draws of the trimmed mean (Drawn, voting.draw) and of the web UI's
"show draws".

A draw is a winner of its own, -1 - the bits of its candidates, and decodes to them.
Drawn must be Highest of the trimmed mean wherever one candidate is ahead, with its lead
as the margin; where the trimmed means tie exactly it must choose them all, with the
margin of the formula, and the margins must be continuous where a draw begins. The
metrics count a draw as a candidate drawn at random among its own. The API draws them
with `draws` only for the methods that are the highest trimmed mean, and the chart of
winners by trim too, but for majority judgment at 50.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from yeelab.build import Coarse, Drawn, Graded, Highest, Score, Scored, Tally, Voters
from yeelab.build.blocks import _trimmed
from yeelab.margin.metrics import _losses
from yeelab.margin.regions import voters, winners
from yeelab.margin.shares import Model
from yeelab.voting import CYCLE, draw, drawn
from yeelab.web.app import DRAW_METHODS, TRIM_METHODS, TRIM_ROWS, TRIMS, app, strategic

CANDIDATES = [[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]]
MODEL = Model("beta", 0.2, "rms")


def _voters(grades):
    """The voters of the shares grades (..., C, levels) of each score of a Score ballot."""
    grades = np.asarray(grades, dtype=np.float64)
    return Voters(graded={Graded(Scored(grades.shape[-1])): grades})


def test_a_draw_is_a_winner_of_its_own():
    chosen = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool)
    winner = draw(chosen)
    assert winner.tolist() == [-6, -3, CYCLE, -16]
    np.testing.assert_array_equal(drawn(winner, 4), chosen)
    np.testing.assert_array_equal(drawn(np.array([2, CYCLE]), 4), [[0, 0, 1, 0], [0, 0, 0, 0]])
    assert (draw(chosen[[0, 3]]) < -3).all()  # two or more: never a candidate, CYCLE or one of them


def test_one_ahead_is_the_highest_trimmed_mean():
    """Without exact ties Drawn elects the highest trimmed mean, by its lead."""
    grades = np.random.default_rng(1).dirichlet(np.ones(6), size=(300, 4))
    tally = Tally(Score(6), trim=25)
    totals = _trimmed(grades, 25, None, tie_break=0)
    winner, margin = Drawn(tally).evaluate(_voters(grades))
    assert (winner >= 0).all()
    np.testing.assert_array_equal(winner, totals.argmax(axis=-1))
    top = np.sort(totals, axis=-1)
    np.testing.assert_allclose(margin, top[:, -1] - top[:, -2])
    np.testing.assert_array_equal(winner, Highest(tally).evaluate(_voters(grades))[0])


def test_the_same_median_is_a_draw():
    """Two candidates with the median 2 and every threshold outside the trim draw; the
    third is behind. The margin is the nearest threshold's distance from the trim, or the
    lead over the third, whichever is smaller."""
    grades = np.array([[0.05, 0.1, 0.7, 0.1, 0.05], [0.0, 0.05, 0.7, 0.05, 0.2], [0.3, 0.4, 0.2, 0.1, 0.0]])
    method = Drawn(Tally(Score(5), trim=40))
    winner, margin = method.evaluate(_voters(grades))
    assert winner == draw(np.array([True, True, False]))
    chosen, same = method.select(_voters(grades))
    assert chosen.tolist() == [True, True, False] and same == margin
    # F of the two: (0.05, 0.15, 0.85, 0.95) and (0, 0.05, 0.75, 0.8): the nearest to the
    # trim (0.4, 0.6) is 0.75, 0.15 from it, over (1 - 0.8) * 4
    totals = _trimmed(grades, 40, None, tie_break=0)
    lead = totals[0] - totals[2]
    assert totals[0] == totals[1] and lead > 0
    np.testing.assert_allclose(margin, min(0.15 / 0.8, lead))
    # where the median is not clear, no draw: Highest of the trimmed means
    grades[0] = [0.05, 0.1, 0.3, 0.5, 0.05]
    assert method.evaluate(_voters(grades))[0] == 0


def test_the_margins_meet_where_a_draw_begins():
    """Along a path of shares from a win of candidate 1 into a draw with candidate 0,
    the margin of each falls to 0 at the same point, with the winner changing there."""
    start = np.array([[0.05, 0.1, 0.7, 0.1, 0.05], [0.0, 0.05, 0.5, 0.05, 0.4]])
    end = np.array([[0.05, 0.1, 0.7, 0.1, 0.05], [0.0, 0.05, 0.7, 0.05, 0.2]])
    steps = np.linspace(0, 1, 2001)[:, None, None]
    winner, margin = Drawn(Tally(Score(5), trim=40)).evaluate(_voters(start + steps * (end - start)))
    both = draw(np.array([True, True]))
    assert winner[0] == 1 and winner[-1] == both and set(winner.tolist()) == {1, both}
    change = np.flatnonzero(winner[1:] != winner[:-1])
    assert len(change) == 1
    k = change[0]
    assert margin[k] < 1e-3 and margin[k + 1] < 1e-3 and margin.max() > 0.05


def test_no_draws_of_a_median():
    with pytest.raises(ValueError, match="median"):
        Drawn(Tally(Score(6), statistic="median"))
    with pytest.raises(TypeError):
        Drawn(Highest(Tally(Score(6), trim=25)))
    assert repr(Drawn(Tally(Score(6), trim=25))) == "Drawn(Tally(Score(6), trim=25))"


def test_a_draw_loses_as_a_random_one_of_its_own():
    candidates = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.9]])
    points = np.random.default_rng(2).random((6, 5, 2))
    every = _losses(candidates, points, np.zeros((6, 5), dtype=int), 0.5)  # the mean of all: [1]
    singles = [_losses(candidates, points, np.full((6, 5), c), 0.5)[0] for c in range(3)]
    loss, random, decided = _losses(candidates, points, np.full((6, 5), draw(np.array([True, False, True]))), 0.5)
    np.testing.assert_allclose(loss, (singles[0] + singles[2]) / 2)
    np.testing.assert_allclose(random, every[1])
    assert decided.all() and not _losses(candidates, points, np.full((6, 5), CYCLE), 0.5)[2].any()


def test_coarse_ballots_draw_over_areas():
    """Approvals counted with a trim of 25 draw over a part of the default diagram;
    elsewhere the winners are those of the trimmed mean."""
    coarse_2 = Coarse(Score(6, power=1.5, fixed=True), levels=2)
    method = Drawn(Tally(coarse_2, trim=25))
    _, winner, margin = winners(method, CANDIDATES, MODEL, 64)
    _, usual, _ = winners(Highest(Tally(coarse_2, trim=25)), CANDIDATES, MODEL, 64)
    draws = winner < CYCLE
    assert 0.05 < draws.mean() < 0.5
    np.testing.assert_array_equal(winner[~draws], usual[~draws])
    assert (drawn(winner[draws], len(CANDIDATES)).sum(axis=-1) >= 2).all() and (margin >= 0).all()


def test_api_draws_for_the_highest_trimmed_means():
    assert DRAW_METHODS == ["trimmed", "coarse_3", "coarse_2"]
    assert set(DRAW_METHODS) <= set(TRIM_METHODS)
    client = TestClient(app)
    request = {"candidates": CANDIDATES, "method": "coarse_2", "trim": 25, "grid": 48, "overlay": False}
    shown = client.post("/api/regions", json={**request, "draws": True}).json()
    assert any(region["winner"] < CYCLE for region in shown["regions"])
    assert shown["metrics"]["diagram"]["relative"] is not None
    usual = client.post("/api/regions", json=request).json()
    assert all(region["winner"] >= CYCLE for region in usual["regions"])
    # majority judgment and the methods that only take a trimmed mean in a condition: as they are
    for other in ({"trim": 50}, {"method": "score"}, {"method": "exp"}):
        asked = {**request, **other}
        assert client.post("/api/regions", json={**asked, "draws": True}).json()["regions"] == \
            client.post("/api/regions", json=asked).json()["regions"]


def test_api_draws_in_the_chart_of_winners_by_trim():
    client = TestClient(app)
    median = [0.45, 0.45]
    answer = client.post("/api/trims", json={"candidates": CANDIDATES, "median": median, "power": 1.5,
                                             "draws": True}).json()
    for row, ballot in zip(answer["rows"], (
            Coarse(Score(6, power=1.5, fixed=True), levels=2),
            Coarse(Score(6, power=1.5, fixed=True), levels=3),
            Score(6, power=1.5))):
        assert (row["name"], row["levels"]) in TRIM_ROWS
        for trim in (0, 25, 40, 49):
            method = Drawn(Tally(ballot, trim=trim))
            expected = method.evaluate(voters(method.needs, CANDIDATES, MODEL, medians=median))[0][0, 1]
            assert row["winners"][TRIMS.index(trim)] == expected, (row["name"], trim)
        assert row["winners"][-1] >= CYCLE  # majority judgment
    assert any(winner < CYCLE for row in answer["rows"] for winner in row["winners"])
    # and the voters vote as the sliders say
    burying = client.post("/api/trims", json={"candidates": CANDIDATES, "median": median, "power": 1.5,
                                              "draws": True, "burial": 0.4}).json()
    method = strategic(Drawn(Tally(Coarse(Score(6, power=1.5, fixed=True), levels=2), trim=40)), {"burial": 0.4})
    expected = method.evaluate(voters(method.needs, CANDIDATES, MODEL, medians=median))[0][0, 1]
    assert burying["rows"][0]["winners"][TRIMS.index(40)] == expected
