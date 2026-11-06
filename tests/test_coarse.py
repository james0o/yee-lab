"""Checks of Coarse (build/blocks.py, score.COARSE): Score(6)'s ballot counted in groups.

A voter's score must be the score of Score(6) in its group, floor(s * levels / 6), at
any power, from the definition and from the compiled ballot table alike: the shares of
every score of Coarse(..., levels=levels) is the corresponding group of Score(6), and
Coarse(..., levels=2)
is Score(2). The block must print as it is built, keep its levels from the categories
slider of the UI but take the power and the trim slider (coarse_3 and coarse_2 are
trimmed means), and refuse unsupported level counts.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from yeelab.build import Coarse, CoarseTrim, Conditional, Highest, METHODS, Plurality, Score, Tally, Voters
from yeelab.build.blocks import TRIM_TIE_BREAK, _trimmed, _trimmed_sweep
from yeelab.build.voters import Graded, Scored
from yeelab.build.methods import TRIM
from yeelab.margin.regions import regions
from yeelab.margin.shares import Model, graded_shares, unscored_shares
from yeelab.score import COARSE, FINE, scored
from yeelab.voting import CYCLE
from yeelab.web.app import app

CANDIDATES = np.array([[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]])
MODEL = Model("beta", 0.2, "rms")


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test_a_score_is_its_group_of_six(power):
    points = np.random.default_rng(0).random((20000, 2))
    six = scored(points, CANDIDATES, FINE, power=power)
    for levels in (2, 3, 4, 6):
        np.testing.assert_array_equal(scored(points, CANDIDATES, levels, COARSE, power=power),
                                      six * levels // FINE)
    np.testing.assert_array_equal(scored(points, CANDIDATES, 2, COARSE, power=power),
                                  scored(points, CANDIDATES, 2, power=power))


def test_the_shares_are_those_of_six_in_groups():
    medians = np.linspace(0.05, 0.95, 7)
    six = graded_shares(CANDIDATES, MODEL, FINE, medians, power=1.5)
    for levels in (2, 3, 4, 6):
        grouped = np.stack(
            [six[..., np.arange(FINE) * levels // FINE == category].sum(axis=-1)
             for category in range(levels)],
            axis=-1,
        )
        coarse = graded_shares(CANDIDATES, MODEL, levels, medians, COARSE, power=1.5)
        np.testing.assert_allclose(coarse, grouped, atol=1e-14)
        unscored = unscored_shares(CANDIDATES, MODEL, levels, medians, COARSE, power=1.5)
        np.testing.assert_allclose(unscored, (grouped * np.arange(levels)[::-1]).sum(axis=-1) / (levels - 1),
                                   atol=1e-14)


def test_the_block():
    score_ballot = Score(6, power=1.5, fixed=True)
    ballot = Coarse(score_ballot, levels=3)
    assert repr(ballot) == "Coarse(Score(6, power=1.5, fixed=True), levels=3)"
    assert eval(repr(ballot)) == ballot
    assert ballot.fixed and ballot.scored.rule == COARSE and ballot != score_ballot
    assert Coarse(Score(6, fixed=True), levels=4).levels == 4
    assert Coarse(Score(6, fixed=True), levels=6).levels == 6
    assert METHODS["coarse_3"] == Highest(Tally(ballot, trim=TRIM))
    assert METHODS["coarse_2"] == Highest(Tally(Coarse(Score(6, power=1.5, fixed=True), levels=2), trim=TRIM))
    for levels in (1, 5, 12):
        with pytest.raises(ValueError):
            Coarse(Score(6, fixed=True), levels=levels)
    with pytest.raises(TypeError, match="fixed Score\\(6\\)"):
        Coarse(Score(6), levels=3)
    with pytest.raises(TypeError, match="fixed Score\\(6\\)"):
        Coarse(Score(5, fixed=True), levels=3)


def test_coarse_trim_has_winner_is_an_exact_conditional():
    """The existential trim condition finds winners between samples and is evaluable
    by Conditional without putting voter data in the method definition."""
    coarse = Coarse(Score(6, fixed=True), levels=4)
    condition = CoarseTrim(coarse).has_winner(Highest(Tally(Plurality())))
    point_grades = np.array([
        [0.4, 0.0, 0.0, 0.6],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    voters = Voters(
        first=np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
        graded={Graded(Scored(4, COARSE)): np.stack((point_grades, point_grades))},
    )
    result = Conditional(condition, when_true=Highest(Tally(coarse, trim=25)),
                         when_false=Highest(Tally(Plurality())))
    winner, margin = result.evaluate(voters)
    assert winner.shape == margin.shape == (2,)
    assert (margin >= 0).all()
    np.testing.assert_array_equal(condition.evaluate(voters)[0], [True, False])


def _sampled_sweep(grades, targets, trims):
    """Whether the trimmed mean (blocks._trimmed) elects each target at one of `trims`,
    and the largest lead of the target over the others among them."""
    totals = np.stack([_trimmed(grades, trim, None) for trim in trims])  # (T, P, C)
    points = np.arange(len(targets))
    chosen = totals[:, points, targets]
    others = totals.copy()
    others[:, points, targets] = -np.inf
    return (totals.argmax(axis=-1) == targets).any(axis=0), (chosen - others.max(axis=-1)).max(axis=0)


@pytest.mark.parametrize("levels", [3, 4, 6])
def test_the_trim_sweep_is_exact(levels):
    """_trimmed_sweep finds every winner a fine sample of the trims finds, and only those
    that win where the sample is too coarse to see: about where its lead is 0. Its margin
    is the largest lead, at least that of the sample and close to it (the sample comes
    near the trim 50 too: a threshold F(k) just below 1/2 leaves the trim only there). No winner (CYCLE, a
    draw) is not found."""
    rng = np.random.default_rng(levels)
    grades = rng.dirichlet(np.full(levels, 0.7), size=(3000, 5))
    grades[rng.random(grades.shape) < 0.1] = 0.0  # thresholds at 0 and 1, outside every trim
    grades[grades.sum(axis=-1) == 0] = 1.0
    grades[:300, 1] = grades[:300, 0]  # equal candidates: exact ties, the lower index wins
    targets = rng.integers(0, 5, len(grades))
    found, margin = _trimmed_sweep(grades, targets, TRIM_TIE_BREAK)
    trims = np.concatenate((np.linspace(0, 50, 5001)[:-1], 50 - np.logspace(-2, -9, 15)))  # and near the median
    sampled, lead = _sampled_sweep(grades, targets, trims)
    assert (found >= sampled).all()
    assert np.abs(margin[found != sampled]).max(initial=0) < 1e-3
    assert (margin >= 0).all() and (margin >= np.abs(lead) - 1e-12)[sampled].all()
    np.testing.assert_allclose(margin[found], lead[found], atol=1e-3)
    assert not found[:300][targets[:300] == 1].any()
    none, margin = _trimmed_sweep(grades[:2], np.array([CYCLE, -4]), TRIM_TIE_BREAK)
    assert not none.any() and np.isinf(margin).all()


def test_the_ui_fixes_the_levels_and_takes_the_power():
    client = TestClient(app)
    config = client.get("/api/config").json()
    for name, levels in (("coarse_3", 3), ("coarse_2", 2)):
        assert name in config["scores"] and name not in config["score_levels"]
        assert config["fixed_score_levels"][name] == [levels] and name in config["trims"]
        assert not any(name in config["unsupported"][strategy] for strategy in ("bullet", "burial", "compromise"))

    def shapes(**settings):
        request = {"candidates": CANDIDATES.tolist(), "method": "coarse_3", "grid": 64, **settings}
        response = client.post("/api/regions", json=request)
        assert response.status_code == 200
        return response.json()["regions"]

    assert shapes(levels=4, power=2) == shapes(levels=6, power=2) != shapes(levels=6, power=1)
    for trim in (0, 10):
        expected = Highest(Tally(Coarse(Score(6, power=2, fixed=True), levels=3), trim=trim))
        assert shapes(power=2, trim=trim) == json.loads(json.dumps(regions(expected, CANDIDATES.tolist(), MODEL, 64)))
