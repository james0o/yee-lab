"""Checks of the trimmed mean of a score ballot (Tally(ballot, trim=t), blocks._trimmed).

trim=0 must be the mean of Tally(ballot), and on the shares of n explicit ballots, where
t percent of n is whole, the trimmed mean of those n scores, as a part of the top score
minus 1, with TRIM_TIE_BREAK of the mean. Towards 50 it must near the median score where
the median is clear, and two candidates with the same median go by the mean. The totals
must be continuous in the shares, keep the scores of the remaining candidates of an
elimination and give -1 to the others, and the block must print as it is built and
refuse a trim outside 0 to 50, a trimmed median and a ballot without scores.
"""

import numpy as np
import pytest

from yeelab.build import (Eliminate, Graded, Highest, Lead, Plurality, Score, ScoreAvg, Scored, Tally, Top,
                          Voters)
from yeelab.build.blocks import TRIM_TIE_BREAK
from yeelab.build.judgment import majority_grades

EPS = TRIM_TIE_BREAK


def _voters(grades):
    """The voters of the shares grades (..., C, levels) of each score of a Score ballot."""
    grades = np.asarray(grades, dtype=np.float64)
    return Voters(graded={Graded(Scored(grades.shape[-1])): grades})


def test_no_trim_is_the_mean():
    """trim=0 is Tally(ballot); on the shares of every score its formula gives the same
    totals as the mean of the scores, counted down from 1."""
    assert Tally(Score(6), trim=0) == Tally(Score(6)) and repr(Tally(Score(6), trim=0)) == "Tally(Score(6))"
    grades = np.random.default_rng(0).dirichlet(np.full(6, 0.7), size=(50, 4))
    mean = (grades * np.arange(6)).sum(axis=-1) / 5 - 1
    tiny = Tally(Score(6), trim=1e-12).evaluate(_voters(grades))
    np.testing.assert_allclose(tiny, mean, atol=1e-9)


@pytest.mark.parametrize("n, trim", [(20, 25), (20, 10), (10, 40), (8, 12.5), (40, 45)])
def test_the_trimmed_mean_of_n_ballots(n, trim):
    """On the shares of n ballots, t percent of n taken off each end, the mean of the
    rest, and EPS of the mean."""
    rng = np.random.default_rng(n)
    levels = 5
    scores = rng.integers(0, levels, size=(30, 3, n))
    grades = np.stack([(scores == k).mean(axis=-1) for k in range(levels)], axis=-1)
    cut = round(n * trim / 100)
    assert cut == n * trim / 100
    kept = np.sort(scores, axis=-1)[..., cut:n - cut]
    expected = ((1 - EPS) * kept.mean(axis=-1) + EPS * scores.mean(axis=-1)) / (levels - 1) - 1
    np.testing.assert_allclose(Tally(Score(levels), trim=trim).evaluate(_voters(grades)), expected, atol=1e-12)


def test_towards_50_it_is_the_median():
    """Where no threshold has half of the voters on each side, the trimmed mean at 49.99
    is the median score, but for EPS of the mean."""
    grades = np.random.default_rng(3).dirichlet(np.ones(6), size=(200, 4))
    at_most = np.cumsum(grades[..., :-1], axis=-1)
    clear = (np.abs(at_most - 0.5) > 1e-3).all(axis=-1)
    totals = Tally(Score(6), trim=49.99).evaluate(_voters(grades))
    mean = (grades * np.arange(6)).sum(axis=-1) / 5 - 1
    expected = (1 - EPS) * (majority_grades(grades) / 5 - 1) + EPS * mean
    np.testing.assert_allclose(totals[clear], expected[clear], atol=1e-9)


def test_the_same_median_goes_by_the_mean():
    """Two candidates with the median 2 and every threshold outside the trim tie in the
    trimmed mean alone: the one with the higher mean is ahead, by EPS of its lead."""
    grades = np.array([[0.05, 0.1, 0.7, 0.1, 0.05], [0.0, 0.05, 0.7, 0.05, 0.2]])
    totals = Tally(Score(5), trim=40).evaluate(_voters(grades))
    lead = ((grades[1] - grades[0]) * np.arange(5)).sum() / 4
    assert lead > 0
    np.testing.assert_allclose(totals[1] - totals[0], EPS * lead, rtol=1e-9)
    winner, margin = Highest(Tally(Score(5), trim=40)).evaluate(_voters(grades))
    assert winner == 1 and margin > 0


def test_continuous_in_the_shares():
    """A change of the shares by d in all moves no total by more than d / (1 - 2a)."""
    rng = np.random.default_rng(4)
    grades = rng.dirichlet(np.ones(4), size=(100, 3))
    moved = np.abs(grades + rng.normal(scale=1e-6, size=grades.shape))
    moved /= moved.sum(axis=-1, keepdims=True)
    tally = Tally(Score(4), trim=30)
    change = np.abs(tally.evaluate(_voters(moved)) - tally.evaluate(_voters(grades)))
    assert (change <= np.abs(moved - grades).sum(axis=-1) / 0.4 + 1e-15).all()


def test_a_tally_like_any_other():
    """It reads the shares of every score, prints as it is built, ranks the remaining
    candidates of an elimination by their scores among all, and Top and Lead normalise
    it as a mean."""
    tally = Tally(Score(6, power=1.5), trim=25)
    assert repr(tally) == "Tally(Score(6, power=1.5), trim=25)" and eval(repr(tally)) == tally
    assert tally != Tally(Score(6, power=1.5)) and hash(tally) == hash(eval(repr(tally)))
    assert tally.needs == tally.needs_remaining == {Graded(Scored(6, power=1.5))}
    grades = np.random.default_rng(5).dirichlet(np.ones(6), size=(40, 5))
    voters = _voters(grades)
    tally = Tally(Score(6), trim=25)
    totals = tally.evaluate(voters)
    alive = np.random.default_rng(6).random(totals.shape) < 0.6
    np.testing.assert_array_equal(tally.evaluate(voters, alive), np.where(alive, totals, -1.0))
    winner, margin = Highest(tally).evaluate(voters)
    np.testing.assert_array_equal(winner, totals.argmax(axis=-1))
    top = np.sort(totals, axis=-1)
    np.testing.assert_allclose(margin, top[:, -1] - top[:, -2])
    np.testing.assert_allclose(Top(tally).evaluate(voters), top[:, -1] + 1)
    np.testing.assert_allclose(Lead(tally).evaluate(voters), top[:, -1] - top[:, -2])
    for how in ("min", "mean"):
        winner, _ = Eliminate(tally, how=how).evaluate(voters)
        assert winner.shape == (40,) and (winner >= 0).all()


@pytest.mark.parametrize("trim", [-1, 50, 60, np.inf, np.nan, True, "25"])
def test_a_trim_from_0_up_to_50(trim):
    with pytest.raises(ValueError, match="trim"):
        Tally(Score(6), trim=trim)


def test_only_the_mean_of_scores_is_trimmed():
    with pytest.raises(ValueError, match="trims only the mean"):
        Tally(Score(6), statistic="median", trim=25)
    with pytest.raises(TypeError, match="score ballot"):
        Tally(Plurality(), trim=25)
    assert Tally(ScoreAvg(3), trim=10).needs == {Graded(Scored(3, "avg"))}
