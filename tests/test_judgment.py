"""Checks of the majority judgment of a score ballot (build/judgment.py,
Tally(ballot, statistic="median")).

The majority grade must be the lower median of the shares of the scores, also where the
voters split evenly and where no voter gives some of the scores. On the shares of n
explicit ballots the order must be Balinski and Laraki's, who take the majority grade out
again and again (references.majority_value), for every n, and the same for the electorate
replicated: the shares of a distribution of voters assume no n. Candidates with the same
median go by that tie-break, the same distributions tie and go by the lowest index, and
where only the rule of the lower median decides, the candidate it puts ahead wins with a
margin of 0. With two levels it is approval voting, with the margin of the mean. The
totals must be the order with the gap of each place, every cost of every threshold
counted (references.majority_totals_reference); the remaining candidates of an
elimination keep their scores; and no change of every candidate's shares by less than
the margin in all changes the winner. Where nearly all voters give two candidates the
top score, the lead of the few who do not is kept.
"""

import numpy as np
import pytest

from yeelab.build import (Eliminate, Graded, Highest, Plurality, Score, ScoreAvg, ScoreDH, Scored, Tally,
                          Voters)
from yeelab.build.judgment import majority_grades, majority_totals

from references import judgment_order, majority_totals_reference, majority_value


def _judgment(levels):
    """Majority judgment on a Score ballot with `levels` scores."""
    return Highest(Tally(Score(levels), statistic="median"))


def _voters(grades):
    """The voters of the shares grades (..., C, levels) of each score of a Score ballot."""
    grades = np.asarray(grades, dtype=np.float64)
    return Voters(graded={Graded(Scored(grades.shape[-1])): grades})


@pytest.mark.parametrize("shares, grade", [
    ([0.1, 0.2, 0.3, 0.4], 2),      # at most 1: 0.3, at most 2: 0.6, the first at least half
    ([0.25, 0.25, 0.25, 0.25], 1),  # an even split: the lower of the two middle scores
    ([0.5, 0.0, 0.0, 0.5], 0),      # an even split across scores no one gives
    ([0.4, 0.0, 0.0, 0.6], 3),      # past the scores no one gives
    ([0.0, 0.0, 1.0, 0.0], 2),      # everyone gives the same score
    ([2, 1, 1, 0], 0),              # counts of ballots: 2 of 4 give 0
    ([1, 1, 1, 1], 1),
    ([0, 3], 1),
])
def test_the_majority_grade_is_the_lower_median(shares, grade):
    """The smallest score with at least half of the voters at it or below, as the
    procedure on ballots takes first."""
    assert majority_grades(shares) == grade
    if all(float(share).is_integer() for share in shares):
        assert majority_value(shares)[0] == grade


def test_the_order_is_that_of_n_ballots_for_every_n():
    """Random grades of n ballots, n from 1 to 30, with ties of every kind (clones,
    equal medians, as many above and below them, even splits): the totals are in the
    order of the majority values, Highest elects the first of it, and the same electorate
    three times over has the same totals, bit for bit."""
    rng = np.random.default_rng(0)
    seen = {"medians tie": 0, "shares above and below tie": 0, "same distributions": 0,
            "behind by the rule alone": 0}
    for trial in range(3000):
        n, levels, ballots = int(rng.integers(2, 7)), int(rng.integers(2, 7)), int(rng.integers(1, 31))
        counts = np.stack([rng.multinomial(ballots, row) for row in rng.dirichlet(np.full(levels, 0.5), size=n)])
        if trial % 3 == 0:
            counts[rng.integers(n)] = counts[rng.integers(n)]
        expected = judgment_order(counts)
        totals = majority_totals(counts)
        assert list(np.argsort(-totals, kind="stable")) == expected, counts
        winner, margin = _judgment(levels).evaluate(_voters(counts[None]))
        assert winner[0] == expected[0]
        np.testing.assert_array_equal(majority_totals(3 * counts), totals)
        first, second = expected[:2]
        grades = majority_grades(counts)
        same = (counts[first] == counts[second]).all()
        seen["medians tie"] += grades[first] == grades[second]
        # the same median and as many above and below it: only a later grade tells them apart
        median = grades[first]
        above, below = counts[:, median + 1:].sum(axis=-1), counts[:, :median].sum(axis=-1)
        seen["shares above and below tie"] += (grades[second] == median and not same and above[first] == above[second]
                                               and below[first] == below[second])
        seen["same distributions"] += same
        seen["behind by the rule alone"] += margin[0] < 1e-300 and first > second
    assert min(seen.values()) > 30, seen  # not vacuous


def test_equal_medians_go_by_the_tie_break():
    """Every median is 1 of 0 to 2. Of six ballots, A has two above 1 and one below, B one
    of each and C one above and two below: taking the 1s out, A's median rises first,
    C's falls first, and B's stays the longest. The gaps are 1/6 (one ballot), whatever
    the order of the candidates."""
    a, b, c = [1, 3, 2], [1, 4, 1], [2, 3, 1]
    assert [majority_value(x) for x in (a, b, c)] == [[1, 1, 1, 2, 0, 2], [1, 1, 1, 1, 0, 2], [1, 1, 0, 1, 0, 2]]
    np.testing.assert_allclose(majority_totals([a, b, c]), [0, -1 / 6, -1 / 3], rtol=0, atol=1e-15)
    np.testing.assert_allclose(majority_totals([c, b, a]), [-1 / 3, -1 / 6, 0], rtol=0, atol=1e-15)
    winner, margin = _judgment(3).evaluate(_voters([[c, b, a]]))
    assert winner[0] == 2
    np.testing.assert_allclose(margin, [1 / 6], rtol=0, atol=1e-15)


def test_an_even_split_is_behind_by_the_rule_of_the_lower_median():
    """A (first) has one ballot below its median 1 and one above it, B all four at 1.
    Taking out the 1s leaves A an even split, whose lower median is 0: B is ahead, with a
    margin of 0, as any change moves A's median one way or the other. Highest gives it
    to B, not to the lower index."""
    a, b = [1, 2, 1], [0, 4, 0]
    assert majority_value(a) == [1, 1, 0, 2] and majority_value(b) == [1, 1, 1, 1]
    winner, margin = _judgment(3).evaluate(_voters([[a, b]]))
    assert winner[0] == 1 and 0 <= margin[0] < 1e-300
    totals = majority_totals([a, b])
    assert totals[0] < totals[1] == 0
    # a little more above A's median than below it, and A's median rises first
    for shares, ahead in (([0.24, 0.5, 0.26], 0), ([0.26, 0.5, 0.24], 1)):
        winner, margin = _judgment(3).evaluate(_voters([[shares, [0, 1, 0]]]))
        assert winner[0] == ahead
        np.testing.assert_allclose(margin, [0.02], rtol=1e-12)


def test_the_same_distributions_tie():
    """They tie, with the same totals behind a candidate ahead of both, and where they
    are ahead, with a margin of 0: Highest takes the lowest index, as for any totals."""
    grades = [[[1, 2, 3], [0, 3, 3], [1, 2, 3]], [[0, 3, 3], [1, 2, 3], [1, 2, 3]]]
    totals = majority_totals(grades)
    assert totals[0, 0] == totals[0, 2] < totals[0, 1] == 0
    assert totals[1, 1] == totals[1, 2] < totals[1, 0] == 0
    winner, margin = _judgment(3).evaluate(_voters([[[2, 2, 2]] * 3, [[1, 2, 3], [1, 2, 3], [3, 2, 1]]]))
    np.testing.assert_array_equal(winner, [0, 0])
    np.testing.assert_array_equal(margin, [0, 0])


def test_scores_no_one_gives():
    """The median is the score where the shares at most it reach half, past the scores
    no one gives. A (0.4 at 0, 0.6 at 3) has the median 3, B (half at 1, half at 2) 1,
    with an even split: the gap is 0.1, how much more of A's voters than of B's give
    more than 1. Shares of 0 tie with shares of 0."""
    winner, margin = _judgment(4).evaluate(_voters([[[0.4, 0, 0, 0.6], [0, 0.5, 0.5, 0]]]))
    assert winner[0] == 0
    np.testing.assert_allclose(margin, [0.1], rtol=1e-12)
    np.testing.assert_array_equal(majority_grades([[0.4, 0, 0, 0.6], [0, 0.5, 0.5, 0]]), [3, 1])
    # only the top score and 0, as with two candidates: the larger share at the top
    winner, margin = _judgment(6).evaluate(_voters([[[0.7, 0, 0, 0, 0, 0.3], [0.3, 0, 0, 0, 0, 0.7]]]))
    assert winner[0] == 1
    np.testing.assert_allclose(margin, [0.4], rtol=1e-12)
    for wrong in ([[1, 0], [0, 0]], [[1, -0.1], [0.5, 0.5]], [[np.nan, 1], [0.5, 0.5]], [[1, np.inf], [1, 0]]):
        with pytest.raises(ValueError, match="the shares of every score must be finite and 0 or more"):
            majority_totals(wrong)
    with pytest.raises(ValueError, match="at least 2 levels"):
        majority_totals([[1.0], [1.0]])


def test_two_levels_are_approval_voting():
    """The most approvals win, by the margin of the mean, ties to the lowest index; on
    the shares of each ballot by every rule."""
    rng = np.random.default_rng(2)
    approved = rng.random((600, 5))
    approved[::5] = np.round(4 * approved[::5]) / 4  # ties, and even splits
    for ballot in (Score(2), ScoreAvg(2), ScoreDH(2)):
        voters = Voters(unscored={ballot.scored: 1 - approved},
                        graded={Graded(ballot.scored): np.stack([1 - approved, approved], axis=-1)})
        mean = Highest(Tally(ballot)).evaluate(voters)
        median = Highest(Tally(ballot, statistic="median")).evaluate(voters)
        np.testing.assert_array_equal(median[0], mean[0])
        np.testing.assert_allclose(median[1], mean[1], rtol=0, atol=1e-15)
        assert (mean[1] == 0).any()  # not vacuous


@pytest.mark.parametrize("levels", [2, 3, 6, 11])
def test_the_totals_are_the_order_with_the_gap_of_each_place(levels):
    """Bit for bit those of every cost of every threshold, among random remaining
    candidates (one at least), with clones and shares of eighths, where much ties."""
    rng = np.random.default_rng(levels)
    grades = rng.dirichlet(np.full(levels, 0.6), size=(300, 6))
    grades[::4, 3] = grades[::4, 1]
    grades[1::4] = np.round(8 * grades[1::4]) / 8
    alive = rng.random((300, 6)) < 0.7
    alive[np.arange(300), rng.integers(6, size=300)] = True
    totals = majority_totals(grades, alive)
    for p in range(len(grades)):
        order, expected = majority_totals_reference(grades[p], alive[p])
        np.testing.assert_array_equal(totals[p], expected)
        assert list(np.argsort(-totals[p][alive[p]], kind="stable")) == [
            int(np.flatnonzero(np.flatnonzero(alive[p]) == c)[0]) for c in order]
    # the remaining candidates as if they were all: the order among them and its gaps
    for p in range(0, len(grades), 7):
        np.testing.assert_array_equal(totals[p][alive[p]], majority_totals(grades[p][alive[p]]))
        assert (totals[p][~alive[p]] == -6).all()


def test_the_margin_bounds_the_change_that_keeps_the_winner():
    """Where no candidate's shares change by as much as the margin in all (the sum of
    |change| over its scores, the shares still adding up to 1), the winner stays: on
    random voters moved 0.99 of the way, and along paths between two of them, where the
    winner changes only between points whose margins are both at most the step."""
    rng = np.random.default_rng(3)
    method = _judgment(5)
    grades = rng.dirichlet(np.full(5, 0.7), size=(2000, 4))
    target = rng.dirichlet(np.full(5, 0.7), size=(2000, 4))
    winner, margin = method.evaluate(_voters(grades))
    distance = np.abs(target - grades).sum(axis=-1).max(axis=-1)
    part = np.minimum(1.0, 0.99 * margin / distance)[:, None, None]
    np.testing.assert_array_equal(method.evaluate(_voters(grades + part * (target - grades)))[0], winner)
    t = np.linspace(0, 1, 801)[:, None, None, None]
    path = (1 - t) * grades[:50] + t * target[:50]  # (801, 50, 4, 5)
    winner, margin = method.evaluate(_voters(path))
    change = winner[1:] != winner[:-1]
    step = (t[1] - t[0]).item() * distance[:50]
    assert change.sum() > 50  # not vacuous
    assert (np.maximum(margin[1:], margin[:-1]) <= step + 1e-12)[change].all()


def test_an_elimination_keeps_the_scores_of_the_remaining_candidates():
    """The order is among the remaining candidates and of the same scores: dropping the
    last one round by round leaves the first, among the finalists of a runoff the better
    one is elected, and the margins are those of the gaps between them."""
    rng = np.random.default_rng(4)
    grades = rng.dirichlet(np.full(6, 0.7), size=(500, 5))
    voters = Voters(first=rng.dirichlet(np.ones(5), size=500), graded={Graded(Scored(6)): grades})
    totals = Tally(Score(6), statistic="median")
    winner, margin = Highest(totals).evaluate(voters)
    eliminated, gap = Eliminate(totals, how="min").evaluate(voters)
    np.testing.assert_array_equal(eliminated, winner)
    assert (gap <= margin).all() and (gap < margin).any()
    finalists = Highest(Tally(Plurality()), n=2)
    runoff, runoff_margin = Highest(totals, among=finalists).evaluate(voters)
    chosen = finalists.select(voters)[0]
    ahead = majority_totals(grades, chosen)
    np.testing.assert_array_equal(runoff, np.where(chosen, ahead, -np.inf).argmax(axis=-1))
    lead = np.sort(np.where(chosen, ahead, -np.inf), axis=-1)
    np.testing.assert_array_equal(runoff_margin, np.minimum(finalists.select(voters)[1], lead[:, -1] - lead[:, -2]))


def test_a_lead_among_candidates_nearly_all_give_the_top_score_is_kept():
    """3e-30 and 1e-30 of the voters give two candidates less than the top score (all
    others give it): both majority grades are the top score, every share at most a score
    below it is 1 to rounding, and the second is still ahead by 2e-30, as on the mean."""
    grades = np.zeros((1, 3, 6))
    grades[0, 0, [0, 5]] = 3e-30, 1.0
    grades[0, 1, [4, 5]] = 1e-30, 1.0
    grades[0, 2, [0, 5]] = 0.6, 0.4
    winner, margin = _judgment(6).evaluate(_voters(grades))
    assert winner[0] == 1
    np.testing.assert_allclose(margin, [2e-30], rtol=1e-12, atol=0)
