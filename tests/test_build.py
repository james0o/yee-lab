"""Checks of the blocks methods are built from (yeelab.build.blocks).

A ballot's fast tally must be the same sum over the profile for any remaining
candidates. Highest chooses the n highest totals and is Eliminate(totals, how="all"); a
block that chooses several candidates is no method, and first | second and Fallback
choose sets. The score ballots tally the parts of the top score not given, which
test_score.py checks; here the highest mean score must win (with two levels: the most
approved candidate), with its lead as the margin, also between candidates nearly all
voters give the top score. A Mix of two ballots must tally the mix of their voters, and
be the one ballot itself where all voters mark it. Ties go to the lowest index.

METHODS are the fifteen expressions, every block prints as the expression that builds
it, needs the shares it reads, and fails when built from wrong blocks or arguments. The
methods themselves are checked against references in test_methods.py, the strategic
voters in test_strategic.py.
"""

import re
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from yeelab.build import (
    FIRST,
    METHODS,
    PAIRWISE,
    PROFILE,
    AntiPlurality,
    BordaCount,
    DescendingCoalitions,
    Eliminate,
    Fallback,
    Highest,
    Margins,
    Mix,
    Pairwise,
    Plurality,
    Schedules,
    Score,
    ScoreAvg,
    ScoreCluster,
    ScoreComparisons,
    ScoreDH,
    ScoreHybrid,
    Scored,
    ScoredPairwise,
    StrongestPaths,
    Tally,
    Unbeaten,
    Union,
    Voters,
    Weakest,
)
from yeelab.voting import CYCLE

from references import TOLERANCE, ranking_weights, voters_of

BALLOTS = [Plurality(), BordaCount(), AntiPlurality()]
# the blocks below the methods that are no method themselves, and Weakest
PAIR_BLOCKS = [Pairwise(), Margins(Pairwise()), StrongestPaths(Margins(Pairwise())),
               StrongestPaths(StrongestPaths(Margins(Pairwise()))), Weakest(Margins(Pairwise()))]


@pytest.mark.parametrize("ballot", BALLOTS, ids=repr)
def test_tally_is_the_sum_of_weights(profile, ballot):
    """For all candidates and for random remaining ones at every point (at least one)."""
    rankings, probs = profile
    voters, n = voters_of(rankings, probs), rankings.shape[1]
    everyone = probs @ ranking_weights(ballot, rankings, np.ones(n, dtype=bool))
    np.testing.assert_allclose(ballot.tally(voters, None), everyone, rtol=0, atol=1e-5)

    rng = np.random.default_rng(n)
    alive = rng.random(probs.shape[:-1] + (n,)) < 0.5
    alive |= np.arange(n) == rng.integers(n, size=probs.shape[:-1] + (1,))
    codes = alive @ (1 << np.arange(n))
    expected = np.empty(alive.shape)
    for code in np.unique(codes):
        at = codes == code
        expected[at] = probs[at] @ ranking_weights(ballot, rankings, (code >> np.arange(n)) & 1)
    np.testing.assert_allclose(ballot.tally(voters, alive), expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(Tally(ballot).evaluate(voters, alive), ballot.tally(voters, alive))


def test_highest_chooses_the_n_highest_totals():
    """Highest(totals, n) is Eliminate(totals, how="all", until=n) by its own name: the n
    highest totals (ties: the lowest index), with the n-th highest minus the next as the
    margin; with no more than n candidates all of them, margin inf. It reads any totals
    once."""
    voters = _scoring_voters()
    totals, shares = Tally(Score(6)), voters.unscored[Scored(6)]
    order = np.argsort(shares, axis=-1, kind="stable")  # the fewest not given first
    ranked = np.take_along_axis(shares, order, axis=-1)
    for n in (1, 2, 4):
        highest, eliminate = Highest(totals, n=n), Eliminate(totals, how="all", until=n)
        assert isinstance(highest, Eliminate) and highest != eliminate
        assert highest.n == highest.until == highest.seats == n and highest.how == "all"
        chosen, margin = highest.select(voters)
        for result, expected in zip((chosen, margin), eliminate.select(voters)):
            np.testing.assert_array_equal(result, expected)
        expected = np.zeros(shares.shape, dtype=bool)
        np.put_along_axis(expected, order[..., :n], True, axis=-1)
        np.testing.assert_array_equal(chosen, expected)
        np.testing.assert_array_equal(margin, ranked[..., n] - ranked[..., n - 1])
        assert (margin == 0).any() and (margin > 0).any()  # the ties are there
    chosen, margin = Highest(totals, n=5).select(voters)
    assert chosen.all() and (margin == np.inf).all()
    assert Highest(totals) == Highest(totals, 1) == Highest(totals=totals, n=1) != Highest(totals, n=2)


def test_a_method_chooses_one_candidate():
    """Blocks that choose several candidates only select(); evaluate() is for a winner.
    A Fallback of two such blocks chooses as many, a Union a number that varies."""
    two, irv = Highest(Tally(Plurality()), n=2), METHODS["irv"]
    assert [block.seats for block in (two, irv, two | irv, Fallback(two, two), Fallback(irv, two))] \
        == [2, 1, None, 2, None]
    assert all(method.seats == 1 for method in METHODS.values())
    assert Schedules(Tally(Plurality())).seats is None
    for block, chooses in ((two, "2 candidates"), (two | irv, "a number of candidates"),
                           (Eliminate(Tally(Plurality()), how="min", until=3), "3 candidates"),
                           (Schedules(Tally(Plurality())), "a number of candidates")):
        with pytest.raises(ValueError, match=re.escape(f"{block!r} chooses {chooses}, not a winner")):
            block.evaluate(Voters(first=np.full((2, 4), 0.25)))


def test_union_and_fallback_choose_sets(profile):
    """first | second is either one's candidates, with the smaller margin; a Fallback of
    sets takes second's where first chooses no one."""
    voters = voters_of(*profile)
    fptp, irv, condorcet, two = METHODS["fptp"], METHODS["irv"], METHODS["condorcet"], Highest(Tally(BordaCount()), n=2)
    chosen, margin = (fptp | irv).select(voters)
    (a, margin_a), (b, margin_b) = fptp.select(voters), irv.select(voters)
    np.testing.assert_array_equal(chosen, a | b)
    np.testing.assert_array_equal(margin, np.minimum(margin_a, margin_b))
    assert (chosen.sum(axis=-1) == 2).any() and (chosen.sum(axis=-1) == 1).any()  # not vacuous

    chosen, margin = Fallback(condorcet, two).select(voters)
    winner, margin_c = condorcet.evaluate(voters)
    cycle = winner == CYCLE
    finalists, margin_two = two.select(voters)
    np.testing.assert_array_equal(chosen[cycle], finalists[cycle])
    np.testing.assert_array_equal(chosen[~cycle].argmax(axis=-1), winner[~cycle])
    np.testing.assert_array_equal(margin, np.where(cycle, np.minimum(margin_c, margin_two), margin_c))


# the shares of the score ballots below: two and six levels, by every rule
SCORED = [Scored(levels, rule) for levels in (2, 6) for rule in ("range", "avg", "dhondt", "hybrid", "cluster")]
# two kinds of approving voters for the mixes: above their largest gap, and closer than
# their mean distance
BY_GAP, BY_MEAN = ScoreDH(2), ScoreAvg(2)


def _scoring_voters(n=5, seed=0):
    """Random parts of the top score not given (4, 6, n) for every ballot of SCORED;
    every fifth point on a grid of quarters, where they tie."""
    rng = np.random.default_rng(seed)
    unscored = {share: rng.random((4, 6, n)) for share in SCORED}
    for shares in unscored.values():
        shares.reshape(-1, n)[::5] = np.round(4 * shares.reshape(-1, n)[::5]) / 4
    return Voters(unscored=unscored)


def test_the_highest_mean_score_wins():
    """Highest(Tally(ballot)) on the shares of the ballot's own number of levels and
    rule: the candidate the voters give the largest part of the top score wins (ties: the
    lowest index), with two levels the most approved one, and the margin is its lead
    over the second. The tally is counted down from 1, the remaining candidates of an
    elimination keep theirs, and the others get -1. ScoreAvg, ScoreDH, ScoreHybrid and
    ScoreCluster read the shares of their own rules, not those of Score."""
    voters = _scoring_voters()
    assert voters.shape == (4, 6) and voters.n_candidates == 5
    for levels, (score, rule) in product((2, 6), ((Score, "range"), (ScoreAvg, "avg"), (ScoreDH, "dhondt"),
                                                   (ScoreHybrid, "hybrid"), (ScoreCluster, "cluster"))):
        ballot, shares = score(levels), voters.unscored[Scored(levels, rule)]
        assert ballot.needs == ballot.needs_remaining == {Scored(levels, rule)}
        np.testing.assert_array_equal(ballot.tally(voters, None), -shares)
        alive = np.random.default_rng(1).random(shares.shape) < 0.5
        np.testing.assert_array_equal(ballot.tally(voters, alive), np.where(alive, -shares, -1))
        winner, margin = Highest(Tally(ballot)).evaluate(voters)
        ordered = np.sort(shares, axis=-1)
        np.testing.assert_array_equal(winner, shares.argmin(axis=-1))
        np.testing.assert_array_equal(margin, ordered[..., 1] - ordered[..., 0])
        assert (margin == 0).any() and (margin > 0).any()  # the ties are there


def test_a_lead_among_candidates_nearly_all_give_the_top_score_is_kept():
    """Two candidates short of the top score by 3e-30 and 1e-30 of it (with two levels:
    not approved by 3e-30 and 1e-30 of the voters): both mean scores are the top score
    in floating point, and the second still wins by 2e-30."""
    unscored = np.array([[3e-30, 1e-30, 0.4], [1e-40, 2e-40, 1.0]])
    assert ((1 - unscored[:, 0]) == (1 - unscored[:, 1])).all()
    for ballot in (Score(6), ScoreAvg(2), ScoreDH(2), ScoreHybrid(6), ScoreCluster(2)):
        winner, margin = Highest(Tally(ballot)).evaluate(Voters(unscored={ballot.scored: unscored}))
        np.testing.assert_array_equal(winner, [1, 0])
        np.testing.assert_allclose(margin, [2e-30, 1e-40], rtol=1e-12, atol=0)


def test_a_score_ballot_is_not_marked_again_in_an_elimination():
    """The remaining candidates keep their totals and the others get -1, so dropping the
    lowest round by round leaves the candidate with the highest mean score."""
    voters, ballot = _scoring_voters(), BY_MEAN
    shares = voters.unscored[ballot.scored]
    alive = np.random.default_rng(1).random(shares.shape) < 0.5
    np.testing.assert_array_equal(Tally(ballot).evaluate(voters, alive), ballot.tally(voters, alive))
    winner, margin = Eliminate(Tally(ballot), how="min").evaluate(voters)
    clear = Highest(Tally(ballot)).evaluate(voters)[1] > 0
    np.testing.assert_array_equal(winner[clear], shares.argmin(axis=-1)[clear])
    ordered = np.sort(shares, axis=-1)
    np.testing.assert_allclose(margin, np.diff(ordered, axis=-1).min(axis=-1), rtol=0, atol=1e-12)


@pytest.mark.parametrize("share", [0, 0.25, 0.9, 1])
def test_a_mix_tallies_both_kinds_of_voters(share):
    """Mix(first, second, share): `share` of the voters mark second and the others first,
    so every total is that mix of the two, among any remaining candidates, and the
    candidate the fewest of all voters do not approve wins. At share 0 and 1 it is the
    one ballot all voters mark, bit for bit, from the shares of that ballot alone."""
    voters, ballot = _scoring_voters(), Mix(BY_GAP, BY_MEAN, share=share)
    unscored = (1 - share) * voters.unscored[BY_GAP.scored] + share * voters.unscored[BY_MEAN.scored]
    alive = np.random.default_rng(1).random(unscored.shape) < 0.5
    for remaining, expected in ((None, -unscored), (alive, np.where(alive, -unscored, -1))):
        np.testing.assert_allclose(ballot.tally(voters, remaining), expected, rtol=1e-12, atol=0)
    winner, margin = Highest(Tally(ballot)).evaluate(voters)
    ordered = np.sort(unscored, axis=-1)
    clear = ordered[..., 1] - ordered[..., 0] > 1e-12
    np.testing.assert_array_equal(winner[clear], unscored.argmin(axis=-1)[clear])
    np.testing.assert_allclose(margin, ordered[..., 1] - ordered[..., 0], rtol=0, atol=1e-12)
    if share in (0, 1):
        one = BY_GAP if share == 0 else BY_MEAN
        assert ballot.needs == ballot.needs_remaining == {one.scored}
        only = Voters(unscored={one.scored: voters.unscored[one.scored]})  # the other is not read
        for remaining in (None, alive):
            np.testing.assert_array_equal(ballot.tally(only, remaining), one.tally(voters, remaining))


def test_a_mix_keeps_a_lead_among_candidates_nearly_all_approve():
    """All but 3e-30 and 1e-30 of the one kind of voters and all but 1e-30 and 5e-30 of
    the other approve two candidates: of half of each, 2e-30 and 3e-30 do not."""
    unscored = {BY_GAP.scored: np.array([[3e-30, 1e-30, 0.4]]), BY_MEAN.scored: np.array([[1e-30, 5e-30, 0.4]])}
    winner, margin = Highest(Tally(Mix(BY_GAP, BY_MEAN, share=0.5))).evaluate(Voters(unscored=unscored))
    np.testing.assert_array_equal(winner, [0])
    np.testing.assert_allclose(margin, [1e-30], rtol=1e-12, atol=0)


def test_a_mix_of_ranked_ballots_is_eliminated_round_by_round(profile):
    """Mix takes any two ballots and has no formula of its own for an elimination:
    Eliminate tallies the remaining candidates again each round, from the shares both
    ballots need for that."""
    voters = voters_of(*profile)
    ballot = Mix(Plurality(), BordaCount(), share=0.3)
    assert ballot.needs == {FIRST, PAIRWISE} and ballot.needs_remaining == {PROFILE, PAIRWISE}
    np.testing.assert_allclose(ballot.tally(voters, None),
                               0.7 * voters.first + 0.3 * voters.pairwise.sum(axis=-1), rtol=0, atol=1e-6)
    method = Eliminate(Tally(ballot), how="min")
    assert method.needs == {PROFILE, PAIRWISE}
    winner, margin = method.evaluate(voters)
    alive = np.ones((*voters.shape, voters.n_candidates), dtype=bool)
    for _ in range(voters.n_candidates - 1):  # the lowest total of the remaining ones goes
        totals = np.where(alive, ballot.tally(voters, alive), np.inf)
        np.put_along_axis(alive, totals.argmin(axis=-1)[..., None], False, axis=-1)
    clear = margin > TOLERANCE
    np.testing.assert_array_equal(winner[clear], alive.argmax(axis=-1)[clear])


def test_antiplurality_and_top_two_among_top_three():
    rankings = np.array([[0, 1, 2, 3], [0, 1, 3, 2], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]])
    probs = np.array([[0.25, 0.15, 0.2, 0.1, 0.3]])
    voters = Voters(first=np.array([[0.4, 0.2, 0.1, 0.3]]), rankings=rankings, probs=probs)
    anti = AntiPlurality()
    assert anti.needs == anti.needs_remaining == {PROFILE}
    assert [anti.weight(k, 4) for k in range(4)] == [1, 1, 1, 0]
    np.testing.assert_allclose(anti.tally(voters, None), [[1.0, 1.0, 0.55, 0.45]])
    bullet_voters = Voters(rankings=rankings, probs=0.8 * probs, bullet=np.array([[0.2, 0, 0, 0]]))
    np.testing.assert_allclose(anti.tally(bullet_voters, None), [[1.0, 0.8, 0.44, 0.36]])

    top3 = Highest(Tally(Plurality()), n=3)
    top2 = Highest(Tally(anti), n=2, among=top3)
    selected, margin = top2.select(voters)
    np.testing.assert_array_equal(selected, [[True, True, False, False]])
    np.testing.assert_allclose(margin, [0.1])


def test_score_takes_the_number_of_levels():
    assert Score(3) == Score(levels=3) != Score(4)
    assert Score(3, fixed=True) != Score(3)
    assert repr(Score(3, power=1.5, fixed=True)) == "Score(3, power=1.5, fixed=True)"
    assert eval(repr(Score(3, power=1.5, fixed=True))) == Score(3, power=1.5, fixed=True)
    with pytest.raises(TypeError, match="Score fixed must be a bool"):
        Score(3, fixed=1)
    assert ScoreAvg(3) == ScoreAvg(levels=3) != ScoreAvg(4)
    assert ScoreDH(3) == ScoreDH(levels=3) != ScoreDH(4)
    assert ScoreHybrid(3) == ScoreHybrid(levels=3) != ScoreHybrid(4)
    assert ScoreCluster(3) == ScoreCluster(levels=3) != ScoreCluster(4)
    scores = (Score(3), ScoreAvg(3), ScoreDH(3), ScoreHybrid(3), ScoreCluster(3))
    assert len(set(scores)) == len({score.needs for score in scores}) == 5
    assert ([repr(score) for score in scores]
            == ["Score(3)", "ScoreAvg(3)", "ScoreDH(3, delta=0.8)", "ScoreHybrid(3, delta=0.8)",
                "ScoreCluster(3, mu=0.1, kappa=2.0)"])
    for score, levels in product((Score, ScoreAvg, ScoreDH, ScoreHybrid, ScoreCluster), (1, 0, -3, 2.5, "3", True)):
        with pytest.raises(ValueError, match=re.escape(
                f"{score.__name__} levels must be a whole number of at least 2, got {levels!r}")):
            score(levels)
    for score in (Score, ScoreAvg, ScoreDH, ScoreHybrid, ScoreCluster):
        with pytest.raises(TypeError):  # no cut or rule to choose: each block has its own
            score(3, "gap")
    # the divisor of ScoreDH and ScoreHybrid: each step to the largest gap / (its steps
    # + delta), by keyword
    for score, rule in ((ScoreDH, "dhondt"), (ScoreHybrid, "hybrid")):
        assert score(3) == score(3, delta=0.8) != score(3, delta=0.5)
        assert score(3, delta=1) == score(3, delta=1.0)
        assert score(3, delta=0.5).needs == {Scored(3, rule, 0.5)} != score(3).needs
        assert eval(repr(score(3, delta=0.55))) == score(3, delta=0.55)
        for delta in (0, -1, np.inf, np.nan, True, "1"):
            with pytest.raises(ValueError, match=re.escape(
                    f"{score.__name__} delta must be a number above 0, got {delta!r}")):
                score(3, delta=delta)
    assert ScoreHybrid(3, delta=0.5) != ScoreDH(3, delta=0.5)
    for score in (Score, ScoreAvg, ScoreCluster):
        with pytest.raises(TypeError):
            score(3, delta=0.5)  # only DHONDT and HYBRID have a divisor
    # the cost of a split and the cohesion of ScoreCluster, by keyword; mu = 0 is allowed
    assert ScoreCluster(3) == ScoreCluster(3, mu=0.1, kappa=2) != ScoreCluster(3, mu=0.2)
    assert ScoreCluster(3, kappa=3).needs == {Scored(3, "cluster", kappa=3)} != ScoreCluster(3).needs
    assert ScoreCluster(3, mu=0).needs == {Scored(3, "cluster", mu=0)}
    assert eval(repr(ScoreCluster(3, mu=0.25, kappa=1.5))) == ScoreCluster(3, mu=0.25, kappa=1.5)
    for mu in (-0.1, np.inf, np.nan, True, "1"):
        with pytest.raises(ValueError, match=re.escape(f"ScoreCluster mu must be a number of 0 or more, got {mu!r}")):
            ScoreCluster(3, mu=mu)
    for kappa in (0, -1, np.inf, np.nan, True, "1"):
        with pytest.raises(ValueError, match=re.escape(f"ScoreCluster kappa must be a number above 0, got {kappa!r}")):
            ScoreCluster(3, kappa=kappa)
    for score in (Score, ScoreAvg, ScoreDH, ScoreHybrid):
        with pytest.raises(TypeError):
            score(3, mu=0.2)  # only CLUSTER has mu and kappa
    # the power of Score's part of the way, by keyword; 1 is the default and not printed
    assert Score(3) == Score(3, power=1) == Score(3, power=1.0) != Score(3, power=1.5)
    assert Score(3, power=1.5).needs == {Scored(3, power=1.5)} != Score(3).needs
    assert repr(Score(3, power=1.5)) == "Score(3, power=1.5)" and repr(Score(3, power=1)) == "Score(3)"
    assert eval(repr(Score(3, power=2.25))) == Score(3, power=2.25)
    for power in (0, -1, np.inf, np.nan, True, "1"):
        with pytest.raises(ValueError, match=re.escape(f"Score power must be a number above 0, got {power!r}")):
            Score(3, power=power)
    for score in (ScoreAvg, ScoreDH, ScoreHybrid, ScoreCluster):
        with pytest.raises(TypeError, match=f"{score.__name__} has no power; only Score has"):
            score(3, power=1.5)  # only RANGE has a power
    # as a ballot like any other: mixed with another, and eliminated round by round
    mixed = Mix(ScoreAvg(4), Score(4), share=0.5)
    assert mixed.needs == {Scored(4, "avg"), Scored(4)}
    assert Eliminate(Tally(Score(4)), how="min").needs == {Scored(4)}


def test_ties_go_to_the_lowest_index():
    """All pairwise shares 1/2 and equal first choices: every total ties and no one beats
    anyone. The highest, Nanson's survivor, Schulze's and Minimax's winner, Black's (by
    Borda: there is no Condorcet winner) and the king of the hill, whom no one
    challenges, are the first candidate; Baldwin drops the lowest index each round, so
    the last one wins. Every margin is 0."""
    n = 4
    d = np.full((3, n, n), 0.5)
    d[:, np.arange(n), np.arange(n)] = 0
    voters = Voters(first=np.full((3, n), 1 / n), pairwise=d)
    for name, expected in (("borda", 0), ("nanson", 0), ("baldwin", n - 1), ("condorcet", CYCLE), ("schulze", 0),
                           ("minimax", 0), ("black", 0), ("koth", 0)):
        winner, margin = METHODS[name].evaluate(voters)
        np.testing.assert_array_equal(winner, expected, err_msg=name)
        np.testing.assert_array_equal(margin, 0.0, err_msg=name)


def test_methods_are_the_fifteen_expressions_they_print_as():
    """The fifteen built-in methods are these blocks, and each prints as its expression."""
    margins = Margins(Pairwise())
    koth = Unbeaten(margins, against=Highest(Tally(Plurality())), order=Tally(Plurality()))
    expected = {
        "fptp": Highest(Tally(Plurality())),
        "two_round": Unbeaten(margins, among=Highest(Tally(Plurality()), n=2)),
        "irv": Eliminate(Tally(Plurality()), how="min"),
        "dac": DescendingCoalitions(),
        "borda": Highest(Tally(BordaCount())),
        "baldwin": Eliminate(Tally(BordaCount()), how="min"),
        "nanson": Eliminate(Tally(BordaCount()), how="mean"),
        "schulze": Unbeaten(StrongestPaths(margins)),
        "condorcet": Unbeaten(margins),
        "minimax": Highest(Weakest(margins)),
        "black": Fallback(Unbeaten(margins), Highest(Tally(BordaCount()))),
        "koth": koth,
        "score": Highest(Tally(Score(6, power=1.5))),
        "star": Unbeaten(ScoreComparisons(Score(6, power=1.5)), among=Highest(Tally(Score(6, power=1.5)), n=2)),
        "irv_set_koth_score": Highest(Tally(Score(6, power=1.5)), among=Union(Schedules(Tally(Plurality())), koth)),
    }
    assert {name: METHODS[name] for name in expected} == expected
    assert list(METHODS)[:len(expected)] == list(expected)
    fptp, irv, borda = METHODS["fptp"], METHODS["irv"], METHODS["borda"]
    assert repr(koth) == "Unbeaten(Margins(Pairwise()), against=Highest(Tally(Plurality())), order=Tally(Plurality()))"
    assert repr(METHODS["irv_set_koth_score"]) == (
        f"Highest(Tally(Score(6, power=1.5)), among=Schedules(Tally(Plurality())) | {koth!r})")
    assert repr(Eliminate(Tally(BordaCount()), how="mean", until=2)) == 'Eliminate(Tally(BordaCount()), how="mean", until=2)'
    assert repr(fptp | (irv | borda)) == f"{fptp!r} | ({irv!r} | {borda!r})"
    blocks = [*expected.values(), *BALLOTS, *PAIR_BLOCKS, fptp | irv | borda, fptp | (irv | borda),
              Unbeaten(StrongestPaths(margins), against=METHODS["black"], order=Weakest(margins)),
              Unbeaten(margins, among=fptp | borda), Fallback(Highest(Tally(BY_MEAN), n=3), fptp | irv),
              Eliminate(Tally(BordaCount()), how="min", until=2), Highest(Tally(Plurality()), n=2),
              Schedules(Tally(BordaCount())), Highest(Weakest(margins), among=fptp | irv),
              Eliminate(Tally(Plurality()), how="all", among=Schedules(Tally(Plurality()))),
              Mix(BY_GAP, BY_MEAN, share=0.25)]
    for block in blocks:
        assert eval(repr(block)) == block


def test_blocks_are_frozen_and_hashable():
    """Equal blocks are equal methods and hash alike, and they cannot be changed."""
    assert len(set(METHODS.values())) == len(METHODS)
    cache = {Eliminate(Tally(BordaCount()), how="mean"): "nanson"}
    assert cache[METHODS["nanson"]] == "nanson"
    assert Unbeaten(Margins(Pairwise()), against=None, order=None) == METHODS["condorcet"]
    with pytest.raises(FrozenInstanceError):
        METHODS["irv"].how = "mean"
    with pytest.raises(FrozenInstanceError):
        METHODS["fptp"].until = 2


def test_needs():
    """The shares each block reads: a Tally counted again among finalists the profile,
    other totals as they are; a runoff, Fallback, Union and a king those of all their
    parts."""
    margins = Margins(Pairwise())
    assert METHODS["fptp"].needs == Highest(Tally(Plurality()), n=2).needs == {FIRST}  # no profile
    assert METHODS["irv"].needs == METHODS["dac"].needs == {PROFILE}
    for name in ("borda", "baldwin", "nanson", "schulze", "condorcet", "minimax", "black"):
        assert METHODS[name].needs == {PAIRWISE}, name
    assert METHODS["koth"].needs == METHODS["two_round"].needs == {FIRST, PAIRWISE}
    assert METHODS["score"].needs == {Scored(6, power=1.5)} == {Scored(6, "range", power=1.5)}
    assert METHODS["star"].needs == {Scored(6, power=1.5), ScoredPairwise(Scored(6, power=1.5))}
    assert METHODS["irv_set_koth_score"].needs == {PROFILE, FIRST, PAIRWISE, Scored(6, power=1.5)}
    assert Unbeaten(margins, among=METHODS["koth"] | METHODS["irv"]).needs == {FIRST, PAIRWISE, PROFILE}
    assert Unbeaten(margins, among=METHODS["borda"] | METHODS["minimax"]).needs == {PAIRWISE}
    assert Eliminate(Tally(Plurality()), how="min", until=2).needs == {PROFILE}
    assert Eliminate(Tally(Plurality()), how="mean").needs == {PROFILE}
    assert Eliminate(Weakest(margins), how="all").needs == {PAIRWISE}
    assert (METHODS["fptp"] | METHODS["irv"]).needs == {FIRST, PROFILE}
    assert all(block.needs == {PAIRWISE} for block in PAIR_BLOCKS)
    assert Highest(Tally(Mix(BY_GAP, BY_MEAN, share=0.5))).needs == {Scored(2, "dhondt"), Scored(2, "avg")}
    for score, rule in ((Score, "range"), (ScoreAvg, "avg"), (ScoreDH, "dhondt"), (ScoreHybrid, "hybrid"),
                        (ScoreCluster, "cluster")):
        assert Highest(Tally(score(6))).needs == {Scored(6, rule)}
    assert Eliminate(Tally(BY_GAP), how="min").needs == {Scored(2, "dhondt")}
    assert Highest(Tally(Score(6, power=1.5)), among=Schedules(Tally(Plurality()))).needs == {
        PROFILE, Scored(6, power=1.5)}
    assert Schedules(Tally(Plurality())).needs == {PROFILE}
    assert Schedules(Tally(BordaCount())).needs == {PAIRWISE}
    assert Highest(Tally(Plurality()), among=Highest(Tally(Plurality()), n=2)).needs == {FIRST, PROFILE}
    assert Highest(Weakest(margins), among=METHODS["fptp"]).needs == {PAIRWISE, FIRST}
    assert Fallback(METHODS["fptp"], METHODS["schulze"]).needs == {FIRST, PAIRWISE}
    assert Fallback(METHODS["irv"], METHODS["fptp"]).needs == {PROFILE, FIRST}
    assert Fallback(METHODS["condorcet"], Highest(Tally(BY_MEAN))).needs == {PAIRWISE, Scored(2, "avg")}
    assert Unbeaten(margins, against=METHODS["irv"], order=Tally(Plurality())).needs == {PAIRWISE, PROFILE, FIRST}
    assert Unbeaten(margins, against=METHODS["borda"], order=Weakest(margins)).needs == {PAIRWISE}


def test_score_comparisons_can_weight_ordinal_preferences_on_ties():
    ballot = Score(6)
    comparisons = ScoreComparisons(ballot, tie_weight=0.35)
    assert comparisons.needs == {ScoredPairwise(ballot.scored, tie_weight=0.35)}
    assert repr(comparisons) == "ScoreComparisons(Score(6), tie_weight=0.35)"
    assert eval(repr(comparisons)) == comparisons
    assert ScoreComparisons(ballot).needs == {ScoredPairwise(ballot.scored)}
    for weight in (-0.1, 1.1, True, float("nan"), "0.5"):
        with pytest.raises(ValueError, match="Score comparison tie weight"):
            ScoreComparisons(ballot, tie_weight=weight)


MARGINS = Margins(Pairwise())


@pytest.mark.parametrize("build, error, message", [
    # a block of the wrong type, in each way the message names it
    (lambda: Highest(BordaCount()), TypeError, "Highest expects CandidateTotals, got a Ballot: BordaCount()"),
    (lambda: Highest(METHODS["schulze"]), TypeError,
     "Highest expects CandidateTotals, got a Winner: Unbeaten(StrongestPaths(Margins(Pairwise())))"),
    (lambda: Highest(MARGINS), TypeError, "Highest expects CandidateTotals, got PairDiffs: Margins(Pairwise())"),
    (lambda: Highest(Pairwise()), TypeError, "Highest expects CandidateTotals, got PairShares: Pairwise()"),
    (lambda: Tally(Plurality), TypeError, "Tally expects a Ballot, got the class Plurality; call it: Plurality()"),
    (lambda: Tally(3), TypeError, "Tally expects a Ballot, got int 3"),
    # and in each block
    (lambda: Mix(Tally(Plurality()), Score(2), share=0.5), TypeError,
     "Mix expects a Ballot, got CandidateTotals: Tally(Plurality())"),
    (lambda: Eliminate(Highest(Tally(Plurality())), how="min"), TypeError,
     "Eliminate expects a Tally (it tallies the remaining candidates again), got a Winner"),
    (lambda: Eliminate(BordaCount(), how="all"), TypeError, "Eliminate expects CandidateTotals, got a Ballot"),
    (lambda: Margins(MARGINS), TypeError, "Margins expects PairShares, got PairDiffs: Margins(Pairwise())"),
    (lambda: StrongestPaths(Pairwise()), TypeError, "StrongestPaths expects PairDiffs, got PairShares: Pairwise()"),
    (lambda: Weakest(Pairwise()), TypeError, "Weakest expects PairDiffs, got PairShares: Pairwise()"),
    (lambda: Unbeaten(Tally(BordaCount())), TypeError,
     "Unbeaten expects PairDiffs, got CandidateTotals: Tally(BordaCount())"),
    (lambda: Unbeaten(MARGINS, against=Tally(Plurality()), order=Tally(Plurality())), TypeError,
     "Unbeaten expects a Winner as against, got CandidateTotals: Tally(Plurality())"),
    (lambda: Unbeaten(MARGINS, against=METHODS["fptp"], order=Plurality()), TypeError,
     "Unbeaten expects CandidateTotals as order, got a Ballot: Plurality()"),
    (lambda: Unbeaten(MARGINS, among=Tally(Plurality())), TypeError,
     "Unbeaten expects a Winner as among, got CandidateTotals: Tally(Plurality())"),
    (lambda: Fallback(BordaCount(), METHODS["borda"]), TypeError,
     "Fallback expects a Winner, got a Ballot: BordaCount()"),
    (lambda: METHODS["fptp"] | Tally(Plurality()), TypeError,
     "Union expects a Winner, got CandidateTotals: Tally(Plurality())"),
    (lambda: Highest(Tally(Plurality()), among=Tally(Plurality())), TypeError,
     "Highest expects a Winner as among, got CandidateTotals: Tally(Plurality())"),
    (lambda: Schedules(BordaCount()), TypeError,
     "Schedules expects a Tally (it tallies the remaining candidates again), got a Ballot: BordaCount()"),
    # arguments that do not go together, or out of their range
    (lambda: Unbeaten(MARGINS, against=METHODS["fptp"]), ValueError,
     "Unbeaten takes against and order together, or neither"),
    (lambda: Unbeaten(MARGINS, order=Tally(Plurality())), ValueError,
     "Unbeaten takes against and order together, or neither"),
    (lambda: Unbeaten(MARGINS, against=METHODS["fptp"], order=Tally(Plurality()), among=METHODS["irv"]),
     ValueError, "Unbeaten takes among or against, not both"),
    (lambda: Unbeaten(MARGINS, against=Highest(Tally(Plurality()), n=2), order=Tally(Plurality())), ValueError,
     "Unbeaten against must choose one candidate, the king"),
    (lambda: Eliminate(Tally(Plurality()), how="min", among=METHODS["irv"]), ValueError,
     'Eliminate takes among only with how="all": Highest(totals, among=...)'),
    (lambda: Eliminate(Tally(BordaCount()), how="max"), ValueError,
     'Eliminate how must be "min", "mean" or "all", got \'max\''),
    (lambda: Eliminate(Tally(BordaCount()), how="min", until=0), ValueError,
     "Eliminate until must be a whole number of at least 1, got 0"),
    (lambda: Highest(Tally(BordaCount()), n=1.5), ValueError,
     "Highest n must be a whole number of at least 1, got 1.5"),
    (lambda: Highest(Tally(BordaCount()), n=True), ValueError,
     "Highest n must be a whole number of at least 1, got True"),
    (lambda: Mix(BY_GAP, BY_MEAN, share=-0.1), ValueError, "Mix share must be between 0 and 1, got -0.1"),
    (lambda: Mix(BY_GAP, BY_MEAN, share=float("nan")), ValueError, "Mix share must be between 0 and 1, got nan"),
])
def test_wrong_blocks_and_arguments_fail_when_built(build, error, message):
    with pytest.raises(error, match=re.escape(message)):
        build()
