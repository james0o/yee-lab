"""Checks of the strategic voters (yeelab.build.strategic).

Every strategy at share 0 must be the method itself, bit for bit. Bullet voters must be
counted as the weak orders they mark, their first choice first and all others equal
below it, where equal candidates share the mean points of their positions and acquiesce
to every set that holds the first choice; where all voters bullet vote, every method
elects the candidate with the most first choices. Burial and compromise are checked
against the method itself on the profile where every ranking comes twice, as it is and
moved (the frontrunner it ranks lower last, or the one it ranks higher first unless one
of them is first already), on the points of each pair of frontrunners, with the score
shares and comparisons of a stand-in for the ballots of the voter grid (_Grid, which
test_score.py checks against sampled voters); their margin is at most the poll's.
Burial keeps FPTP and the two-round system, compromise the two-round system, and where
all voters compromise FPTP and IRV elect its winner too. Bullet(Burial(Compromise(method,
y), q), b), where b of the voters bullet vote and of the others q bury and y compromise,
each whatever the other does, is checked against the references of references.py on the
weak orders of all kinds of voters, and STAR's comparisons moved once for all of them.
"""

import re
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from yeelab.build import (
    FIRST,
    METHODS,
    PAIRWISE,
    PROFILE,
    Ballots,
    BordaCount,
    D21,
    Bullet,
    Burial,
    Compromise,
    Eliminate,
    Graded,
    Highest,
    Plurality,
    Score,
    ScoreComparisons,  # for eval(repr(...))
    Scored,
    ScoredPairwise,
    Striped,
    Tally,
    Unbeaten,  # for eval(repr(...))
    Voters,
)
from yeelab.build.strategic import POLL, bulleted, burying, compromising
from yeelab.build.voters import STRIPS
from yeelab.margin import regions
from yeelab.margin.shares import Model, strip_masks
from yeelab.voting import irv_rounds

from references import (TESTED_METHODS, TOLERANCE, dac_reference, elimination_reference, levels_of,
                        random_profiles, voters_of)

# the shares of the voters who bullet vote, bury and compromise, where any shares will do
BULLET, BURIAL, COMPROMISE = 0.3, 0.4, 0.6
SCORE_BALLOT = Score(6, power=1.5)  # the score ballot of the methods
STRATEGIES = (Bullet, Burial, Compromise)
# the methods checked against the references on weak orders
REFERENCED = ["fptp", "irv", "baldwin", "nanson", "borda", "dac", "schulze", "minimax", "koth"]


def _tied_weights(ballot, levels, alive):
    """W[b, c] = points the weak order b gives c, with L[b, c] the level of c, lower
    better and equal for equal candidates: among the candidates in alive (C,), equal
    candidates share the mean of weight(k, number of them) over the positions k they
    take; 0 for the others. On a ranking (levels_of) it is ranking_weights."""
    n = int(alive.sum())
    live = levels[:, alive]
    above = (live[:, None, :] < live[:, :, None]).sum(axis=-1)  # [b, c]: ranked above c
    equal = (live[:, None, :] == live[:, :, None]).sum(axis=-1)  # [b, c]: with c itself
    points = np.concatenate([[0.0], np.cumsum([ballot.weight(k, n) for k in range(n)])])
    weights = np.zeros(levels.shape)
    weights[:, alive] = (points[above + equal] - points[above]) / equal
    return weights


def _bullet_ballots(rankings, probs, share):
    """The weak orders (R + C, C) of the voters of whom `share` bullet vote, and their
    shares (..., R + C): the rankings with the rest of the voters, then for every
    candidate c a ballot with c first and all others equal below it, with `share` of the
    voters whose first choice c is."""
    n = rankings.shape[1]
    probs = probs.astype(np.float64)
    first = probs @ np.eye(n)[rankings[:, 0]]
    levels = np.concatenate([levels_of(rankings), 1 - np.eye(n, dtype=np.int64)])
    return levels, np.concatenate([(1 - share) * probs, share * first], axis=-1)


def _tied_pairwise(levels, shares):
    """d[..., c, e] = share of the weak orders ranking c above e, and half of those
    ranking them equal; d[..., c, c] = 0."""
    above = levels[:, :, None] < levels[:, None, :]
    equal = (levels[:, :, None] == levels[:, None, :]) & ~np.eye(levels.shape[1], dtype=bool)
    d = shares @ (above + equal / 2).reshape(len(levels), -1)
    return d.reshape(*shares.shape[:-1], levels.shape[1], levels.shape[1])


def _every_share(rankings, probs, seed=0):
    """Every share a method may need: those of a complete profile (voters_of), and random
    score shares of the built-in score ballots."""
    voters = voters_of(rankings, probs)
    rng = np.random.default_rng(seed)
    n = rankings.shape[1]
    scored = (SCORE_BALLOT.scored, Score(3, power=1.5, fixed=True).scored, D21(power=1.5).scored)
    unscored = {ballot: rng.random((*voters.shape, n)) for ballot in scored}
    comparisons = {}
    for ballot in scored:
        above = rng.random((*voters.shape, n, n))
        comparisons[ScoredPairwise(ballot)] = above - np.swapaxes(above, -1, -2)
    graded = {Graded(SCORE_BALLOT.scored): rng.dirichlet(np.ones(SCORE_BALLOT.levels), size=(*voters.shape, n))}
    return replace(voters, unscored=unscored, scored_pairwise=comparisons, graded=graded)


class _Grid:
    """A stand-in for margin.shares.ScoreGrid on random shares at every point (..., C, C):
    what burying takes back from c where e is ranked above it, what compromising adds to c
    over e, and how much the score comparisons of each kind of strategic voter (burying,
    compromising, both) move, mixed by their shares; with the shares of every score
    `graded` (..., C, L), the part of them that burying and compromising move, of the
    voters who rank e above c and of those who rank c above e, neither first (..., C, C,
    L), which add up to at most `graded`. It notes the frontrunners and shares every
    comparison is moved by."""

    def __init__(self, unscored, seed, graded=None):
        rng = np.random.default_rng(seed)
        n = unscored.shape[-1]
        self.taken_all, self.added_all = rng.random((2, *unscored.shape, n)) * (1 - np.eye(n))
        self.added_all *= unscored[..., None]  # at most what is not given
        kinds = rng.normal(size=(3, *unscored.shape, n)) / 10
        self.kinds = kinds - np.swapaxes(kinds, -1, -2)
        self.calls = []
        if graded is not None:
            buried = rng.random((*graded.shape[:-1], n, graded.shape[-1]))
            raised = (1 - buried) * rng.random(buried.shape)
            self.buried_all, self.raised_all = (part * graded[..., :, None, :] for part in (buried, raised))

    def moved_grades(self, a, b, cells):
        """(2, P, 2, L): those of a and b against each other, buried and raised."""
        return np.stack([np.stack([part[cells[:, 0], cells[:, 1], a, b], part[cells[:, 0], cells[:, 1], b, a]], 1)
                         for part in (self.buried_all, self.raised_all)])

    def moved_all(self, a, b, bury, compromise):
        """moved() at every point, (..., C, C)."""
        moved = np.tensordot([bury * (1 - compromise), (1 - bury) * compromise, bury * compromise], self.kinds, 1)
        n = moved.shape[-1]
        pairs = np.zeros((n, n), dtype=bool)
        pairs[[a, b], :] = pairs[:, [a, b]] = True
        return np.where(pairs & ~np.eye(n, dtype=bool), moved, 0.0)

    def taken(self, a, b, cells):
        values = self.taken_all[cells[:, 0], cells[:, 1]]
        return np.stack([values[:, a, b], values[:, b, a]], axis=-1)

    def added(self, a, b, cells):
        values = self.added_all[cells[:, 0], cells[:, 1]]
        return np.stack([values[:, a, b], values[:, b, a]], axis=-1)

    def moved(self, a, b, bury, compromise, cells):
        self.calls.append((int(a), int(b), bury, compromise))
        return self.moved_all(a, b, bury, compromise)[cells[:, 0], cells[:, 1]]


def _strategy_shares(rankings, probs, seed=0):
    """_every_share, and the ballots of the score ballot as a _Grid: what burying takes
    back, compromising adds and the strategies move the comparisons by."""
    voters = _every_share(rankings, probs, seed)
    shape = voters.shape
    cells = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), axis=-1)
    graded = {key.scored: values for key, values in voters.graded.items()}
    grids = {Ballots(scored): _Grid(values, seed + i + 1, graded.get(scored))
             for i, (scored, values) in enumerate(voters.unscored.items())}
    return replace(voters, ballots=grids, cells=cells)


def _grid_of(voters):
    return voters.ballots[Ballots(SCORE_BALLOT.scored)]


def _buried_profile(rankings, probs, share, a, b):
    """Every ranking twice: as it is with 1 - share of its voters, and with share of them
    with the one of a and b it ranks lower moved last."""
    buried = []
    for ranking in rankings.tolist():
        lower = a if ranking.index(a) > ranking.index(b) else b
        buried.append([c for c in ranking if c != lower] + [lower])
    return (np.concatenate([rankings, np.array(buried, dtype=rankings.dtype)]),
            np.concatenate([(1 - share) * probs, share * probs], axis=-1))


def _compromised_profile(rankings, probs, share, a, b):
    """Every ranking twice: as it is with 1 - share of its voters, and with share of them
    with the one of a and b it ranks higher moved first, unless one of them is first."""
    raised = []
    for ranking in rankings.tolist():
        upper = a if ranking.index(a) < ranking.index(b) else b
        raised.append(ranking if ranking[0] in (a, b) else [upper] + [c for c in ranking if c != upper])
    return (np.concatenate([rankings, np.array(raised, dtype=rankings.dtype)]),
            np.concatenate([(1 - share) * probs, share * probs], axis=-1))


def _pairs(voters):
    """The frontrunners of every point as a bit mask, and the poll's margin."""
    chosen, gap = POLL.select(voters)
    return chosen @ (1 << np.arange(voters.n_candidates)), gap


# ---------------------------------------------------------------- Every strategy


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.__name__)
def test_no_strategic_voters_is_the_method_itself(profile, strategy):
    """Bit for bit, from the same shares, and with the same needs."""
    voters = _strategy_shares(*profile)
    for name, method in TESTED_METHODS.items():
        assert strategy(method, 0).needs == method.needs, name
        for result, expected in zip(strategy(method, 0).evaluate(voters), method.evaluate(voters)):
            np.testing.assert_array_equal(result, expected, err_msg=name)


def test_strategies_are_blocks():
    """They print as the expressions that build them, take a Winner and a share, and need
    what their voters' shares come from: the first choices for bullet voters' pairwise and
    score shares, and for burial and compromise the poll's, the profile for the moved
    first-choice and pairwise shares and the ballots of every score ballot."""
    fptp, irv, borda = METHODS["fptp"], METHODS["irv"], METHODS["borda"]
    built = [Bullet(irv, 0.3), Bullet(fptp | irv, share=1), Bullet(Bullet(METHODS["star"], 0.5), 0.25),
             Burial(borda, 0.5), Burial(Bullet(irv, 0.2), 1, poll=Highest(Tally(BordaCount()), n=2)),
             Compromise(fptp, 0.5), Bullet(Burial(Compromise(irv, 0.3), 0.2), 0.1)]
    assert repr(built[0]) == 'Bullet(Eliminate(Tally(Plurality()), how="min"), share=0.3)'
    assert repr(built[3]) == "Burial(Highest(Tally(BordaCount())), share=0.5)"
    assert repr(built[5]) == "Compromise(Highest(Tally(Plurality())), share=0.5)"
    for block in built:
        assert eval(repr(block)) == block
    for strategy in STRATEGIES:
        name = strategy.__name__
        with pytest.raises(TypeError, match=re.escape(f"{name} expects a Winner, got CandidateTotals: "
                                                      "Tally(Plurality())")):
            strategy(Tally(Plurality()), share=0.5)
        for share in (-0.1, 1.5, float("nan")):
            with pytest.raises(ValueError, match=re.escape(f"{name} share must be between 0 and 1, got {share!r}")):
                strategy(irv, share=share)
        assert strategy(fptp | irv, 0.5).seats is None and strategy(Highest(Tally(Plurality()), n=2), 0.5).seats == 2
    scored = Scored(6, power=1.5)
    star = {scored, ScoredPairwise(scored)}
    for name, bullet, polled in (("fptp", {FIRST}, {FIRST}), ("irv", {PROFILE}, {PROFILE, FIRST}),
                                 ("borda", {PAIRWISE, FIRST}, {PAIRWISE, PROFILE, FIRST}),
                                 ("score", {scored, FIRST}, {scored, Ballots(scored), FIRST}),
                                 ("star", star | {FIRST}, star | {Ballots(scored), FIRST})):
        assert Bullet(METHODS[name], 0.5).needs == bullet, name
        assert Burial(METHODS[name], 0.5).needs == polled - ({PROFILE} if name == "fptp" else set()), name
        assert Compromise(METHODS[name], 0.5).needs == polled | ({PROFILE} if name == "fptp" else set()), name


def test_strategic_voters_move_the_shares_of_every_score():
    """A median reads the shares of every score, which strategic voters move as the
    mean scores: bullet voters give their first choice the top score and the others 0;
    of the voters a strategy changes, as many as its share go from their score of a
    frontrunner to 0 (burying the one they rank lower) or the top score (compromising on
    the one they rank higher, neither first), the bullet voters left out. The shares
    still add up to the same, and the other candidates keep theirs."""
    rankings, probs = random_profiles(3, 6)
    voters = _strategy_shares(rankings, probs.reshape(2, 3, -1))
    key, a, b = Graded(SCORE_BALLOT.scored), 2, 0
    g, f = voters.graded[key], voters.first
    others = f.sum(axis=-1, keepdims=True) - f
    bullet = (1 - BULLET) * g
    bullet[..., 0] += BULLET * others
    bullet[..., -1] += BULLET * f
    bulleted_ = bulleted(voters, BULLET)
    np.testing.assert_allclose(bulleted_.graded[key], bullet, rtol=0, atol=1e-15)
    grid = _grid_of(voters)
    for vote, share, part, target in ((burying, BURIAL, grid.buried_all, 0), (compromising, COMPROMISE, grid.raised_all, -1)):
        moved = vote(bulleted_, share, a, b).graded[key]
        expected = bullet.copy()
        for c, e in ((a, b), (b, a)):
            change = share * (1 - BULLET) * part[..., c, e, :]
            expected[..., c, :] -= change
            expected[..., c, target] += change.sum(axis=-1)
        np.testing.assert_allclose(moved, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(moved.sum(axis=-1), g.sum(axis=-1), rtol=0, atol=1e-12)
        assert (moved >= 0).all()
    assert bulleted(voters, 0) is voters and burying(voters, 0, a, b) is voters
    for strategy in STRATEGIES:
        assert strategy(METHODS["mj"], 0.3).needs >= METHODS["mj"].needs | {FIRST}
    assert Burial(METHODS["mj"], 0.3).needs == {key, Ballots(SCORE_BALLOT.scored), FIRST}


def test_the_poll_names_two_frontrunners():
    with pytest.raises(ValueError, match=re.escape("Burial poll must choose two candidates, the frontrunners")):
        Burial(METHODS["borda"], 0.5, poll=METHODS["fptp"])
    with pytest.raises(TypeError, match=re.escape("Burial expects a Winner as poll, got CandidateTotals")):
        Burial(METHODS["borda"], 0.5, poll=Tally(Plurality()))
    assert Burial(METHODS["borda"], 0.5, poll=Highest(Tally(BordaCount()), n=2)).needs == {PAIRWISE, PROFILE}


@pytest.mark.parametrize("strategy", [Burial, Compromise])
def test_the_poll_shares_move_too(strategy):
    """The strategic voters move all the shares of the voters, also those only the poll
    reads, so the strategy needs what moving them takes: the profile for a Borda poll's
    pairwise shares and the ballots for a score poll's. The method is evaluated alike on
    voters with those shares and on voters with every share a method could read."""
    scored = Scored(6, power=1.5)
    for poll, extra in ((Highest(Tally(BordaCount()), n=2), {PAIRWISE, PROFILE}),
                        (Highest(Tally(SCORE_BALLOT), n=2), {scored, Ballots(scored)})):
        method = strategy(METHODS["fptp"], 0.3, poll=poll)
        assert method.needs >= extra
        alone = regions.winners(method, MODEL_CANDIDATES, MODELS[0], 12)[1]
        every = method.needs | METHODS["star"].needs | {PAIRWISE, PROFILE, FIRST, Ballots(scored)}
        voters = regions.voters(every, MODEL_CANDIDATES, MODELS[0], 12)
        np.testing.assert_array_equal(method.evaluate(voters)[0], alone)

# ---------------------------------------------------------------- Bullet voting


def test_the_shares_of_bullet_voters(profile):
    """The first choices stay; a pair of equal candidates counts half for each, so that
    d[c, e] + d[e, c] = 1; the profile keeps the rest of the voters and the bullet
    ballots the others of each first choice; a bullet voter gives the top score to their
    first choice and 0 to the others, and of a pair scores only their first choice
    higher."""
    rankings, probs = profile
    voters = _every_share(rankings, probs)
    moved = bulleted(voters, BULLET)
    levels, shares = _bullet_ballots(rankings, probs, BULLET)
    n = rankings.shape[1]
    np.testing.assert_array_equal(moved.first, voters.first)
    np.testing.assert_allclose(moved.pairwise, _tied_pairwise(levels, shares), rtol=0, atol=1e-6)
    pairs = moved.pairwise + np.swapaxes(moved.pairwise, -1, -2)
    np.testing.assert_allclose(pairs, np.broadcast_to(1 - np.eye(n), pairs.shape), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(moved.rankings, rankings)
    np.testing.assert_allclose(moved.probs, (1 - BULLET) * probs, rtol=1e-6, atol=0)
    np.testing.assert_allclose(moved.bullet, BULLET * voters.first, rtol=0, atol=1e-6)
    f = voters.first
    unscored = (1 - BULLET) * voters.unscored[SCORE_BALLOT.scored] + BULLET * (f.sum(axis=-1, keepdims=True) - f)
    np.testing.assert_allclose(moved.unscored[SCORE_BALLOT.scored], unscored, rtol=0, atol=1e-12)
    key = ScoredPairwise(SCORE_BALLOT.scored)
    lead = f[..., :, None] - f[..., None, :]
    np.testing.assert_allclose(moved.scored_pairwise[key], (1 - BULLET) * voters.scored_pairwise[key]
                               + BULLET * lead, rtol=0, atol=1e-12)


def test_bullet_voters_keep_their_equal_rankings_when_scores_tie():
    key = ScoredPairwise(SCORE_BALLOT.scored, tie_weight=0.5)
    first = np.array([[0.4, 0.35, 0.25]])
    strict = np.array([[[0, 0.2, 0.3], [-0.2, 0, 0.4], [-0.3, -0.4, 0]]])
    voters = Voters(first=first, scored_pairwise={key: strict})
    share = 0.3

    moved = bulleted(voters, share)
    lead = first[..., :, None] - first[..., None, :]
    expected = (1 - share) * strict + share * lead
    np.testing.assert_allclose(moved.scored_pairwise[key], expected, rtol=0, atol=1e-12)
    method = Unbeaten(
        ScoreComparisons(SCORE_BALLOT, tie_weight=key.tie_weight),
        among=Highest(Tally(SCORE_BALLOT), n=2),
    )
    assert Bullet(method, share).needs >= {FIRST, key}


def test_bullet_ballots_are_tallied_with_equal_candidates_sharing(profile):
    """Among any remaining candidates: equal candidates share the mean points of the
    positions they take, so a bullet ballot whose first choice is out counts equally for
    all the remaining ones, and every ballot still hands out all its points."""
    rankings, probs = profile
    voters, n = bulleted(voters_of(rankings, probs), BULLET), rankings.shape[1]
    levels, shares = _bullet_ballots(rankings, probs, BULLET)
    rng = np.random.default_rng(n)
    alive = rng.random(probs.shape[:-1] + (n,)) < 0.5
    alive |= np.arange(n) == rng.integers(n, size=probs.shape[:-1] + (1,))
    codes = alive @ (1 << np.arange(n))
    for ballot in (Plurality(), BordaCount()):
        expected = np.empty(alive.shape)
        for code in np.unique(codes):
            at = codes == code
            expected[at] = shares[at] @ _tied_weights(ballot, levels, ((code >> np.arange(n)) & 1).astype(bool))
        totals = ballot.tally(voters, alive)
        np.testing.assert_allclose(totals, expected, rtol=0, atol=1e-5)
        if isinstance(ballot, Plurality):
            np.testing.assert_allclose(totals.sum(axis=-1), 1, rtol=0, atol=1e-5)


def test_where_all_voters_bullet_vote_the_most_first_choices_win(profile):
    """Every bullet ballot ranks its first choice above all others and the others equal,
    so the candidate with the most first choices beats every other head to head, has
    the most points and the highest mean score, and outlasts every elimination."""
    voters = _every_share(*profile)
    fptp, lead = METHODS["fptp"].evaluate(voters)
    clear = lead > TOLERANCE
    assert clear.mean() > 0.9  # not vacuous
    for name, method in TESTED_METHODS.items():
        np.testing.assert_array_equal(Bullet(method, 1).evaluate(voters)[0][clear], fptp[clear], err_msg=name)


def test_bullet_voters_of_bullet_voters(profile):
    """b of the voters bullet vote, and a of the rest: 1 - (1 - a)(1 - b) of them do."""
    voters = _every_share(*profile)
    for name in ("irv", "dac", "borda", "schulze", "star"):
        winner, margin = Bullet(METHODS[name], 0.6).evaluate(voters)
        nested, nested_margin = Bullet(Bullet(METHODS[name], 0.2), 0.5).evaluate(voters)
        clear = margin > TOLERANCE
        np.testing.assert_array_equal(nested[clear], winner[clear])
        np.testing.assert_allclose(nested_margin, margin, rtol=0, atol=1e-6)


def test_a_lead_among_candidates_nearly_all_give_the_top_score_is_kept_with_few_bullet_voters():
    """3e-30 and 1e-30 of the sincere voters do not give two candidates the top score,
    and 1e-31 of the voters bullet vote, for each in proportion to its first choices."""
    unscored = np.array([[3e-30, 1e-30, 0.4]])
    voters = Voters(first=np.array([[0.6, 0.4, 0.0]]), unscored={SCORE_BALLOT.scored: unscored})
    winner, margin = Bullet(METHODS["score"], 1e-31).evaluate(voters)
    np.testing.assert_array_equal(winner, [1])
    np.testing.assert_allclose(margin, [(3e-30 + 0.4e-31) - (1e-30 + 0.6e-31)], rtol=1e-12, atol=0)

# ---------------------------------------------------------------- Burial and compromise


@pytest.mark.parametrize("strategy", [Burial, Compromise], ids=lambda s: s.__name__)
def test_burial_and_compromise_are_the_method_on_moved_rankings(profile, strategy):
    """On the points of each pair of frontrunners a and b: the method on the profile of
    every ranking as it is and moved, from which voters_of derives the first-choice and
    pairwise shares; the part of the top score not given to a and b raised by what
    burying voters gave them, or lowered by what compromising ones add; and the score
    comparisons moved by them (_Grid). The margin is the smaller of that one and the
    poll's."""
    rankings, probs = profile
    voters, n = _strategy_shares(rankings, probs), rankings.shape[1]
    share, moved_profile = (BURIAL, _buried_profile) if strategy is Burial else (COMPROMISE, _compromised_profile)
    pairs, gap = _pairs(voters)
    for name, method in TESTED_METHODS.items():
        winner, margin = strategy(method, share).evaluate(voters)
        expected, expected_margin = np.empty_like(winner), np.empty_like(margin)
        scored = next((need for need in method.needs if isinstance(need, Scored)), SCORE_BALLOT.scored)
        comparisons = next((need for need in method.needs if isinstance(need, ScoredPairwise)),
                           ScoredPairwise(scored))
        key = comparisons.scored
        grid = voters.grid(key)
        for pair in np.unique(pairs):
            at = pairs == pair
            a, b = np.flatnonzero((pair >> np.arange(n)) & 1)
            unscored = voters.unscored[key][at].copy()
            points, sign = (grid.taken_all, 1) if strategy is Burial else (grid.added_all, -1)
            unscored[:, a] += sign * share * points[at][:, a, b]
            unscored[:, b] += sign * share * points[at][:, b, a]
            bury, compromise = (share, 0) if strategy is Burial else (0, share)
            compared = voters.scored_pairwise[comparisons][at] + grid.moved_all(a, b, bury, compromise)[at]
            # the moved voters' scores of a and b to 0 or the top, of the ballot of the median
            graded = voters.graded[Graded(SCORE_BALLOT.scored)][at].copy()
            part, target = (_grid_of(voters).buried_all, 0) if strategy is Burial else (_grid_of(voters).raised_all, -1)
            for c, e in ((a, b), (b, a)):
                change = share * part[at][:, c, e]
                graded[:, c] -= change
                graded[:, c, target] += change.sum(axis=-1)
            moved = replace(voters_of(*moved_profile(rankings, probs[at], share, a, b)), unscored={key: unscored},
                            scored_pairwise={comparisons: compared}, graded={Graded(SCORE_BALLOT.scored): graded})
            expected[at], expected_margin[at] = method.evaluate(moved)
        expected_margin = np.minimum(expected_margin, gap)
        clear = expected_margin > TOLERANCE
        assert clear.mean() > 0.3, name  # not vacuous; compromise leaves many near-empty tallies, which tie
        np.testing.assert_array_equal(winner[clear], expected[clear], err_msg=name)
        np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5, err_msg=name)


def test_the_shares_of_burying_voters(profile):
    """No first choice changes, nor the head-to-head of the frontrunners; the pairwise
    shares are those of the buried rankings, and only the pairs of a frontrunner with a
    third candidate change."""
    rankings, probs = profile
    voters, n = voters_of(rankings, probs), rankings.shape[1]
    for a, b in ((0, 1), (n - 1, 0)):
        moved = burying(voters, BURIAL, a, b)
        expected = voters_of(*_buried_profile(rankings, probs, BURIAL, a, b))
        np.testing.assert_array_equal(moved.first, voters.first)
        np.testing.assert_allclose(expected.first, voters.first, rtol=0, atol=1e-6)
        np.testing.assert_allclose(moved.pairwise, expected.pairwise, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(moved.pairwise[..., a, b], voters.pairwise[..., a, b])
        others = np.setdiff1d(np.arange(n), [a, b])
        np.testing.assert_array_equal(moved.pairwise[..., others[:, None], others],
                                      voters.pairwise[..., others[:, None], others])
        np.testing.assert_array_equal(moved.rankings, expected.rankings)
        np.testing.assert_allclose(moved.probs, expected.probs, rtol=1e-6, atol=0)


def test_the_shares_of_compromising_voters(profile):
    """The first choices of the voters with neither frontrunner first go, for that share,
    to the one they rank higher: s (d[a, b] - f[a]) of them to a. The head-to-head of the
    frontrunners stays, and only the pairs of a frontrunner with a third candidate change."""
    rankings, probs = profile
    voters, n = voters_of(rankings, probs), rankings.shape[1]
    f, d = voters.first, voters.pairwise
    for a, b in ((0, 1), (n - 1, 0)):
        moved = compromising(voters, COMPROMISE, a, b)
        expected = voters_of(*_compromised_profile(rankings, probs, COMPROMISE, a, b))
        np.testing.assert_allclose(moved.first, expected.first, rtol=0, atol=1e-6)
        np.testing.assert_allclose(moved.first.sum(axis=-1), 1, rtol=0, atol=1e-5)
        np.testing.assert_allclose(moved.first[..., a], f[..., a] + COMPROMISE * (d[..., a, b] - f[..., a]),
                                   rtol=0, atol=1e-6)
        others = np.setdiff1d(np.arange(n), [a, b])
        np.testing.assert_allclose(moved.first[..., others], (1 - COMPROMISE) * f[..., others], rtol=0, atol=1e-6)
        np.testing.assert_allclose(moved.pairwise, expected.pairwise, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(moved.pairwise[..., a, b], d[..., a, b])
        np.testing.assert_array_equal(moved.pairwise[..., others[:, None], others], d[..., others[:, None], others])
    with pytest.raises(ValueError, match="compromise on pairwise shares needs the profile"):
        compromising(Voters(first=f, pairwise=d), 0.5, 0, 1)
    assert compromising(Voters(first=f), 0.5, 0, 1).first is None  # only the profile tells who moves


def test_burial_keeps_fptp_and_the_two_round_system(profile):
    """Burial changes no first choice, and the frontrunners are the two-round system's
    finalists, whose head-to-head it keeps: the same winners and, for the two-round
    system, whose margin already holds the poll's, the same margins."""
    voters = voters_of(*profile)
    for name in ("fptp", "two_round"):
        winner, margin = METHODS[name].evaluate(voters)
        buried, buried_margin = Burial(METHODS[name], BURIAL).evaluate(voters)
        np.testing.assert_array_equal(buried, winner)
        expected = margin if name == "two_round" else np.minimum(margin, _pairs(voters)[1])
        np.testing.assert_array_equal(buried_margin, expected)


def test_compromise_keeps_the_two_round_system_which_fptp_and_irv_become(profile):
    """The frontrunners gain first choices and the others lose them, so they are still the
    two finalists, and their head-to-head stays. Where all voters compromise every first
    choice is a frontrunner, so FPTP elects the one more voters rank above the other, and
    so does IRV after the empty tallies of all other candidates."""
    voters = voters_of(*profile)
    two_round, two_round_margin = METHODS["two_round"].evaluate(voters)
    clear = two_round_margin > TOLERANCE
    for share in (COMPROMISE, 1):
        compromised = Compromise(METHODS["two_round"], share).evaluate(voters)[0]
        np.testing.assert_array_equal(compromised[clear], two_round[clear])
    for name in ("fptp", "irv"):
        winner, margin = Compromise(METHODS[name], 1).evaluate(voters)
        both = clear & (margin > TOLERANCE)
        assert both.mean() > 0.5, name  # not vacuous: the empty tallies decide nothing
        np.testing.assert_array_equal(winner[both], two_round[both], err_msg=name)


def test_two_empty_tallies_decide_nothing():
    """Where no voter has candidates 2 and 3 first, they go first, in either order, and
    take no ballots with them: the margin of IRV is the smaller of the lead of the other
    two and the smaller of their tallies, which the last of 2 and 3 to go falls short of."""
    rankings = np.array(list(permutations(range(4))))
    probs = np.random.default_rng(4).dirichlet(np.full(len(rankings), 0.5), size=50)
    probs[:, rankings[:, 0] >= 2] = 0
    probs /= probs.sum(axis=-1, keepdims=True)
    first = probs @ np.eye(4)[rankings[:, 0]]
    winner, margin = irv_rounds(rankings, probs)
    np.testing.assert_array_equal(winner, first.argmax(axis=-1))
    expected = np.minimum(first[:, :2].min(axis=-1), np.abs(first[:, 0] - first[:, 1]))
    np.testing.assert_allclose(margin, expected, rtol=0, atol=1e-12)


def test_strategic_voters_need_their_shares():
    """Bullet voters need the first choices, burying voters the profile for pairwise
    shares, and both burying and compromising ones the ballots of the voter grid for
    score shares. Where none moved yet, the score comparisons are as they are."""
    rankings, probs = random_profiles(3, 6)
    voters = voters_of(rankings, probs.reshape(2, 3, -1))
    with pytest.raises(ValueError, match="needs the first-choice shares"):
        bulleted(Voters(pairwise=voters.pairwise), 0.5)
    np.testing.assert_array_equal(bulleted(Voters(pairwise=voters.pairwise), 0).pairwise, voters.pairwise)
    with pytest.raises(ValueError, match="burial on pairwise shares needs the profile"):
        burying(Voters(first=voters.first, pairwise=voters.pairwise), 0.5, 0, 1)
    scored = _every_share(rankings, probs.reshape(2, 3, -1))
    with pytest.raises(ValueError, match="the score shares of strategic voters need the ballots"):
        burying(scored, 0.5, 0, 1)
    with pytest.raises(ValueError, match="the score shares of strategic voters need the ballots"):
        compromising(replace(scored, unscored=None), 0.5, 0, 1).compared(ScoredPairwise(SCORE_BALLOT.scored))
    key = ScoredPairwise(SCORE_BALLOT.scored)
    np.testing.assert_array_equal(_strategy_shares(rankings, probs.reshape(2, 3, -1)).compared(key),
                                  scored.scored_pairwise[key])

# ---------------------------------------------------------------- All together


@pytest.mark.parametrize("bullet, bury, compromise", [(BULLET, 0, 0), (BULLET, BURIAL, COMPROMISE)],
                         ids=["bullet", "all"])
def test_strategies_match_the_references_on_weak_orders(profile, bullet, bury, compromise):
    """Bullet(Burial(Compromise(method, y), q), b): b of the voters bullet vote, and of the
    others q bury and y compromise, each whatever the other does. Against the references
    on the weak orders of all kinds of voters, on the points of each pair of frontrunners:
    the rankings as they are, buried, compromised and both, with (1 - b) times the
    products of the shares, and the bullet ballots with b of each first choice, which
    neither burying nor compromising changes. With bullet voters alone there is no poll."""
    rankings, probs = profile
    voters, n = voters_of(rankings, probs), rankings.shape[1]
    pairs, gap = _pairs(voters)
    first = probs.astype(np.float64) @ np.eye(n)[rankings[:, 0]]
    methods = {name: METHODS[name] for name in REFERENCED}
    for name, method in methods.items():
        strategic = Bullet(Burial(Compromise(method, compromise), bury), bullet)
        result, margin = strategic.evaluate(voters)
        expected, expected_margin = np.empty_like(result), np.empty_like(margin)
        for pair in np.unique(pairs):
            at = pairs == pair
            a, b = np.flatnonzero((pair >> np.arange(n)) & 1)
            every, weights = _compromised_profile(*_buried_profile(rankings, probs[at].astype(np.float64), bury, a, b),
                                                  compromise, a, b)
            levels = np.concatenate([levels_of(every), 1 - np.eye(n, dtype=np.int64)])
            shares = np.concatenate([(1 - bullet) * weights, bullet * first[at]], axis=-1)
            if name == "dac":
                expected[at], expected_margin[at] = dac_reference(levels, shares)
            elif isinstance(method, Eliminate) and isinstance(method.totals, Tally):  # not minimax: Weakest
                left, expected_margin[at] = elimination_reference(method.totals.ballot, method.how, levels, shares,
                                                                  weigh=_tied_weights)
                expected[at] = left.argmax(axis=-1)
            else:
                tied_first = shares @ np.eye(n)[np.argmin(levels, axis=1)]  # every ballot has one first
                expected[at], expected_margin[at] = method.evaluate(
                    Voters(first=tied_first, pairwise=_tied_pairwise(levels, shares)))
        if bury or compromise:
            expected_margin = np.minimum(expected_margin, gap)
        clear = expected_margin > TOLERANCE
        assert clear.mean() > 0.5, name  # not vacuous
        np.testing.assert_array_equal(result[clear], expected[clear], err_msg=name)
        np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5, err_msg=name)


@pytest.mark.parametrize("bullet, bury, compromise", [(0.3, BURIAL, COMPROMISE), (0, 1, 1), (0.5, 0.2, 0)])
def test_star_with_all_strategies_together(profile, bullet, bury, compromise):
    """Bullet(Burial(Compromise(star, y), q), b): the score comparisons are moved once, for
    the frontrunners of each point, with the shares q and y of the burying and
    compromising voters, each whatever the other does, for the 1 - b of the voters who do
    not bullet vote; the bullet voters' comparisons are their first choices'."""
    voters = _strategy_shares(*profile)
    grid, n = _grid_of(voters), voters.n_candidates
    star = METHODS["star"]
    winner, margin = Bullet(Burial(Compromise(star, compromise), bury), bullet).evaluate(voters)
    pairs, gap = _pairs(voters)
    key, comparisons = SCORE_BALLOT.scored, ScoredPairwise(SCORE_BALLOT.scored)
    moved = bulleted(voters, bullet)
    expected, expected_margin = np.empty_like(winner), np.empty_like(margin)
    noted = set()
    for pair in np.unique(pairs):
        at = pairs == pair
        a, b = np.flatnonzero((pair >> np.arange(n)) & 1)
        unscored = moved.unscored[key][at].copy()
        for c, e, sign, values, share in ((a, b, 1, grid.taken_all, bury), (b, a, 1, grid.taken_all, bury),
                                          (a, b, -1, grid.added_all, compromise),
                                          (b, a, -1, grid.added_all, compromise)):
            unscored[:, c] += sign * share * (1 - bullet) * values[at][:, c, e]
        compared = moved.scored_pairwise[comparisons][at] + (1 - bullet) * grid.moved_all(a, b, bury, compromise)[at]
        expected[at], expected_margin[at] = star.evaluate(Voters(unscored={key: unscored},
                                                                 scored_pairwise={comparisons: compared}))
        noted.add((int(a), int(b), bury, compromise))
    assert set(grid.calls) == (noted if bury or compromise else set())  # once per pair, both shares together
    np.testing.assert_array_equal(winner, expected)
    np.testing.assert_allclose(margin, np.minimum(expected_margin, gap), rtol=0, atol=1e-12)

# ---------------------------------------------------------------- Tilted strategies


class _PartGrid:
    """The part `weight` (..., ) of a _Grid at every point: the ballots of the voters of a
    strip, of whom that part of the voters of each point are."""

    def __init__(self, grid, weight):
        self.grid, self.weight = grid, weight

    def _at(self, cells):
        return self.weight[cells[:, 0], cells[:, 1]]

    def taken(self, a, b, cells):
        return self.grid.taken(a, b, cells) * self._at(cells)[:, None]

    def added(self, a, b, cells):
        return self.grid.added(a, b, cells) * self._at(cells)[:, None]

    def moved_grades(self, a, b, cells):
        return self.grid.moved_grades(a, b, cells) * self._at(cells)[None, :, None, None]

    def moved(self, a, b, bury, compromise, cells):
        return self.grid.moved(a, b, bury, compromise, cells) * self._at(cells)[:, None, None]


def _cut(voters, striped, seed=0):
    """The voters cut into the strips of `striped`: every ranking in one to three strips,
    at random, with random parts of its voters at every point; the score shares and the
    ballots (_PartGrid) split by random parts at every point too."""
    rng = np.random.default_rng(seed)
    rankings, probs, k = voters.rankings, voters.probs.astype(np.float64), STRIPS
    inside = np.zeros((k, len(rankings)), dtype=bool)
    for r in range(len(rankings)):
        inside[rng.choice(k, size=rng.integers(1, 4), replace=False), r] = True
    inside[np.arange(k), np.arange(k) % len(rankings)] = True  # no strip without voters
    weights = rng.random((k, *probs.shape)) * inside[:, None, None, :]
    weights /= weights.sum(axis=0)
    split = rng.dirichlet(np.ones(k), size=voters.shape)  # (..., K) of the score shares
    parts = []
    for strip in range(k):
        index = np.flatnonzero(inside[strip])
        part = voters_of(rankings[index], (weights[strip] * probs)[..., index])
        w = split[..., strip]

        def scaled(shares, extra):
            return {key: value * w.reshape(w.shape + (1,) * extra) for key, value in shares.items()}

        parts.append(replace(part, index=index, mass=(weights[strip] * probs).sum(axis=-1),
                             unscored=scaled(voters.unscored, 1), scored_pairwise=scaled(voters.scored_pairwise, 2),
                             graded=scaled(voters.graded, 2), cells=voters.cells,
                             ballots={key: _PartGrid(grid, w) for key, grid in voters.ballots.items()}))
    whole = replace(voters, **{name: getattr(voters_of(rankings, probs), name) for name in ("first", "pairwise")})
    return replace(whole, probs=probs, strips=tuple(parts), striped=striped)


def _scattered(part_probs, parts, rows):
    """The shares (..., rows) of the rankings of all voters from those of each strip."""
    out = np.zeros((*part_probs[0].shape[:-1], rows))
    for values, part in zip(part_probs, parts):
        out[..., part.index] += values
    return out


TILT = (1.5, -0.8)


def test_tilted_strategies_are_blocks():
    """A tilt prints, takes two finite numbers and needs the voters cut into strips across
    it, the same strips for parallel tilts and for opposite ones; untilted strategies
    inside or around a tilted one read the same strips, and tilts in two directions
    cannot be read at once. A tilt of share 0 is the method itself."""
    borda, irv = METHODS["borda"], METHODS["irv"]
    built = [Bullet(irv, 0.3, tilt=(1.0, 0.0)), Burial(borda, 0.5, tilt=TILT),
             Compromise(irv, 0.2, tilt=(0, 2), poll=Highest(Tally(BordaCount()), n=2)),
             Bullet(Burial(Compromise(irv, 0.3, tilt=TILT), 0.2), 0.1, tilt=(-3.0, 1.6))]
    assert repr(built[0]) == 'Bullet(Eliminate(Tally(Plurality()), how="min"), share=0.3, tilt=(1.0, 0.0))'
    assert repr(built[1]) == "Burial(Highest(Tally(BordaCount())), share=0.5, tilt=(1.5, -0.8))"
    for block in built:
        assert eval(repr(block)) == block
    assert repr(Burial(borda, 0.5, tilt=(0, 0))) == repr(Burial(borda, 0.5))
    assert Burial(borda, 0.5, tilt=(0, 0)) == Burial(borda, 0.5)
    for strategy in STRATEGIES:
        name = strategy.__name__
        for tilt in ((1.0,), (1, 2, 3), (float("nan"), 0), (float("inf"), 0), "ab", 3):
            with pytest.raises(ValueError, match=re.escape(f"{name} tilt must be two finite numbers, got {tilt!r}")):
                strategy(irv, 0.5, tilt=tilt)
        assert strategy(borda, 0.5, tilt=TILT).needs == strategy(borda, 0.5).needs | {Striped.of(TILT)}
        assert strategy(borda, 0, tilt=TILT).needs == borda.needs
    assert Striped.of((2, 1)) == Striped.of((-4, -2)) == Striped.of((1, 0.5))
    assert Striped.of((1, 0)) == Striped.of((-1, 0)) and Striped.of((1, 0)).angle == 0.0
    assert Striped.of(TILT) in Bullet(Burial(irv, 0.2, tilt=TILT), 0.3, tilt=(-3.0, 1.6)).needs
    with pytest.raises(ValueError, match="strategies tilted in different directions at once"):
        Bullet(Burial(irv, 0.2, tilt=TILT), 0.3, tilt=(1, 1))


def test_the_shares_of_the_strips():
    """The mean of sigma(logit share + l t) over the t of each strip, l the tilt along the
    strips: share itself without a tilt or at 0 and 1, more of the voters in the direction
    of the tilt and fewer opposite it, mirrored for the opposite tilt, and the share at
    the centre, where two strips meet, between theirs. Across the square t runs from -h
    to h, its extent along the direction."""
    striped = Striped.of(TILT)
    u = striped.direction
    h = (abs(u[0]) + abs(u[1])) / 2
    np.testing.assert_allclose(striped.bounds[[0, -1]], [-h, h])
    along = float(np.dot(TILT, u))
    for share in (0.05, 0.3, 0.9):
        shares = striped.shares(share, TILT)
        assert len(shares) == STRIPS and ((0 < shares) & (shares < 1)).all()
        assert (np.diff(shares) > 0).all() == (along > 0)
        np.testing.assert_allclose(striped.shares(share, (-TILT[0], -TILT[1])), shares[::-1], rtol=1e-12)
        assert min(shares[STRIPS // 2 - 1 : STRIPS // 2 + 1]) < share < max(shares[STRIPS // 2 - 1 : STRIPS // 2 + 1])
        t = np.linspace(striped.bounds[:-1], striped.bounds[1:], 20001)  # (points, strips)
        logistic = 1 / (1 + np.exp(-(np.log(share / (1 - share)) + along * t)))
        np.testing.assert_allclose(shares, np.trapezoid(logistic, t, axis=0) / np.diff(striped.bounds), rtol=1e-8)
        np.testing.assert_array_equal(striped.shares(share, (0, 0)), np.full(STRIPS, share))
    for share in (0, 1):
        np.testing.assert_array_equal(striped.shares(share, TILT), np.full(STRIPS, share))
    with pytest.raises(ValueError, match="is not along the strips"):
        striped.shares(0.3, (1, 1))


def test_untilted_strategies_on_voters_cut_into_strips(profile):
    """Each strip moved by the same share and added up: the strategies of the voters
    themselves, bullet voting with the strip's share of the voters for the 1 of the
    pairwise shares, and the comparisons moved strip by strip."""
    rankings, probs = profile
    voters = _strategy_shares(rankings, probs)
    cut = _cut(voters, Striped.of(TILT))
    for name, method in TESTED_METHODS.items():
        for strategic in (Bullet(method, BULLET), Burial(method, BURIAL), Compromise(method, COMPROMISE),
                          Bullet(Burial(Compromise(method, COMPROMISE), BURIAL), BULLET)):
            winner, margin = strategic.evaluate(cut)
            expected, expected_margin = strategic.evaluate(voters)
            clear = expected_margin > TOLERANCE
            np.testing.assert_array_equal(winner[clear], expected[clear], err_msg=f"{name} {strategic!r}")
            np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-6, err_msg=f"{name} {strategic!r}")


def test_tilted_strategies_move_each_strip_by_its_share(profile):
    """Every strip by its own share s_k: of the rankings of strip k, 1 - s_k of the voters
    as they are and s_k bullet voting, buried or compromised; the part of the top score
    not given raised by what s_k of the burying voters of each strip gave the frontrunners,
    and the comparisons by each strip's moved by its share. Ranking for ranking, the
    profile is that of all voters."""
    rankings, probs = profile
    voters = _strategy_shares(rankings, probs)
    striped, n = Striped.of(TILT), rankings.shape[1]
    cut = _cut(voters, striped)
    parts, rows = cut.strips, len(rankings)
    for share in (BULLET, 1.0):
        shares = striped.shares(share, TILT)
        moved = bulleted(cut, share, TILT)
        np.testing.assert_allclose(moved.probs, _scattered([(1 - s) * p.probs for s, p in zip(shares, parts)],
                                                           parts, rows), rtol=0, atol=1e-12)
        np.testing.assert_allclose(moved.bullet, sum(s * p.first for s, p in zip(shares, parts)), rtol=0, atol=1e-12)
        levels, weights = _bullet_ballots(rankings, probs, 0)
        weights = np.concatenate([moved.probs, moved.bullet], axis=-1)
        np.testing.assert_allclose(moved.pairwise, _tied_pairwise(levels, weights), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(moved.rankings, rankings)
    shares = striped.shares(BURIAL, TILT)
    pairs, _ = _pairs(cut)
    key = SCORE_BALLOT.scored
    for vote, moved_profile in ((burying, _buried_profile), (compromising, _compromised_profile)):
        for pair in np.unique(pairs)[:3]:
            at = pairs == pair
            a, b = np.flatnonzero((pair >> np.arange(n)) & 1)
            moved = vote(cut.at(at), BURIAL, a, b, TILT)
            every = [moved_profile(rankings[p.index], p.probs[at], s, a, b) for s, p in zip(shares, parts)]
            # the rankings of strip k at rows 2 r and their moved ones at 2 r + 1 of all voters
            expected = np.zeros((at.sum(), 2 * rows))
            for (_, weights), p in zip(every, parts):
                expected[:, 2 * p.index] += weights[:, :len(p.index)]
                expected[:, 2 * p.index + 1] += weights[:, len(p.index):]
            np.testing.assert_allclose(moved.probs, expected, rtol=0, atol=1e-12)
            reference = voters_of(moved.rankings, expected)
            np.testing.assert_allclose(moved.pairwise, reference.pairwise, rtol=0, atol=1e-12)
            np.testing.assert_allclose(moved.first, reference.first, rtol=0, atol=1e-12)
            points, sign = ("taken", 1) if vote is burying else ("added", -1)
            unscored = cut.unscored[key][at].copy()
            for s, p in zip(shares, parts):
                unscored[:, [a, b]] += sign * s * getattr(p.grid(key), points)(a, b, cut.cells[at])
            np.testing.assert_allclose(moved.unscored[key], unscored, rtol=0, atol=1e-12)
            bury, compromise = (1, 0) if vote is burying else (0, 1)
            compared = cut.scored_pairwise[ScoredPairwise(key)][at] + sum(
                p.grid(key).moved(a, b, bury * s, compromise * s, cut.cells[at]) for s, p in zip(shares, parts))
            np.testing.assert_allclose(moved.compared(ScoredPairwise(key)), compared, rtol=0, atol=1e-12)


def test_a_tilt_needs_the_voters_cut_into_strips(profile):
    voters = voters_of(*profile)
    for vote in (lambda v: bulleted(v, 0.3, TILT), lambda v: burying(v, 0.3, 0, 1, TILT)):
        with pytest.raises(ValueError, match=re.escape(f"needs the voters cut into strips, {Striped.of(TILT)}")):
            vote(voters)
    assert bulleted(voters, 0, TILT) is voters


# the voters of a model, cut into strips by regions.voters
MODELS = [Model("beta", 0.2, "rms"), Model("normal", 0.15)]
MODEL_CANDIDATES = np.array([[0.25, 0.4], [0.75, 0.4], [0.5, 0.7], [0.45, 0.2]])


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_the_voters_of_the_strips_of_a_model(model):
    """The shares of the strips add up to those of all voters: of the polygons, cut by
    the strips and scaled to them, and of the rectangles of the voter grid, which the
    strips split among them (strip_masks). Each strip has the voters across the tilt
    from it: more of the first one at the wall it starts from, and the strips run along
    the direction."""
    method = Bullet(Burial(Compromise(METHODS["star"] | METHODS["mj"] | METHODS["irv"] | METHODS["borda"], 0.3,
                                      tilt=(2, 0)), 0.3), 0.2)
    voters = regions.voters(method.needs, MODEL_CANDIDATES, model, 12)
    parts = voters.strips
    assert len(parts) == STRIPS and voters.striped == Striped.of((1, 0))
    np.testing.assert_allclose(sum(part.mass for part in parts), 1, rtol=0, atol=1e-5)
    for name in ("first", "pairwise"):
        np.testing.assert_allclose(sum(getattr(part, name) for part in parts), getattr(voters, name), rtol=0, atol=2e-6)
    probs = np.zeros(voters.probs.shape)
    for part in parts:
        probs[..., part.index] += part.probs
        np.testing.assert_array_equal(part.rankings, voters.rankings[part.index])
    np.testing.assert_allclose(probs, voters.probs, rtol=0, atol=2e-6)
    assert set(np.concatenate([part.index for part in parts])) == set(range(len(voters.rankings)))
    for name in ("unscored", "graded", "scored_pairwise"):
        for key, value in getattr(voters, name).items():
            np.testing.assert_allclose(sum(getattr(part, name)[key] for part in parts), value, rtol=1e-12, atol=1e-15)
    assert (parts[0].mass[1] > parts[-1].mass[1]).all() and (parts[0].mass[-2] < parts[-1].mass[-2]).all()
    masks = strip_masks(model, Striped.of((1, 0)).direction, Striped.of((1, 0)).bounds)
    np.testing.assert_array_equal(masks.sum(axis=0), 1)
    strip = np.tensordot(np.arange(STRIPS), masks, 1)  # the strip of each rectangle
    assert (np.diff(strip, axis=0) >= 0).all() and (np.diff(strip, axis=1) == 0).all()
    assert strip[0, 0] == 0 and strip[-1, 0] == STRIPS - 1


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_more_voters_vote_strategically_in_the_direction_of_the_tilt(model):
    """Bullet voters tilted along x: more of them to the right, as many up and down, and
    at the centre about the share itself."""
    method = Bullet(METHODS["irv"], 0.3, tilt=(3, 0))
    voters = regions.voters(method.needs, MODEL_CANDIDATES, model, 12)
    share = bulleted(voters, 0.3, (3, 0)).bullet.sum(axis=-1)
    assert (np.diff(share, axis=0) > 0).all()
    assert share[0].max() < 0.3 < share[-1].min()
    assert abs(share[7, 7] - 0.3) < 0.05


def _strategic(method, tilt):
    return Bullet(Burial(Compromise(method, 0.3, tilt=tilt), 0.3, tilt=tilt), 0.2, tilt=tilt)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_a_tilt_of_almost_0_is_the_strategy_itself(model):
    """The strips add up to the voters themselves, so a tilt all but 0 moves no winner
    and the margins only by rounding."""
    for name in ("fptp", "borda", "irv", "schulze", "star", "mj"):
        _, winner, margin = regions.winners(_strategic(METHODS[name], (0, 0)), MODEL_CANDIDATES, model, 12)
        _, tilted, tilted_margin = regions.winners(_strategic(METHODS[name], (1e-9, 1e-9)), MODEL_CANDIDATES, model,
                                                   12)
        clear = margin > 1e-5
        assert clear.mean() > 0.8, name
        np.testing.assert_array_equal(tilted[clear], winner[clear], err_msg=name)
        np.testing.assert_allclose(tilted_margin, margin, rtol=0, atol=1e-6, err_msg=name)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.distribution)
def test_tilts_mirrored_across_the_square(model):
    """Candidates symmetric across x = 1/2: a strategy tilted by (l, m) is the one tilted
    by (-l, m) seen in the mirror, its winners the mirrored candidates. The edge
    integrals are only symmetric to their quadrature, so the margins to 1e-4."""
    candidates = np.array([[0.25, 0.4], [0.75, 0.4], [0.5, 0.7]])
    mirrored = np.array([1, 0, 2])
    for name in ("borda", "irv", "star"):
        _, right, right_margin = regions.winners(_strategic(METHODS[name], (2, 1)), candidates, model, 12)
        _, left, left_margin = regions.winners(_strategic(METHODS[name], (-2, 1)), candidates, model, 12)
        left, left_margin = mirrored[left[::-1]], left_margin[::-1]
        clear = np.minimum(right_margin, left_margin) > 1e-3
        assert clear.mean() > 0.7, name
        np.testing.assert_array_equal(left[clear], right[clear], err_msg=name)
        np.testing.assert_allclose(left_margin, right_margin, rtol=0, atol=1e-4, err_msg=name)
        if name == "borda":
            assert (right != regions.winners(_strategic(METHODS[name], (0, 0)), candidates, model, 12)[1]).any()
