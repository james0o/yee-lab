"""Checks of the voting methods (yeelab.build.METHODS) against plain Python references
that follow their definitions.

The methods of pairwise shares are checked on random pairwise shares, with many cycles
and ties, exactly: Schulze and the Condorcet winner, which needs the strongest paths only
in a cycle; minimax; Black, the Condorcet winner, else Borda; king of the hill, and
Unbeaten against a cycle. A runoff, Unbeaten(diffs, among=first | second), is checked
against its duel between the built finalists, a method against itself and swapped
finalists, and the two-round system and STAR, whose finalists are Highest(totals, n=2),
likewise; Unbeaten among everyone is Unbeaten itself. The Smith set is checked against
its definition, and its margin against the smallest change of the diffs that changes
it, which keeps it continuous.

The methods of the whole profile are checked on the ranking shares of the profiles
(conftest.py): every elimination, down to one and to two candidates (Highest is
Eliminate(totals, how="all")); Highest(totals, among=finalists), the highest total among
the finalists, the Smith set too; IRV among finalists; and DAC, whose margin also bounds
how much the shares must change to change the winner. The references are those of
references.py and those below.
"""

from itertools import product

import numpy as np
import pytest

from yeelab.build import (
    METHODS,
    BordaCount,
    DescendingCoalitions,
    Eliminate,
    Highest,
    Margins,
    Pairwise,
    Plurality,
    Score,
    ScoreComparisons,
    Scored,
    ScoredPairwise,
    Smith,
    StrongestPaths,
    Tally,
    Unbeaten,
    Undecided,
    Voters,
    Weakest,
)
from yeelab.voting import CYCLE

from references import (
    TOLERANCE,
    dac_reference,
    elimination_reference,
    levels_of,
    random_profiles,
    ranking_weights,
    voters_of,
)

BALLOTS = [Plurality(), BordaCount()]
# every elimination: the four methods and the one other combination of these blocks
ELIMINATIONS = [METHODS["irv"], METHODS["baldwin"], METHODS["nanson"],
                Eliminate(Tally(Plurality()), how="mean")]
# the candidates kept down to two, and Highest's one round of "all", down to one too
KEPT = [Eliminate(Tally(Plurality()), how="min", until=2), Eliminate(Tally(BordaCount()), how="min", until=2),
        Eliminate(Tally(BordaCount()), how="mean", until=2), Eliminate(Tally(Plurality()), how="mean", until=2),
        Highest(Tally(Plurality()), n=2), Highest(Tally(BordaCount()), n=2), METHODS["fptp"], METHODS["borda"]]


def _condorcet_reference(d):
    """The Condorcet winner (CYCLE if none) and margin from the definition: worst[c] =
    min_e (d[c, e] - d[e, c]) is c's narrowest head-to-head result, the winner has the
    best, if it is positive, and the margin is its absolute value."""
    n = d.shape[-1]
    lead = d - np.swapaxes(d, -1, -2)
    worst = np.where(np.eye(n, dtype=bool), np.inf, lead).min(axis=-1)
    best = worst.max(axis=-1)
    return np.where(best > 0, worst.argmax(axis=-1), CYCLE), np.abs(best)


def _schulze_reference(d):
    """Schulze from the definition: the Condorcet winner and margin where there is one,
    elsewhere the widest paths of d - d^T."""
    winner, margin = _condorcet_reference(d)
    cycle = winner == CYCLE
    if cycle.any():
        d = d[cycle]
        n = d.shape[-1]
        p = np.maximum(d - np.swapaxes(d, -1, -2), 0.0)
        through = np.empty_like(p)
        for k in range(n):
            np.minimum(p[..., :, k, None], p[..., None, k, :], out=through)
            np.maximum(p, through, out=p)
        beaten = (p - np.swapaxes(p, -1, -2)).max(axis=-2)  # [..., e] = max_f p[f, e] - p[e, f]
        paths_winner = (beaten <= 0).argmax(axis=-1)
        others = np.where(np.arange(n) == paths_winner[..., None], np.inf, beaten)
        winner[cycle], margin[cycle] = paths_winner, np.maximum(others.min(axis=-1), 0.0)
    return winner, margin


def _minimax_reference(d):
    """Minimax point by point from its definition: the candidate whose largest defeat,
    max_e (d[e, c] - d[c, e]) (negative if it beats everyone), is the smallest (ties: the
    lowest index); the margin is the next smallest largest defeat minus its own."""
    n = d.shape[-1]
    winner = np.empty(d.shape[:-2], dtype=np.int64)
    margin = np.empty(d.shape[:-2])
    for point in np.ndindex(*d.shape[:-2]):
        shares = d[point].astype(np.float64)
        defeat = [max(shares[e, c] - shares[c, e] for e in range(n) if e != c) for c in range(n)]
        order = sorted(range(n), key=lambda c: (defeat[c], c))
        winner[point], margin[point] = order[0], defeat[order[1]] - defeat[order[0]]
    return winner, margin


def _condorcet_else_borda(d):
    """The Condorcet winner (beats everyone head to head), else the highest Borda score
    sum_e d[c, e]; ties: the lowest index."""
    wins = (d > np.swapaxes(d, -1, -2)).sum(axis=-1)
    return np.where(wins.max(axis=-1) == d.shape[-1] - 1, wins.argmax(axis=-1), d.sum(axis=-1).argmax(axis=-1))


def _tournaments(n, count=500, seed=0):
    """Random pairwise shares d (count, n, n), d + d^T = 1 off the diagonal, in float32:
    mostly cycles for larger n, and every tenth point on a grid of quarters, where
    shares tie."""
    rng = np.random.default_rng(seed + n)
    above = rng.random((count, n, n))
    above[::10] = np.round(4 * above[::10]) / 4
    d = np.triu(above, 1)
    d = d + np.swapaxes(np.triu(1 - above, 1), -1, -2)
    return d.astype(np.float32)


TOURNAMENTS = [_tournaments(n) for n in (2, 3, 4, 5, 8)]


def _smith_reference(s, by=0.0):
    """The Smith set point by point from its definition, for diffs s (..., C, C): the
    smallest set of candidates who each beat every candidate outside it, s[c, e] > by.
    Such sets are nested, so the smallest is the one of the fewest candidates."""
    n = s.shape[-1]
    smith = np.zeros(s.shape[:-1], dtype=bool)
    found = np.zeros(s.shape[:-2], dtype=bool)
    for mask in sorted(range(1, 1 << n), key=lambda mask: bin(mask).count("1")):
        inside = ((mask >> np.arange(n)) & 1).astype(bool)
        dominant = (s[..., inside, :][..., :, ~inside] > by).all(axis=(-2, -1)) & ~found
        smith[dominant] = inside
        found |= dominant
    return smith


def _smith_distance(s, smith, by=0.0):
    """The smallest change of the diffs s (..., C, C), each pair by at most that much,
    that changes the Smith set from `smith`, where c beats e if s[c, e] > by. Every pair
    is in one of the states 1 (the first beats the second), -1 (the second the first)
    and, with by > 0, 0 (neither, as a tie at by = 0): over every state of every pair,
    the largest change a pair needs to get into its state, where the set of those states
    differs."""
    n = s.shape[-1]
    pairs = [(c, e) for c in range(n) for e in range(c + 1, n)]
    states = np.array(list(product((1.0, -1.0) if by == 0 else (1.0, 0.0, -1.0), repeat=len(pairs))))
    tournaments = np.zeros((len(states), n, n))
    for k, (c, e) in enumerate(pairs):
        tournaments[:, c, e], tournaments[:, e, c] = states[:, k], -states[:, k]
    sets = _smith_reference(tournaments, 0.5)
    diffs = np.stack([s[..., c, e] for c, e in pairs], axis=-1)[..., None, :]
    change = np.where(states == 1, np.maximum(by - diffs, 0.0),
                      np.where(states == -1, np.maximum(diffs + by, 0.0), np.maximum(np.abs(diffs) - by, 0.0)))
    changed = (sets != smith[..., None, :]).any(axis=-1)
    return np.where(changed, change.max(axis=-1), np.inf).min(axis=-1)


def _first_choices(n, count=500, seed=0):
    """Random first-choice shares (count, n) to go with _tournaments(n), every tenth
    point on a grid of quarters, where they tie. They do not sum to 1: king of the hill
    only compares them."""
    first = np.random.default_rng(seed + n).random((count, n))
    first[::10] = np.round(4 * first[::10]) / 4
    return first


def _challengers(first, d):
    """How many candidates beat the one with the most first choices head to head, (...)."""
    king = first.argmax(axis=-1)[..., None, None]
    lead = d - np.swapaxes(d, -1, -2)
    return (np.take_along_axis(lead, king, axis=-1) > 0).sum(axis=(-1, -2))


def _koth_reference(first, d):
    """King of the hill point by point from its rule: the king has the most first choices
    (ties: the lowest index), its challengers are the candidates c with d[c, king] >
    d[king, c], and the challenger with the most first choices wins (ties: the lowest
    index), or the king if there is none. The margin is the smallest of the king's lead
    in first choices, |d[c, king] - d[king, c]| of every other candidate c, and, with two
    challengers or more, the lead in first choices of the best over the second."""
    n = first.shape[-1]
    winner = np.empty(first.shape[:-1], dtype=np.int64)
    margin = np.empty(first.shape[:-1])
    for point in np.ndindex(*first.shape[:-1]):
        votes, shares = first[point].tolist(), d[point].tolist()
        king, *others = sorted(range(n), key=lambda c: (-votes[c], c))  # others: best first
        lead = {c: shares[c][king] - shares[king][c] for c in others}
        challengers = [c for c in others if lead[c] > 0]
        gaps = [votes[king] - votes[others[0]], *(abs(x) for x in lead.values())]
        if len(challengers) > 1:
            gaps.append(votes[challengers[0]] - votes[challengers[1]])
        winner[point], margin[point] = (challengers or [king])[0], min(gaps)
    return winner, margin


def _runoff_reference(diffs, first, second, transitive=False):
    """A runoff point by point from its rule, for the diffs s (..., C, C) and the (winner,
    margin) of both finalists, of which one that is CYCLE drops out: between two, the
    second wins where s[second][first] > 0, the first where s[first][second] > 0, and a
    tie is a draw (CYCLE), on transitive diffs the lower index wins; one wins alone, and
    without any the point is a CYCLE. The margin is the smallest of the finalists'
    margins and, where they are two candidates, |s[first][second]|."""
    (a, margin_a), (b, margin_b) = first, second
    winner = np.empty(a.shape, dtype=np.int64)
    margin = np.empty(a.shape)
    for point in np.ndindex(*a.shape):
        x, y, s = int(a[point]), int(b[point]), diffs[point].tolist()
        gaps = [float(margin_a[point]), float(margin_b[point])]
        running = {x, y} - {CYCLE}
        if len(running) < 2:
            winner[point] = running.pop() if running else CYCLE
        else:
            tie = min(x, y) if transitive else CYCLE
            winner[point] = y if s[y][x] > 0 else x if s[x][y] > 0 else tie
            gaps.append(abs(s[x][y]))
        margin[point] = min(gaps)
    return winner, margin


def _runoff(diffs, first, second):
    """The winners of first and second head to head on the diffs; a tie is a draw."""
    return Unbeaten(diffs, among=first | second)


def _duel(runoff):
    """(diffs, first, second) of a _runoff."""
    return runoff.diffs, runoff.among.first, runoff.among.second


def _finalists(totals):
    """(the highest, the second highest) totals (..., C) as two (winner, margin), ties to
    the lowest index, both with the margin of the second over the third (inf with two
    candidates): their order does not matter in a duel."""
    order = np.argsort(-totals, axis=-1, kind="stable")
    ranked = np.take_along_axis(totals, order, axis=-1)
    gap = ranked[..., 1] - ranked[..., 2] if totals.shape[-1] > 2 else np.full(totals.shape[:-1], np.inf)
    return (order[..., 0], gap), (order[..., 1], gap)


# finalists that agree, differ and tie, one that elects no one in a cycle (condorcet),
# and a duel on the strongest paths
RUNOFFS = [
    _runoff(Margins(Pairwise()), METHODS["koth"], METHODS["borda"]),
    _runoff(Margins(Pairwise()), METHODS["condorcet"], METHODS["minimax"]),
    _runoff(Margins(Pairwise()), METHODS["fptp"], METHODS["condorcet"]),
    _runoff(StrongestPaths(Margins(Pairwise())), METHODS["borda"], METHODS["fptp"]),
]


def _highest_among_reference(totals, chosen, margin):
    """The highest of the totals (..., C) among the candidates in chosen (ties: the
    lowest index), CYCLE where there are none, and the smaller of margin and its lead
    over the next of them (inf if it is alone)."""
    winner = np.empty(margin.shape, dtype=np.int64)
    expected = np.empty(margin.shape)
    for point in np.ndindex(*margin.shape):
        running = np.flatnonzero(chosen[point])
        order = running[np.argsort(-totals[point][running], kind="stable")]
        winner[point] = order[0] if len(order) else CYCLE
        lead = totals[point][order[0]] - totals[point][order[1]] if len(order) > 1 else np.inf
        expected[point] = min(margin[point], lead)
    return winner, expected


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
@pytest.mark.parametrize("name", ["schulze", "condorcet"])
def test_condorcet_and_schulze_match_references(d, name):
    """Exactly, bit for bit, on random pairwise shares: many cycles, and ties on the quarters."""
    expected = {"schulze": _schulze_reference, "condorcet": _condorcet_reference}[name](d)
    winner, margin = METHODS[name].evaluate(Voters(pairwise=d))
    np.testing.assert_array_equal(winner, expected[0])
    np.testing.assert_array_equal(margin, expected[1])
    if name == "condorcet":  # not vacuous: both cycles (or ties) and winners are in there
        assert (winner == CYCLE).any() and (winner != CYCLE).any()


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_schulze_paths_are_only_needed_in_cycles(d):
    """Unbeaten(StrongestPaths(...)) takes the winner and margin of the direct diffs where
    there is a Condorcet winner. The paths everywhere (decide) elect the same and have a
    margin at least as large."""
    schulze = METHODS["schulze"]
    voters = Voters(pairwise=d)
    winner, margin = schulze.evaluate(voters)
    full_winner, full_margin = schulze.decide(voters)
    np.testing.assert_array_equal(winner, full_winner)
    assert (margin <= full_margin).all()
    cycle = METHODS["condorcet"].evaluate(voters)[0] == CYCLE
    np.testing.assert_array_equal(margin[cycle], full_margin[cycle])


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_minimax_matches_reference(d):
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    winner, margin = METHODS["minimax"].evaluate(Voters(pairwise=d))
    expected_winner, expected_margin = _minimax_reference(d)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_black_is_condorcet_else_borda(d):
    """The Condorcet winner with its margin where there is one; in a cycle the Borda
    winner, with the smaller of the two margins."""
    voters = Voters(pairwise=d)
    winner, margin = METHODS["black"].evaluate(voters)
    np.testing.assert_array_equal(winner, _condorcet_else_borda(d))
    condorcet, margin_condorcet = METHODS["condorcet"].evaluate(voters)
    margin_borda = METHODS["borda"].evaluate(voters)[1]
    cycle = condorcet == CYCLE
    np.testing.assert_array_equal(margin, np.where(cycle, np.minimum(margin_condorcet, margin_borda), margin_condorcet))
    assert cycle.any() and (winner != CYCLE).all()  # not vacuous


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_koth_matches_reference(d):
    """Random first choices next to the random pairwise shares: kings without a
    challenger, with one and (from three candidates on) with several, and ties."""
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    n = d.shape[-1]
    first = _first_choices(n, len(d))
    winner, margin = METHODS["koth"].evaluate(Voters(first, d))
    expected_winner, expected_margin = _koth_reference(first, d)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)
    assert set(np.unique(_challengers(first, d))) >= set(range(min(n, 3)))  # not vacuous
    assert np.isfinite(margin).all() and (margin == 0).any()


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_unbeaten_against_a_cycle_stays_a_cycle(d):
    """Against the Condorcet winner: a CYCLE with its margin where there is none. Elsewhere
    no one beats the king, and the closest result against it is its own margin, so the
    method is condorcet again, bit for bit."""
    condorcet = METHODS["condorcet"]
    method = Unbeaten(Margins(Pairwise()), against=condorcet, order=Tally(BordaCount()))
    voters = Voters(pairwise=d)
    winner, margin = method.evaluate(voters)
    expected_winner, expected_margin = condorcet.evaluate(voters)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_array_equal(margin, expected_margin)
    assert (winner == CYCLE).any() and (winner != CYCLE).any()  # not vacuous


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
@pytest.mark.parametrize("method", RUNOFFS, ids=repr)
def test_runoff_matches_reference(d, method):
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    voters = Voters(_first_choices(d.shape[-1], len(d)), d)
    diffs, first, second = _duel(method)
    first, second = first.evaluate(voters), second.evaluate(voters)
    expected_winner, expected_margin = _runoff_reference(diffs.evaluate(voters), first, second, diffs.transitive)
    winner, margin = method.evaluate(voters)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)
    cycles = (first[0] == CYCLE) & (second[0] == CYCLE)
    assert (winner[cycles] == CYCLE).all() and (margin[(winner == CYCLE) & ~cycles] == 0).all()


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_runoff_of_the_same_and_of_swapped_finalists(d):
    """A method against itself is that method, margin and cycles included. Swapping the
    finalists keeps the margin, and the winner wherever the duel is not a tie."""
    margins = Margins(Pairwise())
    voters = Voters(_first_choices(d.shape[-1], len(d)), d)
    fptp, borda = METHODS["fptp"], METHODS["borda"]
    for method in (fptp, borda, METHODS["condorcet"]):
        winner, margin = _runoff(margins, method, method).evaluate(voters)
        expected_winner, expected_margin = method.evaluate(voters)
        np.testing.assert_array_equal(winner, expected_winner)
        np.testing.assert_array_equal(margin, expected_margin)

    winner, margin = _runoff(margins, fptp, borda).evaluate(voters)
    swapped_winner, swapped_margin = _runoff(margins, borda, fptp).evaluate(voters)
    np.testing.assert_array_equal(margin, swapped_margin)
    np.testing.assert_array_equal(winner[margin > 0], swapped_winner[margin > 0])
    if d.shape[-1] > 2:  # not vacuous: the finalists differ, and each of them wins duels
        a, b = fptp.evaluate(voters)[0], borda.evaluate(voters)[0]
        assert ((winner == a) & (a != b)).any() and ((winner == b) & (a != b)).any()


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_two_round_matches_reference(d):
    """The two most first choices as the finalists, head to head; a tie is a draw."""
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    first = _first_choices(d.shape[-1], len(d))
    expected_winner, expected_margin = _runoff_reference(d - np.swapaxes(d, -1, -2), *_finalists(first))
    winner, margin = METHODS["two_round"].evaluate(Voters(first, d))
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)
    assert (margin[winner == CYCLE] == 0).all()  # a draw only on a border
    assert (margin == 0).any() and (winner != CYCLE).any()  # not vacuous


def test_star_matches_reference():
    """Random mean scores (ties on the quarters) and score comparisons (ties on the
    halves): the duel of the two highest mean scores, a tie a draw."""
    ballot, rng, n = Score(6, power=1.5), np.random.default_rng(0), 5
    unscored = rng.random((400, n))
    unscored[::4] = np.round(4 * unscored[::4]) / 4
    above = rng.random((400, n, n))
    above[::3] = np.round(2 * above[::3]) / 2
    diffs = above - np.swapaxes(above, -1, -2)
    voters = Voters(unscored={ballot.scored: unscored},
                    scored_pairwise={ScoredPairwise(ballot.scored): diffs})
    expected_winner, expected_margin = _runoff_reference(diffs, *_finalists(-unscored))
    winner, margin = METHODS["star"].evaluate(voters)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_array_equal(margin, expected_margin)
    assert (winner == CYCLE).any() and (winner != unscored.argmin(axis=-1)).any()  # not vacuous


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_unbeaten_among(d):
    """Among all candidates (margin inf) it is Unbeaten itself, bit for bit; among the
    one winner of a method, that winner with its margin; where the method elects no
    one, no one wins."""
    voters = Voters(_first_choices(d.shape[-1], len(d)), d)
    margins, n = Margins(Pairwise()), d.shape[-1]
    everyone = Highest(Tally(Plurality()), n=n)
    for diffs in (margins, StrongestPaths(margins)):
        winner, margin = Unbeaten(diffs, among=everyone).evaluate(voters)
        expected_winner, expected_margin = Unbeaten(diffs).decide(voters)
        np.testing.assert_array_equal(winner, expected_winner)
        np.testing.assert_array_equal(margin, expected_margin)
    for method in (METHODS["fptp"], METHODS["borda"], METHODS["condorcet"]):
        winner, margin = Unbeaten(margins, among=method).evaluate(voters)
        expected_winner, expected_margin = method.evaluate(voters)
        np.testing.assert_array_equal(winner, expected_winner)
        np.testing.assert_array_equal(margin, expected_margin)


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_smith_set_matches_reference(d):
    """On random pairwise shares, with many cycles and ties: the Smith set of its
    definition, the Condorcet winner alone where there is one, with its margin, and two
    candidates or more where there is none. The margin is the smallest change of the
    diffs that changes the set, exactly, ties included (up to five candidates, of 2^10
    signs of the pairs)."""
    voters = Voters(pairwise=d)
    s = Margins(Pairwise()).evaluate(voters)
    chosen, margin = Smith(Margins(Pairwise())).select(voters)
    np.testing.assert_array_equal(chosen, _smith_reference(s))
    winner, condorcet_margin = METHODS["condorcet"].evaluate(voters)
    decided = winner != CYCLE
    np.testing.assert_array_equal(chosen[decided], winner[decided, None] == np.arange(d.shape[-1]))
    np.testing.assert_array_equal(margin[decided], condorcet_margin[decided])
    assert (chosen[~decided].sum(axis=-1) > 1).all() and (~decided).any() and decided.any()  # not vacuous
    if d.shape[-1] <= 5:
        np.testing.assert_array_equal(margin, _smith_distance(s, chosen))


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
@pytest.mark.parametrize("majority", [0.55, 0.75])
def test_smith_with_a_larger_majority(d, majority):
    """A candidate beats another only with `majority` of the voters, where the diff is
    above 2 majority - 1: the set of that definition, which holds the Smith set of a
    majority and grows with `majority`, to all candidates at 1. The margin is again the
    smallest change of the diffs that changes the set, a pair neither one wins so being
    a state of its own (up to four candidates, of 3^6 states)."""
    voters, margins = Voters(pairwise=d), Margins(Pairwise())
    s, by = margins.evaluate(voters), 2 * majority - 1
    chosen, margin = Smith(margins, majority=majority).select(voters)
    np.testing.assert_array_equal(chosen, _smith_reference(s, by))
    smaller = Smith(margins, majority=(0.5 + majority) / 2).select(voters)[0]
    assert (chosen >= smaller).all() and (smaller >= Smith(margins).select(voters)[0]).all()
    assert (chosen.sum(axis=-1) > smaller.sum(axis=-1)).any()  # not vacuous
    assert Smith(margins, majority=1).select(voters)[0].all()
    if d.shape[-1] <= 4:
        np.testing.assert_array_equal(margin, _smith_distance(s, chosen, by))


@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("majority", [0.5, 0.6])
def test_smith_margin_is_continuous(n, majority):
    """As a distance in the diffs the margin changes by at most as much as they do, also
    where a pair inside the set changes who wins it and the set stays; where the set
    changes, it is at most the step along the way."""
    d = _tournaments(n).astype(np.float64)
    t = np.linspace(0, 1, 401)[:, None, None, None]
    path = (1 - t) * d[:100] + t * d[100:200]
    chosen, margin = Smith(Margins(Pairwise()), majority=majority).select(Voters(pairwise=path))
    s, by = path - np.swapaxes(path, -1, -2), 2 * majority - 1
    step = np.abs(np.diff(s, axis=0)).max(axis=(-2, -1))
    assert (np.abs(np.diff(margin, axis=0)) <= step + 1e-12).all()
    change = (chosen[1:] != chosen[:-1]).any(axis=-1)
    assert change.sum() > 5  # not vacuous: with 8 candidates and 60% the set is nearly always all of them
    assert (np.maximum(margin[1:], margin[:-1]) <= step + 1e-12)[change].all()
    inside = chosen[..., :, None] & chosen[..., None, :]
    state = (s > by).astype(int) - (s < -by)  # who wins the pair: 1, -1, or 0 neither
    flips = ((state[1:] != state[:-1]) & inside[1:] & inside[:-1]).any(axis=(-2, -1)) & ~change
    assert flips.any() or n < 5  # not vacuous: the set stays where a pair inside it changes


def test_smith_of_score_comparisons():
    """Of any diffs that can form a cycle: the runoff comparisons of a score ballot too."""
    ballot, n = Score(6), 5
    above = np.random.default_rng(0).random((400, n, n))
    diffs = above - np.swapaxes(above, -1, -2)
    voters = Voters(scored_pairwise={ScoredPairwise(ballot.scored): diffs})
    smith = Smith(ScoreComparisons(ballot))
    assert smith.needs == {ScoredPairwise(ballot.scored)}
    chosen, margin = smith.select(voters)
    np.testing.assert_array_equal(chosen, _smith_reference(diffs))
    np.testing.assert_array_equal(margin, _smith_distance(diffs, chosen))


@pytest.mark.parametrize("method", ELIMINATIONS, ids=repr)
def test_eliminations_match_reference(profile, method):
    """Both the method (for irv the compiled voting.irv_rounds) and its rounds with
    one tally each."""
    rankings, probs = profile
    voters = voters_of(rankings, probs)
    left, expected_margin = elimination_reference(method.totals.ballot, method.how, rankings, probs)
    expected_winner = left.argmax(axis=-1)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous; near-empty candidates tie, e.g. in IRV
    for winner, margin in (method.evaluate(voters), method.rounds(voters)):
        np.testing.assert_array_equal(winner[clear], expected_winner[clear])
        np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)


@pytest.mark.parametrize("method", KEPT, ids=repr)
def test_the_candidates_kept_match_reference(profile, method):
    """The candidates left, from the rounds of each and from Highest's one."""
    rankings, probs = profile
    voters = voters_of(rankings, probs)
    expected, expected_margin = elimination_reference(method.totals.ballot, method.how, rankings, probs, method.until)
    chosen, margin = method.select(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(chosen[clear], expected[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)
    assert (chosen.sum(axis=-1) == method.until).all()


def test_highest_among_the_finalists(profile):
    """The highest total among the candidates a Winner chooses, with the smaller margin;
    a Tally counted again among them, other totals as they are. The most first choices
    between the two with the most is the one more voters rank above the other: the
    two-round system (where its duel ties, a draw, the lower index)."""
    rankings, probs = profile
    voters = voters_of(rankings, probs)
    weakest = Weakest(Margins(Pairwise()))
    for among in (METHODS["fptp"] | METHODS["irv"], Highest(Tally(BordaCount()), n=3), METHODS["koth"],
                  Smith(Margins(Pairwise()))):
        chosen, gap = among.select(voters)
        for totals, counted in ((Tally(BordaCount()), BordaCount().tally(voters, chosen)),
                                (weakest, weakest.evaluate(voters))):
            winner, margin = Highest(totals, among=among).evaluate(voters)
            expected, expected_margin = _highest_among_reference(counted, chosen, gap)
            clear = expected_margin > TOLERANCE
            np.testing.assert_array_equal(winner[clear], expected[clear])
            np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)

    runoff = Highest(Tally(Plurality()), among=Highest(Tally(Plurality()), n=2))
    winner, margin = runoff.evaluate(voters)
    expected, expected_margin = METHODS["two_round"].evaluate(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)


@pytest.mark.parametrize("among", [Highest(Tally(Plurality()), n=3), Smith(Margins(Pairwise()))], ids=repr)
def test_irv_among_finalists_matches_reference(profile, among):
    """The finalists, the top three of the first choices or the Smith set (Smith//IRV),
    seed the IRV rounds and contribute their gap."""
    rankings, probs = profile
    voters = voters_of(rankings, probs)
    method = Eliminate(Tally(Plurality()), how="min", among=among)
    finalists, finalist_margin = among.select(voters)
    expected = np.zeros_like(finalists)
    expected_margin = finalist_margin.copy()

    for point in np.ndindex(*probs.shape[:-1]):
        alive = finalists[point].copy()
        gaps = []
        while alive.sum() > 1:
            weights = ranking_weights(Plurality(), rankings, alive)
            totals = probs[point] @ weights
            remaining = np.flatnonzero(alive)
            order = np.argsort(totals[remaining], kind="stable")
            gaps.append(totals[remaining[order[1]]] - totals[remaining[order[0]]])
            alive[remaining[order[0]]] = False
        expected[point] = alive
        expected_margin[point] = min([finalist_margin[point], *gaps])

    winner, margin = method.evaluate(voters)
    clear = expected_margin > TOLERANCE
    np.testing.assert_array_equal(winner[clear], expected.argmax(axis=-1)[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)
    round_winner, round_margin = method.rounds(voters)
    np.testing.assert_array_equal(round_winner, winner)
    np.testing.assert_array_equal(round_margin, margin)


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_highest_among_no_one_is_no_one(d):
    """Among the Condorcet winner: it wins alone, with its own margin; in a cycle no one
    runs and no one wins, with the margin of the cycle."""
    if d.shape[-1] < 3:
        return
    unscored = np.random.default_rng(0).random(d.shape[:-1])
    voters = Voters(pairwise=d, unscored={Scored(6): unscored})
    winner, margin = Highest(Tally(Score(6)), among=METHODS["condorcet"]).evaluate(voters)
    expected, expected_margin = METHODS["condorcet"].evaluate(voters)
    assert (expected == CYCLE).any() and (expected != CYCLE).any()  # not vacuous
    np.testing.assert_array_equal(winner, expected)
    np.testing.assert_array_equal(margin, expected_margin)


def test_dac_matches_reference(profile):
    rankings, probs = profile
    expected, expected_margin = dac_reference(levels_of(rankings), probs)
    winner, margin = METHODS["dac"].evaluate(voters_of(rankings, probs))
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.9  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_dac_margin_bounds_the_change_of_shares_that_changes_the_winner(n):
    """A change of the shares by less than the margin in all (sum of |dp|) keeps the
    winner: the set of a round and its rival begin no ranking together, so their gap
    changes by at most that. Between two profiles the winner changes only where the
    margin is 0, so next to a change it is at most the step along the way."""
    rankings, probs = random_profiles(n, 200)
    method = DescendingCoalitions()
    winner, margin = method.evaluate(Voters(rankings=rankings, probs=probs))
    assert np.isfinite(margin).all() and (margin > 0).all()
    rng = np.random.default_rng(n)
    for _ in range(20):
        dp = rng.normal(size=probs.shape) * (rng.random(probs.shape) < rng.random())
        dp[np.arange(len(dp)), rng.integers(dp.shape[-1], size=len(dp))] += rng.normal(size=len(dp))
        dp *= 0.99 * margin[:, None] / np.abs(dp).sum(axis=-1, keepdims=True)
        np.testing.assert_array_equal(method.evaluate(Voters(rankings=rankings, probs=probs + dp))[0], winner)

    t = np.linspace(0, 1, 401)[:, None, None]
    path = (1 - t) * probs[:100] + t * probs[100:]
    winner, margin = method.evaluate(Voters(rankings=rankings, probs=path))
    change = winner[1:] != winner[:-1]
    step = (t[1, 0, 0] - t[0, 0, 0]) * np.abs(probs[100:] - probs[:100]).sum(axis=-1)
    assert change.sum() > 20  # not vacuous
    assert (np.maximum(margin[1:], margin[:-1]) <= step + 1e-12)[change].all()
