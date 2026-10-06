"""Checks of the methods built from blocks (yeelab.build).

On the complete profile of every pixel (pixels/), the built fptp, irv, borda, schulze
and condorcet must pick the winners of pixels.methods, and every elimination the
winners and margins of a plain Python reference that follows the definition: totals
from weight(k, C) among the remaining candidates, one point at a time. The fast tally
of every ballot must be that same sum over the profile, for any remaining candidates.

schulze and condorcet must give exactly the winners and margins that the
functions of the former margin/methods.py gave (copied below as the reference), on the
profiles and on random pairwise shares with many cycles and ties. minimax is checked
against a Python reference of its definition, black against "the Condorcet winner,
else Borda". koth (king of the hill) is checked against a Python reference of its rule,
winner and margin, on the profiles and on random first-choice and pairwise shares.
A runoff, Unbeaten(diffs, among=first | second), is checked against a
Python reference of its duel between the built finalists: king_runoff on the profiles,
other finalists (with cycles and ties) on random shares, and two_round and STAR, whose
finalists are Highest(totals, n=2), likewise. Highest is Eliminate(totals, how="all"),
and every elimination down to two candidates must keep those of the reference.
The score ballots tally the parts of the top score not given they are given, which
test_score.py checks; here the highest mean score must win (with two levels: the most
approved candidate), with its lead as the margin. A Mix of two ballots must tally the
mix of their voters, and be the one ballot itself where all voters mark it.

Wrong blocks must fail when built, and every method must print as the expression
that builds it.
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
    BordaCount,
    Eliminate,
    Fallback,
    Highest,
    Margins,
    Mix,
    Pairwise,
    Plurality,
    Score,
    ScoreComparisons,
    ScoreAvg,
    ScoreCluster,
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
from yeelab.pixels import beta as pixel_beta, methods as pixel_methods, normal as pixel_normal
from yeelab.voting import CYCLE

PIXELS = 40
TOLERANCE = 1e-6  # winners are compared where the margin is above (float32 profiles)
# voters (distribution, deviation, Beta spread) and random candidates (number, seed)
VOTERS = [("beta", 0.1, "rms"), ("beta", 0.3, "mean_abs"), ("normal", 0.1, None), ("normal", 0.3, None)]
CANDIDATES = [(3, 0), (5, 1), (7, 2)]
BALLOTS = [Plurality(), BordaCount()]
# the blocks below the methods that are no method themselves, and Weakest
PAIR_BLOCKS = [Pairwise(), Margins(Pairwise()), StrongestPaths(Margins(Pairwise())),
               StrongestPaths(StrongestPaths(Margins(Pairwise()))), Weakest(Margins(Pairwise()))]
# every elimination: the four methods and the one other combination of these blocks
ELIMINATIONS = [METHODS["irv"], METHODS["baldwin"], METHODS["nanson"],
                Eliminate(Tally(Plurality()), how="mean")]
# the same down to two candidates, and Highest's one round of "all"
FINALISTS = [Eliminate(Tally(Plurality()), how="min", until=2), Eliminate(Tally(BordaCount()), how="min", until=2),
             Eliminate(Tally(BordaCount()), how="mean", until=2), Eliminate(Tally(Plurality()), how="mean", until=2),
             Highest(Tally(Plurality()), n=2), Highest(Tally(BordaCount()), n=2)]


@pytest.fixture(scope="module", params=list(product(VOTERS, CANDIDATES)),
                ids=lambda p: f"{p[0][0]}-{p[0][1]}-{p[1][0]}")
def profile(request):
    """(rankings (R, C), probabilities (PIXELS, PIXELS, R)) of the pixel pipeline."""
    (distribution, deviation, spread), (n, seed) = request.param
    candidates = np.random.default_rng(seed).random((n, 2))
    if distribution == "beta":
        return pixel_beta.ranking_probabilities(candidates, PIXELS, deviation, spread=spread)
    return pixel_normal.ranking_probabilities(candidates, PIXELS, deviation)


def _voters(rankings, probs):
    """Every share a method may need, from a complete profile."""
    first = probs @ np.eye(rankings.shape[1], dtype=probs.dtype)[rankings[:, 0]]
    d = pixel_methods._pairwise_preferences(rankings, probs)
    return Voters(first, d, rankings, probs)


def _weights(ballot, rankings, alive):
    """W[r, c] = points ranking r gives c: weight(position of c among the candidates in
    alive (C,), number of them), 0 for the others."""
    weights = np.zeros(rankings.shape)
    remaining = [[c for c in ranking if alive[c]] for ranking in rankings]
    for r, order in enumerate(remaining):
        for k, c in enumerate(order):
            weights[r, c] = ballot.weight(k, len(order))
    return weights


def _reference(ballot, how, rankings, probs, until=1):
    """Eliminate(Tally(ballot), how, until) point by point, straight from its
    definition: the candidates left (..., C) and the margin."""
    n = rankings.shape[1]
    weights = {}  # by the remaining candidates
    left = np.empty(probs.shape[:-1] + (n,), dtype=bool)
    margin = np.empty(probs.shape[:-1])
    for point in np.ndindex(*probs.shape[:-1]):
        alive, gaps = np.ones(n, dtype=bool), [np.inf]
        while alive.sum() > until:
            key = alive.tobytes()
            if key not in weights:
                weights[key] = _weights(ballot, rankings, alive)
            remaining = np.flatnonzero(alive)
            totals = (probs[point].astype(np.float64) @ weights[key])[remaining]
            highest = np.argsort(-totals, kind="stable")  # ties: the lowest index first
            if how == "all":
                gaps.append(totals[highest[until - 1]] - totals[highest[until]])
                alive[remaining[highest[until:]]] = False
            elif how == "min":
                order = np.argsort(totals, kind="stable")  # ties: the lowest index goes
                gaps.append(totals[order[1]] - totals[order[0]])
                alive[remaining[order[0]]] = False
            else:
                mean = totals.mean()
                gaps.append(np.abs(totals - mean).min())
                out = (totals <= mean) | (totals == totals.min())  # min: in case of rounding
                if out.all():  # all totals equal: the lowest index stays
                    out[0] = False
                if (~out).sum() < until:  # too few would stay: the `until` highest do
                    gaps.append(totals[highest[until - 1]] - totals[highest[until]])
                    out[:] = True
                    out[highest[:until]] = False
                alive[remaining[out]] = False
        left[point], margin[point] = alive, min(gaps)
    return left, margin


def _condorcet_margin(d):
    """The Condorcet winner (CYCLE if none) and margin of the former
    margin.methods.condorcet_margin: worst[c] = min_e (d[c, e] - d[e, c]) is c's narrowest
    head-to-head result, the winner has the best, if it is positive, and the margin is
    its absolute value."""
    n = d.shape[-1]
    lead = d - np.swapaxes(d, -1, -2)
    worst = np.where(np.eye(n, dtype=bool), np.inf, lead).min(axis=-1)
    best = worst.max(axis=-1)
    return np.where(best > 0, worst.argmax(axis=-1), CYCLE), np.abs(best)


def _schulze_margin(d):
    """The former margin.methods.schulze_margin: the Condorcet winner and margin where
    there is one, elsewhere the widest paths of d - d^T."""
    winner, margin = _condorcet_margin(d)
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


def _challengers(first, d):
    """How many candidates beat the one with the most first choices head to head, (...)."""
    king = first.argmax(axis=-1)[..., None, None]
    lead = d - np.swapaxes(d, -1, -2)
    return (np.take_along_axis(lead, king, axis=-1) > 0).sum(axis=(-1, -2))


def _first_choices(n, count=500, seed=0):
    """Random first-choice shares (count, n) to go with _tournaments(n), every tenth
    point on a grid of quarters, where they tie. They do not sum to 1: king of the hill
    only compares them."""
    first = np.random.default_rng(seed + n).random((count, n))
    first[::10] = np.round(4 * first[::10]) / 4
    return first


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


@pytest.mark.parametrize("name", ["fptp", "irv", "borda", "schulze", "condorcet"])
def test_methods_match_pixel_methods(profile, name):
    rankings, probs = profile
    winner, margin = METHODS[name].evaluate(_voters(rankings, probs))
    assert (margin >= 0).all()
    clear = margin > TOLERANCE  # off the borders
    assert clear.mean() > 0.5  # not vacuous; near-empty candidates tie, e.g. in IRV
    np.testing.assert_array_equal(winner[clear], pixel_methods.METHODS[name](rankings, probs)[clear])


@pytest.mark.parametrize("name", ["schulze", "condorcet"])
def test_condorcet_methods_are_the_former_functions(profile, name):
    """Exactly: the same winners (CYCLE included) and the same margins, bit for bit."""
    rankings, probs = profile
    d = _voters(rankings, probs).pairwise
    expected = {"schulze": _schulze_margin, "condorcet": _condorcet_margin}[name](d)
    winner, margin = METHODS[name].evaluate(Voters(pairwise=d))
    np.testing.assert_array_equal(winner, expected[0])
    np.testing.assert_array_equal(margin, expected[1])


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
@pytest.mark.parametrize("name", ["schulze", "condorcet"])
def test_condorcet_methods_are_the_former_functions_on_random_shares(d, name):
    """Random pairwise shares: many cycles, and ties on the quarters."""
    expected = {"schulze": _schulze_margin, "condorcet": _condorcet_margin}[name](d)
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
def test_minimax_matches_reference_on_random_shares(d):
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    winner, margin = METHODS["minimax"].evaluate(Voters(pairwise=d))
    expected_winner, expected_margin = _minimax_reference(d)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)


def test_minimax_matches_reference(profile):
    rankings, probs = profile
    voters = _voters(rankings, probs)
    expected_winner, expected_margin = _minimax_reference(voters.pairwise)
    winner, margin = METHODS["minimax"].evaluate(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected_winner[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)


def _condorcet_else_borda(d):
    """The Condorcet winner (beats everyone head to head), else the highest Borda score
    sum_e d[c, e]; ties: the lowest index."""
    wins = (d > np.swapaxes(d, -1, -2)).sum(axis=-1)
    return np.where(wins.max(axis=-1) == d.shape[-1] - 1, wins.argmax(axis=-1), d.sum(axis=-1).argmax(axis=-1))


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_black_is_condorcet_else_borda_on_random_shares(d):
    winner, _ = METHODS["black"].evaluate(Voters(pairwise=d))
    np.testing.assert_array_equal(winner, _condorcet_else_borda(d))


def test_black_is_condorcet_else_borda(profile):
    """The winners of pixels.methods, and the margin of the Condorcet winner, where there
    is one; in a cycle the smaller of that and the Borda margin."""
    rankings, probs = profile
    voters = _voters(rankings, probs)
    winner, margin = METHODS["black"].evaluate(voters)
    condorcet = pixel_methods.condorcet(rankings, probs)
    expected = np.where(condorcet == CYCLE, pixel_methods.borda(rankings, probs), condorcet)
    clear = margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected[clear])

    condorcet_winner, margin_condorcet = METHODS["condorcet"].evaluate(voters)
    _, margin_borda = METHODS["borda"].evaluate(voters)
    cycle = condorcet_winner == CYCLE
    np.testing.assert_array_equal(margin, np.where(cycle, np.minimum(margin_condorcet, margin_borda), margin_condorcet))
    assert (winner != CYCLE).all()


def test_koth_matches_reference(profile):
    rankings, probs = profile
    voters = _voters(rankings, probs)
    expected_winner, expected_margin = _koth_reference(voters.first, voters.pairwise)
    winner, margin = METHODS["koth"].evaluate(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected_winner[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_koth_matches_reference_on_random_shares(d):
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


def test_koth_is_fptp_where_no_one_beats_its_winner(profile):
    """And another candidate where someone does; the margin is at most that of fptp."""
    rankings, probs = profile
    voters = _voters(rankings, probs)
    winner, margin = METHODS["koth"].evaluate(voters)
    fptp_winner, fptp_margin = METHODS["fptp"].evaluate(voters)
    unbeaten = _challengers(voters.first, voters.pairwise) == 0
    assert unbeaten.any() and not unbeaten.all()  # not vacuous
    np.testing.assert_array_equal(winner[unbeaten], fptp_winner[unbeaten])
    assert (winner[~unbeaten] != fptp_winner[~unbeaten]).all()
    assert (margin <= fptp_margin).all()


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


def test_king_runoff_matches_reference(profile):
    """The built koth and irv as the finalists, their duel from its rule."""
    rankings, probs = profile
    voters = _voters(rankings, probs)
    koth, irv = METHODS["koth"].evaluate(voters), METHODS["irv"].evaluate(voters)
    d = voters.pairwise.astype(np.float64)
    expected_winner, expected_margin = _runoff_reference(d - np.swapaxes(d, -1, -2), koth, irv)
    winner, margin = METHODS["king_runoff"].evaluate(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected_winner[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)
    assert ((winner == koth[0]) | (winner == irv[0]) | (margin == 0)).all()  # a draw only on a border


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
@pytest.mark.parametrize("method", RUNOFFS, ids=repr)
def test_runoff_matches_reference_on_random_shares(d, method):
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
def test_two_round_matches_reference_on_random_shares(d):
    """The two most first choices as the finalists, head to head; a tie is a draw."""
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    first = _first_choices(d.shape[-1], len(d))
    expected_winner, expected_margin = _runoff_reference(d - np.swapaxes(d, -1, -2), *_finalists(first))
    winner, margin = METHODS["two_round"].evaluate(Voters(first, d))
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)
    assert (margin[winner == CYCLE] == 0).all()  # a draw only on a border
    assert (margin == 0).any() and (winner != CYCLE).any()  # not vacuous


def test_two_round_matches_reference(profile):
    rankings, probs = profile
    voters = _voters(rankings, probs)
    d = voters.pairwise.astype(np.float64)
    expected_winner, expected_margin = _runoff_reference(d - np.swapaxes(d, -1, -2), *_finalists(voters.first))
    winner, margin = METHODS["two_round"].evaluate(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(winner[clear], expected_winner[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)
    fptp = METHODS["fptp"].evaluate(voters)[0]
    assert (winner[clear] != fptp[clear]).any()  # the second round changes something


def test_star_uses_the_two_highest_score_totals_and_abstaining_score_comparisons():
    """At the first point the duel ties: a draw, with margin 0."""
    ballot = Score(6, power=1.5)
    scored = ballot.scored
    score_gaps = np.array([[0.0, 0.2, 0.8], [0.0, 0.2, 0.7], [0.0, 0.0, 0.8]])
    score_diffs = np.zeros((3, 3, 3))
    score_diffs[1:, 0, 1], score_diffs[1:, 1, 0] = -0.4, 0.4
    voters = Voters(
        unscored={scored: score_gaps},
        scored_pairwise={ScoredPairwise(scored): score_diffs},
    )

    method = METHODS["star"]
    winner, margin = method.evaluate(voters)
    np.testing.assert_array_equal(winner, [CYCLE, 1, 1])
    np.testing.assert_array_equal(margin, [0.0, 0.4, 0.4])
    assert PAIRWISE not in method.needs
    assert method.needs == {scored, ScoredPairwise(scored)}
    comparisons = ScoreComparisons(ballot).evaluate(voters)
    np.testing.assert_array_equal(comparisons[0], 0.0)  # tied ratings abstain


def test_star_matches_reference_on_random_scores():
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


def test_highest_is_eliminate_of_all_but_the_highest():
    """Highest(totals, n) is a block of its own name, and does what Eliminate(totals,
    how="all", until=n) does. It takes any totals, which it reads once."""
    voters = _scoring_voters()
    totals = Tally(Score(6))
    for n in (1, 2, 3):
        highest, eliminate = Highest(totals, n=n), Eliminate(totals, how="all", until=n)
        assert isinstance(highest, Eliminate) and highest != eliminate
        assert highest.n == highest.until == highest.seats == n and highest.how == "all"
        for result, expected in zip(highest.select(voters), eliminate.select(voters)):
            np.testing.assert_array_equal(result, expected)
    assert Highest(totals) == Highest(totals, 1) == Highest(totals=totals, n=1) != Highest(totals, n=2)
    assert Eliminate(Weakest(Margins(Pairwise())), how="all").needs == {PAIRWISE}
    assert Highest(Tally(Plurality()), n=2).needs == {FIRST}  # the first choices, no profile
    with pytest.raises(FrozenInstanceError):
        METHODS["fptp"].until = 2


def test_the_n_highest_totals_are_chosen():
    """The n highest totals (ties: the lowest index) with the n-th highest minus the
    next as the margin; with no more than n candidates all of them, margin inf."""
    voters = _scoring_voters()
    shares = voters.unscored[Scored(6)]
    order = np.argsort(shares, axis=-1, kind="stable")  # the fewest not given first
    ranked = np.take_along_axis(shares, order, axis=-1)
    for n in (1, 2, 4):
        chosen, margin = Highest(Tally(Score(6)), n=n).select(voters)
        expected = np.zeros(shares.shape, dtype=bool)
        np.put_along_axis(expected, order[..., :n], True, axis=-1)
        np.testing.assert_array_equal(chosen, expected)
        np.testing.assert_array_equal(margin, ranked[..., n] - ranked[..., n - 1])
        assert (margin == 0).any() and (margin > 0).any()  # the ties are there
    chosen, margin = Highest(Tally(Score(6)), n=5).select(voters)
    assert chosen.all() and (margin == np.inf).all()


def test_a_method_chooses_one_candidate():
    """Blocks that choose several candidates only select(); evaluate() is for a winner.
    A Fallback of two such blocks chooses as many, a Union a number that varies."""
    two, irv = Highest(Tally(Plurality()), n=2), METHODS["irv"]
    assert [block.seats for block in (two, irv, two | irv, Fallback(two, two), Fallback(irv, two))] \
        == [2, 1, None, 2, None]
    assert all(method.seats == 1 for method in METHODS.values())
    for block, chooses in ((two, "2 candidates"), (two | irv, "a number of candidates"),
                           (Eliminate(Tally(Plurality()), how="min", until=3), "3 candidates")):
        with pytest.raises(ValueError, match=re.escape(f"{block!r} chooses {chooses}, not a winner")):
            block.evaluate(Voters(first=np.full((2, 4), 0.25)))


def test_union_and_fallback_choose_sets(profile):
    """first | second is either one's candidates, with the smaller margin; a Fallback of
    sets takes second's where first chooses no one."""
    voters = _voters(*profile)
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


@pytest.mark.parametrize("method", ELIMINATIONS, ids=repr)
def test_eliminations_match_reference(profile, method):
    """Both the method (for irv the compiled voting.irv_rounds) and its rounds with
    one tally each."""
    rankings, probs = profile
    voters = _voters(rankings, probs)
    left, expected_margin = _reference(method.totals.ballot, method.how, rankings, probs)
    expected_winner = left.argmax(axis=-1)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous; near-empty candidates tie, e.g. in IRV
    for winner, margin in (method.evaluate(voters), method.rounds(voters)):
        np.testing.assert_array_equal(winner[clear], expected_winner[clear])
        np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)


@pytest.mark.parametrize("method", FINALISTS, ids=repr)
def test_eliminations_down_to_two_match_reference(profile, method):
    """The two candidates left, from the rounds of each and from Highest's one."""
    rankings, probs = profile
    voters = _voters(rankings, probs)
    expected, expected_margin = _reference(method.totals.ballot, method.how, rankings, probs, method.until)
    chosen, margin = method.select(voters)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous
    np.testing.assert_array_equal(chosen[clear], expected[clear])
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)
    assert (chosen.sum(axis=-1) == 2).all()


@pytest.mark.parametrize("ballot", BALLOTS, ids=repr)
def test_tally_is_the_sum_of_weights(profile, ballot):
    """For all candidates and for random remaining ones at every point (at least one)."""
    rankings, probs = profile
    voters, n = _voters(rankings, probs), rankings.shape[1]
    everyone = probs @ _weights(ballot, rankings, np.ones(n, dtype=bool))
    np.testing.assert_allclose(ballot.tally(voters, None), everyone, rtol=0, atol=1e-5)

    rng = np.random.default_rng(n)
    alive = rng.random(probs.shape[:-1] + (n,)) < 0.5
    alive |= np.arange(n) == rng.integers(n, size=probs.shape[:-1] + (1,))
    codes = alive @ (1 << np.arange(n))
    expected = np.empty(alive.shape)
    for code in np.unique(codes):
        at = codes == code
        expected[at] = probs[at] @ _weights(ballot, rankings, (code >> np.arange(n)) & 1)
    np.testing.assert_allclose(ballot.tally(voters, alive), expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(Tally(ballot).evaluate(voters, alive), ballot.tally(voters, alive))


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


@pytest.mark.parametrize("share", [0.25, 0.5, 0.9])
def test_a_mix_tallies_both_kinds_of_voters(share):
    """Mix(first, second, share): `share` of the voters mark second and the others first,
    so every total is that mix of the two, among any remaining candidates, and the
    candidate the fewest of all voters do not approve wins."""
    voters, ballot = _scoring_voters(), Mix(BY_GAP, BY_MEAN, share=share)
    unscored = (1 - share) * voters.unscored[BY_GAP.scored] + share * voters.unscored[BY_MEAN.scored]
    np.testing.assert_allclose(ballot.tally(voters, None), -unscored, rtol=1e-12, atol=0)
    alive = np.random.default_rng(1).random(unscored.shape) < 0.5
    np.testing.assert_allclose(ballot.tally(voters, alive), np.where(alive, -unscored, -1),
                               rtol=1e-12, atol=0)
    winner, margin = Highest(Tally(ballot)).evaluate(voters)
    ordered = np.sort(unscored, axis=-1)
    clear = ordered[..., 1] - ordered[..., 0] > 1e-12
    np.testing.assert_array_equal(winner[clear], unscored.argmin(axis=-1)[clear])
    np.testing.assert_allclose(margin, ordered[..., 1] - ordered[..., 0], rtol=0, atol=1e-12)


def test_a_mix_all_voters_mark_one_ballot_of_is_that_ballot():
    """At share 0 all voters mark the first ballot and at 1 the second: the same totals,
    bit for bit, from the shares of that ballot alone."""
    voters = _scoring_voters()
    alive = np.random.default_rng(1).random((4, 6, 5)) < 0.5
    for share, ballot in ((0, BY_GAP), (1, BY_MEAN), (1.0, BY_MEAN)):
        mix = Mix(BY_GAP, BY_MEAN, share=share)
        assert mix.needs == mix.needs_remaining == {ballot.scored}
        only = Voters(unscored={ballot.scored: voters.unscored[ballot.scored]})  # the other is not read
        for remaining in (None, alive):
            np.testing.assert_array_equal(mix.tally(only, remaining), ballot.tally(voters, remaining))
        for result, expected in zip(Highest(Tally(mix)).evaluate(only),
                                    Highest(Tally(ballot)).evaluate(voters)):
            np.testing.assert_array_equal(result, expected)


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
    voters = _voters(*profile)
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


@pytest.mark.parametrize("share", [-0.1, 1.5, float("nan")])
def test_a_mix_share_is_a_share(share):
    with pytest.raises(ValueError, match=re.escape(f"Mix share must be between 0 and 1, got {share!r}")):
        Mix(BY_GAP, BY_MEAN, share=share)


def test_score_takes_the_number_of_levels():
    assert Score(3) == Score(levels=3) != Score(4)
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
    """All pairwise shares 1/2: every total ties. The highest and Nanson's survivor are
    the first candidate; the lowest, which Baldwin drops each round, too, so the last
    one wins. Every margin is 0."""
    n = 4
    d = np.full((3, n, n), 0.5)
    d[:, np.arange(n), np.arange(n)] = 0
    voters = Voters(pairwise=d)
    for name, expected in (("borda", 0), ("nanson", 0), ("baldwin", n - 1)):
        winner, margin = METHODS[name].evaluate(voters)
        np.testing.assert_array_equal(winner, expected, err_msg=name)
        np.testing.assert_array_equal(margin, 0.0, err_msg=name)


def test_condorcet_ties():
    """All pairwise shares 1/2: no one beats anyone. There is no Condorcet winner, and
    Borda, which Black falls back to, elects the first candidate; everyone else ties, and
    the first wins. With equal first choices the first candidate is the king of the hill,
    and no one challenges it. Every margin is 0."""
    n = 4
    d = np.full((3, n, n), 0.5)
    d[:, np.arange(n), np.arange(n)] = 0
    voters = Voters(first=np.full((3, n), 1 / n), pairwise=d)
    for name, expected in (("condorcet", CYCLE), ("schulze", 0), ("minimax", 0), ("black", 0),
                           ("koth", 0)):
        winner, margin = METHODS[name].evaluate(voters)
        np.testing.assert_array_equal(winner, expected, err_msg=name)
        np.testing.assert_array_equal(margin, 0.0, err_msg=name)


def test_methods_are_the_fourteen_expressions():
    pairwise = Pairwise()
    margins = Margins(pairwise)
    koth = Unbeaten(margins, against=Highest(Tally(Plurality())), order=Tally(Plurality()))
    score = Highest(Tally(Score(6, power=1.5)))
    assert METHODS == {
        "fptp": Highest(Tally(Plurality())),
        "two_round": Unbeaten(margins, among=Highest(Tally(Plurality()), n=2)),
        "irv": Eliminate(Tally(Plurality()), how="min"),
        "borda": Highest(Tally(BordaCount())),
        "baldwin": Eliminate(Tally(BordaCount()), how="min"),
        "nanson": Eliminate(Tally(BordaCount()), how="mean"),
        "schulze": Unbeaten(StrongestPaths(margins)),
        "condorcet": Unbeaten(margins),
        "minimax": Highest(Weakest(margins)),
        "black": Fallback(Unbeaten(margins), Highest(Tally(BordaCount()))),
        "koth": Unbeaten(margins, against=Highest(Tally(Plurality())), order=Tally(Plurality())),
        "king_runoff": Unbeaten(margins, among=Union(koth, Eliminate(Tally(Plurality()), how="min"))),
        "score": score,
        "star": Unbeaten(ScoreComparisons(Score(6, power=1.5)), among=Highest(Tally(Score(6, power=1.5)), n=2)),
    }
    assert list(METHODS) == ["fptp", "two_round", "irv", "borda", "baldwin", "nanson", "schulze",
                             "condorcet", "minimax", "black", "koth", "king_runoff", "score", "star"]


def test_repr_is_the_expression():
    assert repr(METHODS["nanson"]) == 'Eliminate(Tally(BordaCount()), how="mean")'
    assert repr(METHODS["schulze"]) == "Unbeaten(StrongestPaths(Margins(Pairwise())))"
    assert repr(METHODS["condorcet"]) == "Unbeaten(Margins(Pairwise()))"
    assert repr(METHODS["minimax"]) == "Highest(Weakest(Margins(Pairwise())))"
    assert repr(METHODS["black"]) == (
        "Fallback(Unbeaten(Margins(Pairwise())), Highest(Tally(BordaCount())))")
    assert repr(METHODS["fptp"]) == "Highest(Tally(Plurality()))"
    assert repr(METHODS["koth"]) == (
        "Unbeaten(Margins(Pairwise()), against=Highest(Tally(Plurality())), order=Tally(Plurality()))")
    koth = "Unbeaten(Margins(Pairwise()), against=Highest(Tally(Plurality())), order=Tally(Plurality()))"
    assert repr(METHODS["king_runoff"]) == (
        f'Unbeaten(Margins(Pairwise()), among={koth} | Eliminate(Tally(Plurality()), how="min"))')
    assert repr(METHODS["two_round"]) == (
        "Unbeaten(Margins(Pairwise()), among=Highest(Tally(Plurality()), n=2))")
    assert repr(METHODS["score"]) == "Highest(Tally(Score(6, power=1.5)))"
    assert repr(METHODS["star"]) == (
        "Unbeaten(ScoreComparisons(Score(6, power=1.5)), among=Highest(Tally(Score(6, power=1.5)), n=2))")
    assert repr(Eliminate(Tally(BordaCount()), how="mean", until=2)) == 'Eliminate(Tally(BordaCount()), how="mean", until=2)'
    fptp, irv, borda = METHODS["fptp"], METHODS["irv"], METHODS["borda"]
    assert repr(fptp | (irv | borda)) == f"{fptp!r} | ({irv!r} | {borda!r})"
    challenged =Unbeaten(StrongestPaths(Margins(Pairwise())), against=METHODS["black"],
                          order=Weakest(Margins(Pairwise())))
    unions = [fptp | irv | borda, fptp | (irv | borda), Fallback(Highest(Tally(BY_MEAN), n=3), fptp | irv)]
    for method in [*METHODS.values(), *ELIMINATIONS, *FINALISTS, *BALLOTS, *PAIR_BLOCKS, challenged, *RUNOFFS,
                   *unions]:
        assert eval(repr(method)) == method


def test_unbeaten_takes_against_and_order_together():
    margins = Margins(Pairwise())
    for build in (lambda: Unbeaten(margins, against=METHODS["fptp"]),
                  lambda: Unbeaten(margins, order=Tally(Plurality()))):
        with pytest.raises(ValueError, match=re.escape("Unbeaten takes against and order together, or neither")):
            build()
    assert Unbeaten(margins, against=None, order=None) == METHODS["condorcet"]


def test_blocks_are_frozen_and_hashable():
    assert len(set(METHODS.values())) == len(METHODS)
    cache = {Eliminate(Tally(BordaCount()), how="mean"): "nanson"}
    assert cache[METHODS["nanson"]] == "nanson"
    with pytest.raises(FrozenInstanceError):
        METHODS["irv"].how = "mean"


def test_needs():
    assert METHODS["fptp"].needs == {FIRST}
    assert METHODS["irv"].needs == {PROFILE}
    for name in ("borda", "baldwin", "nanson", "schulze", "condorcet", "minimax", "black"):
        assert METHODS[name].needs == {PAIRWISE}, name
    assert METHODS["koth"].needs == {FIRST, PAIRWISE}
    assert METHODS["king_runoff"].needs == {FIRST, PAIRWISE, PROFILE}
    assert _runoff(Margins(Pairwise()), METHODS["borda"], METHODS["minimax"]).needs == {PAIRWISE}
    assert METHODS["two_round"].needs == {FIRST, PAIRWISE}
    assert Eliminate(Tally(Plurality()), how="min", until=2).needs == {PROFILE}
    assert (METHODS["fptp"] | METHODS["irv"]).needs == {FIRST, PROFILE}
    assert Eliminate(Tally(Plurality()), how="mean").needs == {PROFILE}
    assert all(block.needs == {PAIRWISE} for block in PAIR_BLOCKS)
    assert Highest(Tally(Mix(BY_GAP, BY_MEAN, share=0.5))).needs == {Scored(2, "dhondt"), Scored(2, "avg")}
    assert METHODS["score"].needs == {Scored(6, power=1.5)} == {Scored(6, "range", power=1.5)}
    assert Highest(Tally(Score(6))).needs == {Scored(6)} == {Scored(6, "range")}
    assert Highest(Tally(ScoreAvg(6))).needs == {Scored(6, "avg")}
    assert Highest(Tally(ScoreDH(6))).needs == {Scored(6, "dhondt")}
    assert Highest(Tally(ScoreHybrid(6))).needs == {Scored(6, "hybrid")}
    assert Highest(Tally(ScoreCluster(6))).needs == {Scored(6, "cluster")}
    assert Eliminate(Tally(BY_GAP), how="min").needs == {Scored(2, "dhondt")}
    assert Fallback(METHODS["condorcet"], Highest(Tally(BY_MEAN))).needs == {PAIRWISE, Scored(2, "avg")}


def test_fallback_needs_the_shares_of_both():
    assert Fallback(METHODS["fptp"], METHODS["schulze"]).needs == {FIRST, PAIRWISE}
    assert Fallback(METHODS["irv"], METHODS["fptp"]).needs == {PROFILE, FIRST}
    assert Fallback(METHODS["condorcet"], METHODS["borda"]).needs == {PAIRWISE}


def test_unbeaten_against_needs_the_shares_of_all_three():
    margins = Margins(Pairwise())
    assert Unbeaten(margins, against=METHODS["irv"], order=Tally(Plurality())).needs == {PAIRWISE, PROFILE, FIRST}
    assert Unbeaten(margins, against=METHODS["borda"], order=Weakest(margins)).needs == {PAIRWISE}


@pytest.mark.parametrize("build, message", [
    (lambda: Highest(BordaCount()), "Highest expects CandidateTotals, got a Ballot: BordaCount()"),
    (lambda: Highest(METHODS["schulze"]),
     "Highest expects CandidateTotals, got a Winner: Unbeaten(StrongestPaths(Margins(Pairwise())))"),
    (lambda: Highest(Margins(Pairwise())),
     "Highest expects CandidateTotals, got PairDiffs: Margins(Pairwise())"),
    (lambda: Highest(Pairwise()), "Highest expects CandidateTotals, got PairShares: Pairwise()"),
    (lambda: Tally(Tally(Plurality())), "Tally expects a Ballot, got CandidateTotals: Tally(Plurality())"),
    (lambda: Tally(METHODS["condorcet"]),
     "Tally expects a Ballot, got a Winner: Unbeaten(Margins(Pairwise()))"),
    (lambda: Tally(Margins(Pairwise())), "Tally expects a Ballot, got PairDiffs: Margins(Pairwise())"),
    (lambda: Tally(Plurality), "Tally expects a Ballot, got the class Plurality; call it: Plurality()"),
    (lambda: Tally(3), "Tally expects a Ballot, got int 3"),
    (lambda: Mix(Tally(Plurality()), Score(2), share=0.5),
     "Mix expects a Ballot, got CandidateTotals: Tally(Plurality())"),
    (lambda: Mix(Score(2), BordaCount, share=0.5),
     "Mix expects a Ballot, got the class BordaCount; call it: BordaCount()"),
    (lambda: Eliminate(BordaCount(), how="min"), "Eliminate expects a Tally"),
    (lambda: Eliminate(Highest(Tally(Plurality())), how="min"),
     "Eliminate expects a Tally (it tallies the remaining candidates again), got a Winner"),
    (lambda: Eliminate(Weakest(Margins(Pairwise())), how="min"),
     "Eliminate expects a Tally (it tallies the remaining candidates again), got CandidateTotals"),
    (lambda: Margins(Tally(BordaCount())),
     "Margins expects PairShares, got CandidateTotals: Tally(BordaCount())"),
    (lambda: Margins(Margins(Pairwise())), "Margins expects PairShares, got PairDiffs: Margins(Pairwise())"),
    (lambda: Margins(Pairwise), "Margins expects PairShares, got the class Pairwise; call it: Pairwise()"),
    (lambda: StrongestPaths(Pairwise()), "StrongestPaths expects PairDiffs, got PairShares: Pairwise()"),
    (lambda: StrongestPaths(BordaCount()), "StrongestPaths expects PairDiffs, got a Ballot: BordaCount()"),
    (lambda: Weakest(Pairwise()), "Weakest expects PairDiffs, got PairShares: Pairwise()"),
    (lambda: Weakest(Highest(Weakest(Margins(Pairwise())))),
     "Weakest expects PairDiffs, got a Winner: Highest(Weakest(Margins(Pairwise())))"),
    (lambda: Unbeaten(Tally(BordaCount())),
     "Unbeaten expects PairDiffs, got CandidateTotals: Tally(BordaCount())"),
    (lambda: Unbeaten(Pairwise()), "Unbeaten expects PairDiffs, got PairShares: Pairwise()"),
    (lambda: Unbeaten(METHODS["borda"]),
     "Unbeaten expects PairDiffs, got a Winner: Highest(Tally(BordaCount()))"),
    (lambda: Unbeaten(Margins(Pairwise()), against=Tally(Plurality()), order=Tally(Plurality())),
     "Unbeaten expects a Winner as against, got CandidateTotals: Tally(Plurality())"),
    (lambda: Unbeaten(Margins(Pairwise()), against=METHODS["fptp"], order=Plurality()),
     "Unbeaten expects CandidateTotals as order, got a Ballot: Plurality()"),
    (lambda: Fallback(BordaCount(), METHODS["borda"]), "Fallback expects a Winner, got a Ballot: BordaCount()"),
    (lambda: Fallback(METHODS["borda"], Weakest(Margins(Pairwise()))),
     "Fallback expects a Winner, got CandidateTotals: Weakest(Margins(Pairwise()))"),
    (lambda: METHODS["fptp"] | Tally(Plurality()), "Union expects a Winner, got CandidateTotals: Tally(Plurality())"),
    (lambda: Union(BordaCount(), METHODS["borda"]), "Union expects a Winner, got a Ballot: BordaCount()"),
    (lambda: Unbeaten(Margins(Pairwise()), among=Tally(Plurality())),
     "Unbeaten expects a Winner as among, got CandidateTotals: Tally(Plurality())"),
    (lambda: Highest(BordaCount(), n=2), "Highest expects CandidateTotals, got a Ballot: BordaCount()"),
    (lambda: Eliminate(BordaCount(), how="all"), "Eliminate expects CandidateTotals, got a Ballot"),
])
def test_wrong_blocks_fail_when_built(build, message):
    with pytest.raises(TypeError, match=re.escape(message)):
        build()


def test_unknown_how_fails():
    with pytest.raises(ValueError, match=re.escape('Eliminate how must be "min", "mean" or "all", got \'max\'')):
        Eliminate(Tally(BordaCount()), how="max")


@pytest.mark.parametrize("until", [0, -1, 1.5, True, "2"])
def test_until_and_n_are_whole_numbers(until):
    with pytest.raises(ValueError, match=re.escape(f"Eliminate until must be a whole number of at least 1, got {until!r}")):
        Eliminate(Tally(BordaCount()), how="min", until=until)
    with pytest.raises(ValueError, match=re.escape(f"Highest n must be a whole number of at least 1, got {until!r}")):
        Highest(Tally(BordaCount()), n=until)


def test_unbeaten_takes_among_or_a_king():
    margins = Margins(Pairwise())
    with pytest.raises(ValueError, match=re.escape("Unbeaten takes among or against, not both")):
        Unbeaten(margins, against=METHODS["fptp"], order=Tally(Plurality()), among=METHODS["irv"])
    with pytest.raises(ValueError, match=re.escape("Unbeaten against must choose one candidate, the king")):
        Unbeaten(margins, against=Highest(Tally(Plurality()), n=2), order=Tally(Plurality()))
