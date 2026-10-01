"""Checks of the methods built from blocks (yeelab.build).

On the complete profile of every pixel (pixels/), the built fptp, irv, borda, schulze
and condorcet must pick the winners of pixels.methods, and every elimination the
winners and margins of a plain Python reference that follows the definition: scores
from weight(k, C) among the remaining candidates, one point at a time. The fast tally
of every ballot must be that same sum over the profile, for any remaining candidates.

schulze and condorcet must give exactly the winners and margins that the
functions of the former margin/methods.py gave (copied below as the reference), on the
profiles and on random pairwise shares with many cycles and ties. minimax is checked
against a Python reference of its definition, black against "the Condorcet winner,
else Borda". koth (king of the hill) is checked against a Python reference of its rule,
winner and margin, on the profiles and on random first-choice and pairwise shares.
Runoff is checked against a Python reference of its duel between the built finalists:
king_runoff on the profiles, other finalists (with cycles and ties) on random shares.

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
    Pairwise,
    Plurality,
    Runoff,
    StrongestPaths,
    Tally,
    Unbeaten,
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
LINK_BLOCKS = [Pairwise(), Margins(Pairwise()), StrongestPaths(Margins(Pairwise())),
               StrongestPaths(StrongestPaths(Margins(Pairwise()))), Weakest(Margins(Pairwise()))]
# every elimination: the four methods and the one other combination of these blocks
ELIMINATIONS = [METHODS["irv"], METHODS["baldwin"], METHODS["nanson"],
                Eliminate(Tally(Plurality()), how="mean")]


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


def _reference(ballot, how, rankings, probs):
    """Eliminate(Tally(ballot), how) point by point, straight from its definition."""
    n = rankings.shape[1]
    weights = {}  # by the remaining candidates
    winner = np.empty(probs.shape[:-1], dtype=np.int64)
    margin = np.empty(probs.shape[:-1])
    for point in np.ndindex(*probs.shape[:-1]):
        alive, gaps = np.ones(n, dtype=bool), []
        while alive.sum() > 1:
            key = alive.tobytes()
            if key not in weights:
                weights[key] = _weights(ballot, rankings, alive)
            remaining = np.flatnonzero(alive)
            scores = (probs[point].astype(np.float64) @ weights[key])[remaining]
            if how == "min":
                order = np.argsort(scores, kind="stable")  # ties: the lowest index goes
                gaps.append(scores[order[1]] - scores[order[0]])
                alive[remaining[order[0]]] = False
            else:
                mean = scores.mean()
                gaps.append(np.abs(scores - mean).min())
                out = (scores <= mean) | (scores == scores.min())  # min: in case of rounding
                if out.all():  # all scores equal: the lowest index stays
                    out[0] = False
                alive[remaining[out]] = False
        winner[point], margin[point] = np.flatnonzero(alive)[0], min(gaps)
    return winner, margin


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


def _runoff_reference(links, first, second):
    """Runoff point by point from its rule, for the links s (..., C, C) and the (winner,
    margin) of both finalists: the second wins where s[second][first] > 0, otherwise the
    first, and a point where either is CYCLE stays one. The margin is the smallest of the
    finalists' margins and, where they are two candidates, |s[first][second]|."""
    (a, margin_a), (b, margin_b) = first, second
    winner = np.empty(a.shape, dtype=np.int64)
    margin = np.empty(a.shape)
    for point in np.ndindex(*a.shape):
        x, y, s = int(a[point]), int(b[point]), links[point].tolist()
        gaps = [float(margin_a[point]), float(margin_b[point])]
        if CYCLE in (x, y):
            winner[point] = CYCLE
        else:
            winner[point] = y if s[y][x] > 0 else x
            if x != y:
                gaps.append(abs(s[x][y]))
        margin[point] = min(gaps)
    return winner, margin


# finalists that agree, differ and tie, one that elects no one in a cycle (condorcet),
# and a duel on the strongest paths
RUNOFFS = [
    Runoff(Margins(Pairwise()), METHODS["koth"], METHODS["borda"]),
    Runoff(Margins(Pairwise()), METHODS["condorcet"], METHODS["minimax"]),
    Runoff(Margins(Pairwise()), METHODS["fptp"], METHODS["condorcet"]),
    Runoff(StrongestPaths(Margins(Pairwise())), METHODS["borda"], METHODS["fptp"]),
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
    """Unbeaten(StrongestPaths(...)) takes the winner and margin of the direct links where
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
    assert ((winner == koth[0]) | (winner == irv[0])).all()


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
@pytest.mark.parametrize("method", RUNOFFS, ids=repr)
def test_runoff_matches_reference_on_random_shares(d, method):
    d = d.astype(np.float64)  # the reference computes in float64: the same arithmetic
    voters = Voters(_first_choices(d.shape[-1], len(d)), d)
    first, second = method.first.evaluate(voters), method.second.evaluate(voters)
    expected_winner, expected_margin = _runoff_reference(method.links.evaluate(voters), first, second)
    winner, margin = method.evaluate(voters)
    np.testing.assert_array_equal(winner, expected_winner)
    np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(winner == CYCLE, (first[0] == CYCLE) | (second[0] == CYCLE))


@pytest.mark.parametrize("d", TOURNAMENTS, ids=lambda d: f"{d.shape[-1]}-candidates")
def test_runoff_of_the_same_and_of_swapped_finalists(d):
    """A method against itself is that method, margin and cycles included. Swapping the
    finalists keeps the margin, and the winner wherever the duel is not a tie."""
    margins = Margins(Pairwise())
    voters = Voters(_first_choices(d.shape[-1], len(d)), d)
    fptp, borda = METHODS["fptp"], METHODS["borda"]
    for method in (fptp, borda, METHODS["condorcet"]):
        winner, margin = Runoff(margins, method, method).evaluate(voters)
        expected_winner, expected_margin = method.evaluate(voters)
        np.testing.assert_array_equal(winner, expected_winner)
        np.testing.assert_array_equal(margin, expected_margin)

    winner, margin = Runoff(margins, fptp, borda).evaluate(voters)
    swapped_winner, swapped_margin = Runoff(margins, borda, fptp).evaluate(voters)
    np.testing.assert_array_equal(margin, swapped_margin)
    np.testing.assert_array_equal(winner[margin > 0], swapped_winner[margin > 0])
    if d.shape[-1] > 2:  # not vacuous: the finalists differ, and each of them wins duels
        a, b = fptp.evaluate(voters)[0], borda.evaluate(voters)[0]
        assert ((winner == a) & (a != b)).any() and ((winner == b) & (a != b)).any()


@pytest.mark.parametrize("method", ELIMINATIONS, ids=repr)
def test_eliminations_match_reference(profile, method):
    """Both the method (for irv the compiled voting.irv_rounds) and its rounds with
    one tally each."""
    rankings, probs = profile
    voters = _voters(rankings, probs)
    expected_winner, expected_margin = _reference(method.scores.ballot, method.how, rankings, probs)
    clear = expected_margin > TOLERANCE
    assert clear.mean() > 0.5  # not vacuous; near-empty candidates tie, e.g. in IRV
    for winner, margin in (method.evaluate(voters), method.rounds(voters)):
        np.testing.assert_array_equal(winner[clear], expected_winner[clear])
        np.testing.assert_allclose(margin, expected_margin, rtol=0, atol=1e-5)


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


def test_ties_go_to_the_lowest_index():
    """All pairwise shares 1/2: every score ties. The highest and Nanson's survivor are
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


def test_methods_are_the_eleven_expressions():
    pairwise = Pairwise()
    margins = Margins(pairwise)
    assert METHODS == {
        "fptp": Highest(Tally(Plurality())),
        "irv": Eliminate(Tally(Plurality()), how="min"),
        "borda": Highest(Tally(BordaCount())),
        "baldwin": Eliminate(Tally(BordaCount()), how="min"),
        "nanson": Eliminate(Tally(BordaCount()), how="mean"),
        "schulze": Unbeaten(StrongestPaths(margins)),
        "condorcet": Unbeaten(margins),
        "minimax": Highest(Weakest(margins)),
        "black": Fallback(Unbeaten(margins), Highest(Tally(BordaCount()))),
        "koth": Unbeaten(margins, against=Highest(Tally(Plurality())), order=Tally(Plurality())),
        "king_runoff": Runoff(
            margins,
            Unbeaten(margins, against=Highest(Tally(Plurality())), order=Tally(Plurality())),
            Eliminate(Tally(Plurality()), how="min")),
    }
    assert list(METHODS) == ["fptp", "irv", "borda", "baldwin", "nanson", "schulze",
                             "condorcet", "minimax", "black", "koth", "king_runoff"]


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
    assert repr(METHODS["king_runoff"]) == (
        "Runoff(Margins(Pairwise()), "
        "Unbeaten(Margins(Pairwise()), against=Highest(Tally(Plurality())), order=Tally(Plurality())), "
        'Eliminate(Tally(Plurality()), how="min"))')
    challenged = Unbeaten(StrongestPaths(Margins(Pairwise())), against=METHODS["black"],
                          order=Weakest(Margins(Pairwise())))
    for method in [*METHODS.values(), *ELIMINATIONS, *BALLOTS, *LINK_BLOCKS, challenged, *RUNOFFS]:
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
    assert Runoff(Margins(Pairwise()), METHODS["borda"], METHODS["minimax"]).needs == {PAIRWISE}
    assert Eliminate(Tally(Plurality()), how="mean").needs == {PROFILE}
    assert all(block.needs == {PAIRWISE} for block in LINK_BLOCKS)


def test_fallback_needs_the_shares_of_both():
    assert Fallback(METHODS["fptp"], METHODS["schulze"]).needs == {FIRST, PAIRWISE}
    assert Fallback(METHODS["irv"], METHODS["fptp"]).needs == {PROFILE, FIRST}
    assert Fallback(METHODS["condorcet"], METHODS["borda"]).needs == {PAIRWISE}


def test_unbeaten_against_needs_the_shares_of_all_three():
    margins = Margins(Pairwise())
    assert Unbeaten(margins, against=METHODS["irv"], order=Tally(Plurality())).needs == {PAIRWISE, PROFILE, FIRST}
    assert Unbeaten(margins, against=METHODS["borda"], order=Weakest(margins)).needs == {PAIRWISE}


@pytest.mark.parametrize("build, message", [
    (lambda: Highest(BordaCount()), "Highest expects Scores, got a Ballot: BordaCount()"),
    (lambda: Highest(METHODS["schulze"]),
     "Highest expects Scores, got a Winner: Unbeaten(StrongestPaths(Margins(Pairwise())))"),
    (lambda: Highest(Margins(Pairwise())), "Highest expects Scores, got Links: Margins(Pairwise())"),
    (lambda: Highest(Pairwise()), "Highest expects Scores, got Duels: Pairwise()"),
    (lambda: Tally(Tally(Plurality())), "Tally expects a Ballot, got Scores: Tally(Plurality())"),
    (lambda: Tally(METHODS["condorcet"]),
     "Tally expects a Ballot, got a Winner: Unbeaten(Margins(Pairwise()))"),
    (lambda: Tally(Margins(Pairwise())), "Tally expects a Ballot, got Links: Margins(Pairwise())"),
    (lambda: Tally(Plurality), "Tally expects a Ballot, got the class Plurality; call it: Plurality()"),
    (lambda: Tally(3), "Tally expects a Ballot, got int 3"),
    (lambda: Eliminate(BordaCount(), how="min"), "Eliminate expects a Tally"),
    (lambda: Eliminate(Highest(Tally(Plurality())), how="min"),
     "Eliminate expects a Tally (it tallies the remaining candidates again), got a Winner"),
    (lambda: Eliminate(Weakest(Margins(Pairwise())), how="min"),
     "Eliminate expects a Tally (it tallies the remaining candidates again), got Scores"),
    (lambda: Margins(Tally(BordaCount())), "Margins expects Duels, got Scores: Tally(BordaCount())"),
    (lambda: Margins(Margins(Pairwise())), "Margins expects Duels, got Links: Margins(Pairwise())"),
    (lambda: Margins(Pairwise), "Margins expects Duels, got the class Pairwise; call it: Pairwise()"),
    (lambda: StrongestPaths(Pairwise()), "StrongestPaths expects Links, got Duels: Pairwise()"),
    (lambda: StrongestPaths(BordaCount()), "StrongestPaths expects Links, got a Ballot: BordaCount()"),
    (lambda: Weakest(Pairwise()), "Weakest expects Links, got Duels: Pairwise()"),
    (lambda: Weakest(Highest(Weakest(Margins(Pairwise())))),
     "Weakest expects Links, got a Winner: Highest(Weakest(Margins(Pairwise())))"),
    (lambda: Unbeaten(Tally(BordaCount())), "Unbeaten expects Links, got Scores: Tally(BordaCount())"),
    (lambda: Unbeaten(Pairwise()), "Unbeaten expects Links, got Duels: Pairwise()"),
    (lambda: Unbeaten(METHODS["borda"]), "Unbeaten expects Links, got a Winner: Highest(Tally(BordaCount()))"),
    (lambda: Unbeaten(Margins(Pairwise()), against=Tally(Plurality()), order=Tally(Plurality())),
     "Unbeaten expects a Winner as against, got Scores: Tally(Plurality())"),
    (lambda: Unbeaten(Margins(Pairwise()), against=METHODS["fptp"], order=Plurality()),
     "Unbeaten expects Scores as order, got a Ballot: Plurality()"),
    (lambda: Fallback(BordaCount(), METHODS["borda"]), "Fallback expects a Winner, got a Ballot: BordaCount()"),
    (lambda: Fallback(METHODS["borda"], Weakest(Margins(Pairwise()))),
     "Fallback expects a Winner, got Scores: Weakest(Margins(Pairwise()))"),
    (lambda: Runoff(Pairwise(), METHODS["fptp"], METHODS["borda"]), "Runoff expects Links, got Duels: Pairwise()"),
    (lambda: Runoff(Margins(Pairwise()), BordaCount(), METHODS["borda"]),
     "Runoff expects a Winner, got a Ballot: BordaCount()"),
    (lambda: Runoff(Margins(Pairwise()), METHODS["fptp"], Tally(Plurality())),
     "Runoff expects a Winner, got Scores: Tally(Plurality())"),
])
def test_wrong_blocks_fail_when_built(build, message):
    with pytest.raises(TypeError, match=re.escape(message)):
        build()


def test_unknown_how_fails():
    with pytest.raises(ValueError, match=re.escape('Eliminate how must be "min" or "mean", got \'max\'')):
        Eliminate(Tally(BordaCount()), how="max")
