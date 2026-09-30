"""Checks of the methods built from blocks (yeelab.build).

On the complete profile of every pixel (pixels/), the built fptp, irv, borda and
schulze must pick the winners of pixels.methods, and every elimination the winners
and margins of a plain Python reference that follows the definition: scores from
weight(k, C) among the remaining candidates, one point at a time. The fast tally of
every ballot must be that same sum over the profile, for any remaining candidates.
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
    Highest,
    Plurality,
    Schulze,
    Tally,
    Voters,
)
from yeelab.pixels import beta as pixel_beta, methods as pixel_methods, normal as pixel_normal

PIXELS = 40
TOLERANCE = 1e-6  # winners are compared where the margin is above (float32 profiles)
# voters (distribution, deviation, Beta spread) and random candidates (number, seed)
VOTERS = [("beta", 0.1, "rms"), ("beta", 0.3, "mean_abs"), ("normal", 0.1, None), ("normal", 0.3, None)]
CANDIDATES = [(3, 0), (5, 1), (7, 2)]
BALLOTS = [Plurality(), BordaCount()]
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


@pytest.mark.parametrize("name", ["fptp", "irv", "borda", "schulze"])
def test_methods_match_pixel_methods(profile, name):
    rankings, probs = profile
    winner, margin = METHODS[name].evaluate(_voters(rankings, probs))
    assert (margin >= 0).all()
    clear = margin > TOLERANCE  # off the borders
    assert clear.mean() > 0.5  # not vacuous; near-empty candidates tie, e.g. in IRV
    np.testing.assert_array_equal(winner[clear], pixel_methods.METHODS[name](rankings, probs)[clear])


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


def test_methods_are_the_six_expressions():
    assert METHODS == {
        "fptp": Highest(Tally(Plurality())),
        "irv": Eliminate(Tally(Plurality()), how="min"),
        "borda": Highest(Tally(BordaCount())),
        "baldwin": Eliminate(Tally(BordaCount()), how="min"),
        "nanson": Eliminate(Tally(BordaCount()), how="mean"),
        "schulze": Schulze(),
    }


def test_repr_is_the_expression():
    assert repr(METHODS["nanson"]) == 'Eliminate(Tally(BordaCount()), how="mean")'
    assert repr(METHODS["schulze"]) == "Schulze()"
    assert repr(METHODS["fptp"]) == "Highest(Tally(Plurality()))"
    for method in [*METHODS.values(), *ELIMINATIONS, *BALLOTS]:
        assert eval(repr(method)) == method


def test_blocks_are_frozen_and_hashable():
    assert len(set(METHODS.values())) == len(METHODS)
    cache = {Eliminate(Tally(BordaCount()), how="mean"): "nanson"}
    assert cache[METHODS["nanson"]] == "nanson"
    with pytest.raises(FrozenInstanceError):
        METHODS["irv"].how = "mean"


def test_needs():
    assert METHODS["fptp"].needs == {FIRST}
    assert METHODS["irv"].needs == {PROFILE}
    for name in ("borda", "baldwin", "nanson", "schulze"):
        assert METHODS[name].needs == {PAIRWISE}, name
    assert Eliminate(Tally(Plurality()), how="mean").needs == {PROFILE}


@pytest.mark.parametrize("build, message", [
    (lambda: Highest(BordaCount()), "Highest expects Scores, got a Ballot: BordaCount()"),
    (lambda: Highest(Schulze()), "Highest expects Scores, got a Winner: Schulze()"),
    (lambda: Tally(Tally(Plurality())), "Tally expects a Ballot, got Scores: Tally(Plurality())"),
    (lambda: Tally(Schulze()), "Tally expects a Ballot, got a Winner: Schulze()"),
    (lambda: Tally(Plurality), "Tally expects a Ballot, got the class Plurality; call it: Plurality()"),
    (lambda: Tally(3), "Tally expects a Ballot, got int 3"),
    (lambda: Eliminate(BordaCount(), how="min"), "Eliminate expects a Tally"),
    (lambda: Eliminate(Highest(Tally(Plurality())), how="min"),
     "Eliminate expects a Tally (it tallies the remaining candidates again), got a Winner"),
])
def test_wrong_blocks_fail_when_built(build, message):
    with pytest.raises(TypeError, match=re.escape(message)):
        build()


def test_unknown_how_fails():
    with pytest.raises(ValueError, match=re.escape('Eliminate how must be "min" or "mean", got \'max\'')):
        Eliminate(Tally(BordaCount()), how="max")
