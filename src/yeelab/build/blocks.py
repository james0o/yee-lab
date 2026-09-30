"""The blocks voting methods are built from, like in Scratch but as constructors:

    Highest(Tally(BordaCount()))                Borda count
    Eliminate(Tally(Plurality()), how="min")    instant runoff

Every block has one type of output:

    Ballot   Plurality(), BordaCount()          points one voter gives a candidate
    Scores   Tally(ballot)                      mean points of each candidate over the
                                                voters of a point
    Winner   Highest(scores), Schulze(),        winner and margin at every point
             Eliminate(scores, how=...)

A ballot gives weight(k, C) points to the candidate at position k (0 = closest) of C;
both count only the remaining candidates, and a higher score is better. Each ballot
also knows the cheapest formula for its mean over the voters (Ballot.tally), from the
shares in voters.Voters.

A Winner returns (winner, margin) like margin/methods.py: the margin is >= 0,
continuous in the shares and 0 on every border between two winners, since
margin/regions.py draws the borders as its zero set. Each decision a block takes has
a gap, 0 where the decision flips, and the margin is the smallest gap of all of them.

Blocks are frozen dataclasses, so equal blocks are equal methods and hash alike, and
repr(block) is the expression that builds it: eval(repr(block)) == block.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yeelab.build.rounds import drop_below_mean, drop_lowest
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Share, Voters
from yeelab.margin.methods import schulze_margin
from yeelab.voting import irv_rounds

Result = tuple[np.ndarray, np.ndarray]  # (winner, margin) at every point
HOW = ("min", "mean")  # what Eliminate drops each round


class Block:
    """Any block; `kind` names its type of output in error messages."""

    kind = "a block"


def _expect(block: Block, value, expected: type, name: str):
    """TypeError unless value is an `expected`, e.g. "Highest expects Scores, got a
    Ballot: BordaCount()"."""
    if isinstance(value, expected):
        return
    if isinstance(value, Block):
        got = f"{value.kind}: {value!r}"
    elif isinstance(value, type) and issubclass(value, Block):
        got = f"the class {value.__name__}; call it: {value.__name__}()"
    else:
        got = f"{type(value).__name__} {value!r}"
    raise TypeError(f"{type(block).__name__} expects {name}, got {got}")


def _top_two(scores: np.ndarray) -> Result:
    """Highest score (winner; ties: the lowest index) and its lead over the second."""
    top = np.partition(scores, -2, axis=-1)
    return scores.argmax(axis=-1), top[..., -1] - top[..., -2]

# ---------------------------------------------------------------- Ballot


class Ballot(Block):
    """Points one voter gives each candidate, by its position among the remaining ones."""

    kind = "a Ballot"
    needs: frozenset[Share]            # shares tally() reads for all candidates
    needs_remaining: frozenset[Share]  # ... for the remaining ones of an elimination

    def weight(self, k: int, n: int) -> float:
        """Points for the candidate at position k (0 = closest) of n remaining candidates."""
        raise NotImplementedError

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        """Mean points of each candidate over the voters of every point, (..., C): the
        sum over rankings of their share times weight(position among the remaining,
        number remaining). alive (..., C) marks the remaining candidates of every point,
        None all of them; the others score 0."""
        raise NotImplementedError

    def eliminate(self, voters: Voters, how: str) -> Result | None:
        """Eliminate(Tally(self), how) by a formula of the ballot's own, or None to
        tally the remaining candidates again in every round."""
        return None


@dataclass(frozen=True)
class Plurality(Ballot):
    """One point for the first choice among the remaining candidates."""

    needs = frozenset({FIRST})
    needs_remaining = frozenset({PROFILE})

    def weight(self, k: int, n: int) -> float:
        return 1 if k == 0 else 0

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        """The first-choice shares, or, among the remaining candidates, the shares of the
        rankings whose first remaining candidate each one is."""
        if alive is None:
            return voters.first
        rankings, probs = voters.rankings, voters.probs
        n_ballots, n = rankings.shape
        top = alive[..., rankings].argmax(axis=-1)  # position of the first remaining, (..., R)
        choice = rankings[np.arange(n_ballots), top].reshape(-1, n_ballots)
        index = choice + n * np.arange(len(choice))[:, None]  # (point, candidate), flat
        votes = np.bincount(index.ravel(), probs.reshape(-1), n * len(choice))
        return votes.reshape(*probs.shape[:-1], n)

    def eliminate(self, voters: Voters, how: str) -> Result | None:
        """Instant runoff (how="min") by the compiled rounds of voting.irv_rounds, which
        move only the ballots of the eliminated candidate."""
        return irv_rounds(voters.rankings, voters.probs) if how == "min" else None


@dataclass(frozen=True)
class BordaCount(Ballot):
    """n - 1 - k points for position k: one per remaining candidate ranked below."""

    needs = needs_remaining = frozenset({PAIRWISE})

    def weight(self, k: int, n: int) -> float:
        return n - 1 - k

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        """sum over the remaining e of d[c, e]: a ballot gives c one point per remaining
        candidate below it, and the mean of that count is the sum of P(c above e)."""
        d = voters.pairwise
        if alive is None:
            return d.sum(axis=-1)
        weights = alive.astype(d.dtype)
        return np.einsum("...ce,...e->...c", d, weights) * weights

# ---------------------------------------------------------------- Scores


class Scores(Block):
    """A score for each candidate at every point, higher is better."""

    kind = "Scores"
    needs: frozenset[Share]

    def evaluate(self, voters: Voters) -> np.ndarray:
        """Scores (..., C)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Tally(Scores):
    """Mean points of the ballot over the voters of every point."""

    ballot: Ballot

    def __post_init__(self):
        _expect(self, self.ballot, Ballot, "a Ballot")

    @property
    def needs(self) -> frozenset[Share]:
        return self.ballot.needs

    @property
    def needs_remaining(self) -> frozenset[Share]:
        return self.ballot.needs_remaining

    def evaluate(self, voters: Voters, alive: np.ndarray | None = None) -> np.ndarray:
        """Scores (..., C) among the candidates marked in alive (..., C), all for None."""
        return self.ballot.tally(voters, alive)

    def __repr__(self):
        return f"Tally({self.ballot!r})"

# ---------------------------------------------------------------- Winner


class Winner(Block):
    """A voting method: winner and margin at every point, from the shares in `needs`."""

    kind = "a Winner"
    needs: frozenset[Share]

    def evaluate(self, voters: Voters) -> Result:
        """(winner, margin), both of voters.shape."""
        raise NotImplementedError


@dataclass(frozen=True)
class Highest(Winner):
    """The highest score wins (ties: the lowest index); the margin is its lead over the
    second."""

    scores: Scores

    def __post_init__(self):
        _expect(self, self.scores, Scores, "Scores")

    @property
    def needs(self) -> frozenset[Share]:
        return self.scores.needs

    def evaluate(self, voters: Voters) -> Result:
        return _top_two(self.scores.evaluate(voters))

    def __repr__(self):
        return f"Highest({self.scores!r})"


@dataclass(frozen=True)
class Eliminate(Winner):
    """Drops candidates round by round, tallying the remaining ones again each round,
    until one is left:

        how="min"   the lowest score goes (ties: the lowest index); the round's gap is
                    the second lowest score minus the lowest
        how="mean"  every score at most the mean of the remaining ones goes; if that is
                    all of them, all scores are equal and the lowest index wins. The
                    round's gap is the smallest distance of a score from the mean

    The margin is the smallest gap of all rounds, like voting.irv_rounds: the winner
    changes only where some round's elimination flips, and there the gap is 0.
    """

    scores: Tally
    how: Literal["min", "mean"]

    def __post_init__(self):
        _expect(self, self.scores, Tally, "a Tally (it tallies the remaining candidates again)")
        if self.how not in HOW:
            raise ValueError(f'Eliminate how must be "min" or "mean", got {self.how!r}')

    @property
    def needs(self) -> frozenset[Share]:
        return self.scores.needs_remaining

    def evaluate(self, voters: Voters) -> Result:
        result = self.scores.ballot.eliminate(voters, self.how)
        return self.rounds(voters) if result is None else result

    def rounds(self, voters: Voters) -> Result:
        """evaluate() with one tally per round, whatever formula the ballot has of its
        own (Ballot.eliminate)."""
        n = voters.n_candidates
        alive = np.ones((*voters.shape, n), dtype=bool)
        margin = np.full(voters.shape, np.inf)
        drop = drop_lowest if self.how == "min" else drop_below_mean
        for _ in range(n - 1):  # every round drops at least one candidate
            gap = drop(self.scores.evaluate(voters, alive), alive)
            np.minimum(margin, gap, out=margin)
            if self.how == "mean" and alive.sum(axis=-1).max() == 1:  # often done early
                break
        return alive.argmax(axis=-1), margin

    def __repr__(self):
        return f'Eliminate({self.scores!r}, how="{self.how}")'


@dataclass(frozen=True)
class Schulze(Winner):
    """Schulze method on pairwise shares: winner and margin of
    margin.methods.schulze_margin."""

    needs = frozenset({PAIRWISE})

    def evaluate(self, voters: Voters) -> Result:
        return schulze_margin(voters.pairwise)
