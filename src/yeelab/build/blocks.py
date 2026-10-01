"""The blocks voting methods are built from, like in Scratch but as constructors:

    Highest(Tally(BordaCount()))                    Borda count
    Eliminate(Tally(Plurality()), how="min")        instant runoff
    Unbeaten(StrongestPaths(Margins(Pairwise())))   Schulze

Every block has one type of output:

    Ballot   Plurality(), BordaCount()          points one voter gives a candidate
    Scores   Tally(ballot), Weakest(links)      a score of each candidate at a point
    Duels    Pairwise()                         share of the voters ranking c above e
    Links    Margins(duels),                    how strongly c beats e (antisymmetric)
             StrongestPaths(links)
    Winner   Highest(scores), Unbeaten(links),  winner and margin at every point
             Eliminate(scores, how=...),
             Fallback(first, second),
             Unbeaten(links, against=winner, order=scores),
             Runoff(links, first, second)

A ballot gives weight(k, C) points to the candidate at position k (0 = closest) of C;
both count only the remaining candidates, and a higher score is better. Each ballot
also knows the cheapest formula for its mean over the voters (Ballot.tally), from the
shares in voters.Voters. Tally averages them over the voters of a point.

A link s[c, e] = -s[e, c] says that c beats e where it is positive. Links are
transitive (Links.transitive) if the candidates that beat each other cannot form a
cycle: Margins are not, StrongestPaths are.

Unbeaten(links) elects the candidate no one beats. Unbeaten(links, against=winner,
order=scores) only asks that of the winner of `against`, the king: if someone beats it,
the highest `order` score among those who do wins instead. King of the hill is

    Unbeaten(Margins(Pairwise()), against=fptp, order=Tally(Plurality()))

Runoff(links, first, second) puts the winners of two methods against each other: the
one who beats the other on the links wins.

A Winner returns (winner, margin): the margin is >= 0, continuous in the shares and 0
on every border between two winners, since margin/regions.py draws the borders as its
zero set. Each decision a block takes has a gap, 0 where the decision flips, and the
margin is the smallest gap of all of them. Where no one is elected, the winner is
voting.CYCLE.

Blocks are frozen dataclasses, so equal blocks are equal methods and hash alike, and
repr(block) is the expression that builds it: eval(repr(block)) == block.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yeelab.build.rounds import drop_below_mean, drop_lowest
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Share, Voters
from yeelab.voting import CYCLE, irv_rounds

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


def _off_diagonal(links: np.ndarray, fill: float) -> np.ndarray:
    """links (..., C, C) with the diagonal, a candidate against itself, replaced by fill
    (inf or -inf), so that min and max are over the other candidates only."""
    return np.where(np.eye(links.shape[-1], dtype=bool), fill, links)


def _paths(links: np.ndarray) -> np.ndarray:
    """p - p^T for links s (..., C, C), where p[c, e] is the strength of the strongest
    path from c to e over the links max(s, 0): the largest, over the paths, of the
    smallest link along it (Floyd-Warshall)."""
    p = np.maximum(links, 0.0)
    through = np.empty_like(p)
    for k in range(p.shape[-1]):  # in place: row and column k do not change in step k (p[k, k] = 0)
        np.minimum(p[..., :, k, None], p[..., None, k, :], out=through)
        np.maximum(p, through, out=p)
    return p - np.swapaxes(p, -1, -2)


def _unbeaten(links: np.ndarray, transitive: bool) -> Result:
    """Winner and margin of Unbeaten for links s (..., C, C) (see there)."""
    n = links.shape[-1]
    beaten = _off_diagonal(links, -np.inf).max(axis=-2)  # [..., e] = max_{f != e} s[f, e]
    winner = beaten.argmin(axis=-1)
    others = np.where(np.arange(n) == winner[..., None], np.inf, np.maximum(beaten, 0.0))
    margin = others.min(axis=-1)
    if not transitive:
        best = beaten.min(axis=-1)  # the winner's: negative if it beats everyone
        margin = np.minimum(margin, np.abs(best))
        winner = np.where(best < 0, winner, CYCLE)
    return winner, margin


def _challenged(links: np.ndarray, king: np.ndarray, margin: np.ndarray, scores: np.ndarray) -> Result:
    """Winner and margin of Unbeaten with `against` (see there) for links s (..., C, C):
    the king and its margin (...), CYCLE where there is none, and the scores (..., C)
    that order its challengers."""
    cycle = king == CYCLE
    seat = np.where(cycle, 0, king)  # any candidate in a cycle: the result is replaced below
    to_king = np.take_along_axis(links, seat[..., None, None], axis=-1)[..., 0]  # [..., c] = s[c, king]
    is_king = np.arange(links.shape[-1]) == seat[..., None]
    flip = np.where(is_king, np.inf, np.abs(to_king)).min(axis=-1)
    challenger = to_king > 0  # never the king: s[king, king] = 0
    # the others score -inf, but an unchallenged king inf: it wins, and the lead is inf
    # with fewer than two challengers (never -inf - -inf)
    unchallenged = is_king & ~challenger.any(axis=-1, keepdims=True)
    winner, lead = _top_two(np.where(challenger, scores, np.where(unchallenged, np.inf, -np.inf)))
    gap = np.minimum(margin, np.minimum(flip, lead))
    return np.where(cycle, CYCLE, winner), np.where(cycle, margin, gap)

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

# ---------------------------------------------------------------- Duels


class Duels(Block):
    """For every pair of candidates, the share of the voters who rank one above the other."""

    kind = "Duels"
    needs: frozenset[Share]

    def evaluate(self, voters: Voters) -> np.ndarray:
        """d (..., C, C): d[c, e] = share ranking c above e, d[c, c] = 0."""
        raise NotImplementedError


@dataclass(frozen=True)
class Pairwise(Duels):
    """The pairwise shares of the voters, as they are."""

    needs = frozenset({PAIRWISE})

    def evaluate(self, voters: Voters) -> np.ndarray:
        return voters.pairwise

# ---------------------------------------------------------------- Links


class Links(Block):
    """How strongly each candidate beats each other one: s[c, e] = -s[e, c], and c beats
    e where s[c, e] > 0. `transitive` is True if the links cannot form a cycle, c beating
    e beating f and f beating c, which is what Unbeaten needs to elect someone always."""

    kind = "Links"
    needs: frozenset[Share]
    transitive: bool

    def evaluate(self, voters: Voters) -> np.ndarray:
        """s (..., C, C), s[c, c] = 0."""
        raise NotImplementedError

    def unbeaten(self, voters: Voters) -> Result | None:
        """Unbeaten(self) by a formula of the links' own, or None to decide on evaluate()."""
        return None


@dataclass(frozen=True)
class Margins(Links):
    """d - d^T: how much more of the voters rank c above e than e above c. Not transitive,
    the majorities can go round in a cycle."""

    duels: Duels
    transitive = False

    def __post_init__(self):
        _expect(self, self.duels, Duels, "Duels")

    @property
    def needs(self) -> frozenset[Share]:
        return self.duels.needs

    def evaluate(self, voters: Voters) -> np.ndarray:
        d = self.duels.evaluate(voters)
        return d - np.swapaxes(d, -1, -2)

    def __repr__(self):
        return f"Margins({self.duels!r})"


@dataclass(frozen=True)
class StrongestPaths(Links):
    """p - p^T, where p[c, e] is the strength of the strongest path from c to e over the
    links that say a candidate beats another (see _paths). Transitive: c beats e where
    p[c, e] > p[e, c], and that relation has no cycles.

    A candidate c who beats everyone directly is unbeaten on the paths as well: p[c, e] >=
    s[c, e] > 0, and no link leads into c, so p[e, c] = 0. Unbeaten(self) therefore only
    computes the paths where there is no such candidate."""

    links: Links
    transitive = True

    def __post_init__(self):
        _expect(self, self.links, Links, "Links")

    @property
    def needs(self) -> frozenset[Share]:
        return self.links.needs

    def evaluate(self, voters: Voters) -> np.ndarray:
        return _paths(self.links.evaluate(voters))

    def unbeaten(self, voters: Voters) -> Result:
        """Unbeaten(self), with the paths only at the points where no candidate beats
        everyone directly. Elsewhere that candidate wins, with the margin of the links
        themselves: positive, at most the margin on the paths, and 0 on the border of
        that region."""
        s = self.links.evaluate(voters)
        winner, margin = _unbeaten(s, transitive=False)
        cycle = winner == CYCLE
        if cycle.any():
            winner[cycle], margin[cycle] = _unbeaten(_paths(s[cycle]), transitive=True)
        return winner, margin

    def __repr__(self):
        return f"StrongestPaths({self.links!r})"

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


@dataclass(frozen=True)
class Weakest(Scores):
    """The weakest link of a candidate, min_{e != c} s[c, e]: its narrowest win, or,
    if it loses somewhere, minus its worst defeat."""

    links: Links

    def __post_init__(self):
        _expect(self, self.links, Links, "Links")

    @property
    def needs(self) -> frozenset[Share]:
        return self.links.needs

    def evaluate(self, voters: Voters) -> np.ndarray:
        return _off_diagonal(self.links.evaluate(voters), np.inf).min(axis=-1)

    def __repr__(self):
        return f"Weakest({self.links!r})"

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
class Unbeaten(Winner):
    """The candidate no one beats, or the one beaten the least. beaten[e] = max_{f != e}
    s[f, e] is the strongest link into e: negative if e beats everyone, positive if
    someone beats e. The winner has the lowest beaten (ties: the lowest index).

    Its margin is the smallest max(beaten[e], 0) over the other candidates e: how far
    the closest of them is from being unbeaten, which is when the winner would change.

    Links that are not transitive can form a cycle. Then the winner must also beat
    everyone (beaten < 0), or the point is a voting.CYCLE, and since that flips where
    the winner's beaten crosses 0, |beaten| of the winner is a gap too: the margin is at
    most that. Transitive links leave it out: someone is always unbeaten, and on
    strongest paths the winner's beaten is 0 on whole areas, where the widest paths to
    and from another candidate share their weakest link.

    With `against` and `order` (both or neither) only one candidate has to stay unbeaten,
    the king: the winner of `against`. Its challengers are the candidates c who beat it,
    s[c, king] > 0. Without challengers the king wins, otherwise the challenger with the
    highest `order` score does (ties: the lowest index), in one step: no one challenges
    that winner in turn. Where `against` elects no one, the point stays a voting.CYCLE
    with the margin of `against`. Elsewhere the margin is the smallest of three gaps:

        the margin of `against`             the king changes
        min_{c != king} |s[c, king]|        a candidate starts or stops being a challenger
        the best challenger's lead in       another challenger wins; inf with fewer than
        `order` over the second             two challengers
    """

    links: Links
    against: Winner | None = None
    order: Scores | None = None

    def __post_init__(self):
        _expect(self, self.links, Links, "Links")
        if (self.against is None) != (self.order is None):
            raise ValueError("Unbeaten takes against and order together, or neither")
        if self.against is not None:
            _expect(self, self.against, Winner, "a Winner as against")
            _expect(self, self.order, Scores, "Scores as order")

    @property
    def needs(self) -> frozenset[Share]:
        if self.against is None:
            return self.links.needs
        return self.links.needs | self.against.needs | self.order.needs

    def evaluate(self, voters: Voters) -> Result:
        result = self.links.unbeaten(voters) if self.against is None else None
        return self.decide(voters) if result is None else result

    def decide(self, voters: Voters) -> Result:
        """evaluate() on the links as they are, whatever formula they have of their own
        (Links.unbeaten)."""
        links = self.links.evaluate(voters)
        if self.against is None:
            return _unbeaten(links, self.links.transitive)
        return _challenged(links, *self.against.evaluate(voters), self.order.evaluate(voters))

    def __repr__(self):
        if self.against is None:
            return f"Unbeaten({self.links!r})"
        return f"Unbeaten({self.links!r}, against={self.against!r}, order={self.order!r})"


@dataclass(frozen=True)
class Fallback(Winner):
    """The winner of `first`, or where it elects no one (voting.CYCLE), the winner of
    `second`. The margin is first's where it elects someone, and the smaller of the two
    in a cycle of first: the winner changes where either one would."""

    first: Winner
    second: Winner

    def __post_init__(self):
        _expect(self, self.first, Winner, "a Winner")
        _expect(self, self.second, Winner, "a Winner")

    @property
    def needs(self) -> frozenset[Share]:
        return self.first.needs | self.second.needs

    def evaluate(self, voters: Voters) -> Result:
        winner, margin = self.first.evaluate(voters)
        cycle = winner == CYCLE
        if cycle.any():
            other, gap = self.second.evaluate(voters)
            winner = np.where(cycle, other, winner)
            margin = np.where(cycle, np.minimum(margin, gap), margin)
        return winner, margin

    def __repr__(self):
        return f"Fallback({self.first!r}, {self.second!r})"


@dataclass(frozen=True)
class Runoff(Winner):
    """The winner of `first` against the winner of `second`, one on one: `second` wins
    where it beats `first` on the links, s[second, first] > 0, otherwise `first` does.
    Two candidates cannot form a cycle, so not even links that are not transitive leave
    the duel open. Where both elect the same candidate, that candidate wins.

    The margin is the smallest of three gaps: the margins of `first` and of `second` (a
    finalist changes) and |s[first, second]| (the duel flips), which is left out where
    both are the same candidate. Where either one elects no one (voting.CYCLE) there is
    no duel either, and the point stays a CYCLE with the smaller of their two margins."""

    links: Links
    first: Winner
    second: Winner

    def __post_init__(self):
        _expect(self, self.links, Links, "Links")
        _expect(self, self.first, Winner, "a Winner")
        _expect(self, self.second, Winner, "a Winner")

    @property
    def needs(self) -> frozenset[Share]:
        return self.links.needs | self.first.needs | self.second.needs

    def evaluate(self, voters: Voters) -> Result:
        first, margin = self.first.evaluate(voters)
        second, gap = self.second.evaluate(voters)
        links = self.links.evaluate(voters)
        n = links.shape[-1]
        cycle = (first == CYCLE) | (second == CYCLE)
        a, b = np.where(cycle, 0, first), np.where(cycle, 0, second)  # a = b in a cycle: no duel
        pairs = links.reshape(*links.shape[:-2], n * n)
        duel = np.take_along_axis(pairs, (a * n + b)[..., None], axis=-1)[..., 0]  # s[first, second]
        margin = np.minimum(np.minimum(margin, gap), np.where(a == b, np.inf, np.abs(duel)))
        return np.where(cycle, CYCLE, np.where(duel < 0, second, first)), margin

    def __repr__(self):
        return f"Runoff({self.links!r}, {self.first!r}, {self.second!r})"
