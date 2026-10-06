"""The blocks voting methods are built from, like in Scratch but as constructors:

    Highest(Tally(BordaCount()))                    Borda count
    Eliminate(Tally(Plurality()), how="min")        instant runoff
    Unbeaten(StrongestPaths(Margins(Pairwise())))   Schulze

Every block has one type of output:

    Ballot           Plurality(), BordaCount(),         points one voter gives a candidate
                     Score(levels, power=...), ScoreAvg(levels), ScoreDH(levels, delta=...),
                     ScoreHybrid(levels, delta=...), ScoreCluster(levels, mu=..., kappa=...),
                     Mix(first, second, share=...)
    CandidateTotals  Tally(ballot), Weakest(diffs)      a total of each candidate at a point
    PairShares       Pairwise()                         share of the voters ranking c above e
    PairDiffs        Margins(shares),                   how strongly c beats e (antisymmetric)
                     StrongestPaths(diffs), ScoreComparisons(Score(...))
    Winner           Highest(totals, n=...),            the candidates chosen and the
                     Eliminate(totals, how=..., until=...),  margin at every point;
                     Unbeaten(diffs, among=...),        a method chooses one, its winner
                     Unbeaten(diffs, against=winner, order=totals),
                     Fallback(first, second), first | second

A ranked ballot gives weight(k, C) points to the candidate at position k (0 = closest)
of C; both count only the remaining candidates, and a higher total is better. A score
ballot gives a score from 0 to levels - 1 (yeelab.score), which depends on how far the
candidates are and not only on their order: the top score to the closest candidate, 0 to
the farthest, and to the others, for Score(levels, power=...), by where their distance
is between the two, its part of the way to the power `power` (1: in proportion to it);
ScoreAvg(levels) likewise, with the mean distance in the middle of
the scale; ScoreDH(levels, delta=...) shares the steps from the top score to 0 out among
the gaps between neighbours in the order of distance, each to the gap with the largest
gap / (its steps + delta), D'Hondt's rule at delta = 1; ScoreHybrid(levels, delta=...)
does that for the candidates closer than halfway, on the upper half of the scale, and
Score's for the others, on the lower half; ScoreCluster(levels, mu=..., kappa=...)
gives Score's scores unless that splits a cluster of candidates at nearly the same
distance, at a cost of mu, kappa setting how far a cluster must stand apart. With two levels
each is an approval ballot. Its total is the mean score as a part of the top score, counted down
from 1: a voter who does not give a candidate the top score takes off what is missing.
Mix(first, second, share=s) is two kinds of voters: s of them mark `second`, the others
`first`.
Each ballot also knows the cheapest formula for its mean over the voters (Ballot.tally),
from the shares in voters.Voters. Tally averages them over the voters of a point.

A diff s[c, e] = -s[e, c] says that c beats e where it is positive. PairDiffs are
transitive (PairDiffs.transitive) if the candidates that beat each other cannot form a
cycle: Margins are not, StrongestPaths are.

ScoreComparisons(Score(...)) compares the cardinal scores on each ballot: a voter who
gives a pair equal scores abstains. It does not use ranked pairwise shares.

Unbeaten(diffs) elects the candidate no one beats. Unbeaten(diffs, against=winner,
order=totals) only asks that of the winner of `against`, the king: if someone beats it,
the highest `order` total among those who do wins instead. King of the hill is

    Unbeaten(Margins(Pairwise()), against=fptp, order=Tally(Plurality()))

Eliminate drops the lowest total (how="min"), every total at most the mean ("mean"),
or all but the highest ("all", in one round) until `until` candidates are left;
Highest(totals, n) is Eliminate(totals, how="all", until=n). Unbeaten(diffs,
among=finalists) is a runoff among the candidates `finalists` chooses: the one who
beats the others wins, and a tied duel on Margins or ScoreComparisons is a draw, no
one. A tie is exact, so it is only ever on a border, where the margin is 0. STAR is

    Unbeaten(ScoreComparisons(ballot), among=Highest(Tally(ballot), n=2))

the finalist more voters score higher. first | second (Union) chooses the candidates either one does: the winners of two
methods as the finalists of a runoff.

A Winner chooses candidates (select): a mask (..., C) and a margin, >= 0, continuous in
the shares and 0 wherever the choice changes, since margin/regions.py draws the borders
as its zero set. Each decision a block takes has a gap, 0 where the decision flips, and
the margin is the smallest gap of all of them. A voting method chooses one candidate
(seats == 1), and evaluate() gives its (winner, margin); where no one is chosen, in a
cycle or a draw, the winner is voting.CYCLE.

Blocks are frozen dataclasses, so equal blocks are equal methods and hash alike, and
repr(block) is the expression that builds it: eval(repr(block)) == block.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from yeelab.build.rounds import drop_below_mean, drop_lowest
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Scored, ScoredPairwise, Share, Voters
from yeelab.score import AVG, CLUSTER, DELTA, DHONDT, HYBRID, KAPPA, MU, POWER, RANGE
from yeelab.voting import CYCLE, irv_rounds

Result = tuple[np.ndarray, np.ndarray]  # (winner, margin) or (chosen, margin) at every point
HOW = ("min", "mean", "all")  # what Eliminate drops each round


class Block:
    """Any block; `kind` names its type of output in error messages."""

    kind = "a block"


def _expect(block: Block, value, expected: type, name: str):
    """TypeError unless value is an `expected`, e.g. "Highest expects CandidateTotals,
    got a Ballot: BordaCount()"."""
    if isinstance(value, expected):
        return
    if isinstance(value, Block):
        got = f"{value.kind}: {value!r}"
    elif isinstance(value, type) and issubclass(value, Block):
        got = f"the class {value.__name__}; call it: {value.__name__}()"
    else:
        got = f"{type(value).__name__} {value!r}"
    raise TypeError(f"{type(block).__name__} expects {name}, got {got}")


def _top_two(totals: np.ndarray) -> Result:
    """Highest total (winner; ties: the lowest index) and its lead over the second."""
    top = np.partition(totals, -2, axis=-1)
    return totals.argmax(axis=-1), top[..., -1] - top[..., -2]


def _off_diagonal(diffs: np.ndarray, fill: float) -> np.ndarray:
    """diffs (..., C, C) with the diagonal, a candidate against itself, replaced by fill
    (inf or -inf), so that min and max are over the other candidates only."""
    return np.where(np.eye(diffs.shape[-1], dtype=bool), fill, diffs)


def _paths(diffs: np.ndarray) -> np.ndarray:
    """p - p^T for diffs s (..., C, C), where p[c, e] is the strength of the strongest
    path from c to e over the diffs max(s, 0): the largest, over the paths, of the
    smallest diff along it (Floyd-Warshall)."""
    p = np.maximum(diffs, 0.0)
    through = np.empty_like(p)
    for k in range(p.shape[-1]):  # in place: row and column k do not change in step k (p[k, k] = 0)
        np.minimum(p[..., :, k, None], p[..., None, k, :], out=through)
        np.maximum(p, through, out=p)
    return p - np.swapaxes(p, -1, -2)


def _unbeaten(diffs: np.ndarray, transitive: bool, running: np.ndarray | None = None) -> Result:
    """Winner and margin of Unbeaten for diffs s (..., C, C) (see there), among the
    candidates marked in running (..., C), all for None."""
    n = diffs.shape[-1]
    if running is None:
        beaten = _off_diagonal(diffs, -np.inf).max(axis=-2)  # [..., e] = max_{f != e} s[f, e]
    else:  # only the running f count, and only the running e can win
        beaten = np.full(running.shape, -np.inf, dtype=diffs.dtype)
        for f in range(n):  # faster than a max over the short axis -2
            into = np.where(running[..., f, None], diffs[..., f, :], -np.inf)
            into[..., f] = -np.inf
            np.maximum(beaten, into, out=beaten)
        beaten = np.where(running, beaten, np.inf)
    winner = beaten.argmin(axis=-1)
    others = np.where(np.arange(n) == winner[..., None], np.inf, np.maximum(beaten, 0.0))
    margin = others.min(axis=-1)
    if not transitive:
        best = beaten.min(axis=-1)  # the winner's: negative if it beats everyone
        margin = np.minimum(margin, np.abs(best))
        winner = np.where(best < 0, winner, CYCLE)
    if running is not None:  # no one runs: no one wins (margin inf, the runners' own decides)
        winner = np.where(running.any(axis=-1), winner, CYCLE)
    return winner, margin


def _challenged(diffs: np.ndarray, king: np.ndarray, margin: np.ndarray, totals: np.ndarray) -> Result:
    """Winner and margin of Unbeaten with `against` (see there) for diffs s (..., C, C):
    the king and its margin (...), CYCLE where there is none, and the totals (..., C)
    that order its challengers."""
    cycle = king == CYCLE
    seat = np.where(cycle, 0, king)  # any candidate in a cycle: the result is replaced below
    to_king = np.take_along_axis(diffs, seat[..., None, None], axis=-1)[..., 0]  # [..., c] = s[c, king]
    is_king = np.arange(diffs.shape[-1]) == seat[..., None]
    flip = np.where(is_king, np.inf, np.abs(to_king)).min(axis=-1)
    challenger = to_king > 0  # never the king: s[king, king] = 0
    # the others get -inf, but an unchallenged king inf: it wins, and the lead is inf
    # with fewer than two challengers (never -inf - -inf)
    unchallenged = is_king & ~challenger.any(axis=-1, keepdims=True)
    winner, lead = _top_two(np.where(challenger, totals, np.where(unchallenged, np.inf, -np.inf)))
    gap = np.minimum(margin, np.minimum(flip, lead))
    return np.where(cycle, CYCLE, winner), np.where(cycle, margin, gap)

# ---------------------------------------------------------------- Ballot


class Ballot(Block):
    """Points one voter gives each candidate: by its position among the remaining ones
    (a ranked ballot, which has a weight), or by its distance (a score ballot)."""

    kind = "a Ballot"
    needs: frozenset[Share]            # shares tally() reads for all candidates
    needs_remaining: frozenset[Share]  # ... for the remaining ones of an elimination

    def weight(self, k: int, n: int) -> float:
        """Points for the candidate at position k (0 = closest) of n remaining candidates;
        only ranked ballots have one."""
        raise NotImplementedError

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        """Mean points of each candidate over the voters of every point, (..., C): for a
        ranked ballot the sum over rankings of their share times weight(position among
        the remaining, number remaining). alive (..., C) marks the remaining candidates
        of every point, None all of them; the others get 0 (-1 on a score ballot,
        which counts down from 1: Score.tally)."""
        raise NotImplementedError

    def eliminate(self, voters: Voters, how: str) -> Result | None:
        """Eliminate(Tally(self), how) down to one by a formula of the ballot's own, or
        None to tally the remaining candidates again in every round."""
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


@dataclass(frozen=True)
class Score(Ballot):
    """A score from 0 to levels - 1 for every candidate (yeelab.score): the top score for
    the closest, 0 for the farthest, and for the others the score by where their
    distance is between those two: its part of the way to the power `power`, rounded to
    a whole score. power = 1, the default score.POWER, is in proportion to the distance;
    above 1 the voter keeps the top scores for the candidates near the closest.

    The total is the mean score as a part of the top score, counted down from 1.
    Score(2) is an approval ballot: the voter approves the candidates beyond
    2^(-1 / power) of the way from the farthest to the closest, halfway for power = 1.
    The other score ballots, subclasses of this one, have no power."""

    levels: int
    power: float = field(default=POWER, kw_only=True)
    rule = RANGE  # of the ballot (yeelab.score); not a field

    def __post_init__(self):
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or self.levels < 2:
            raise ValueError(f"{type(self).__name__} levels must be a whole number of at least 2, "
                             f"got {self.levels!r}")
        power = self.power
        if isinstance(power, bool) or not isinstance(power, int | float) or not 0 < power < np.inf:
            raise ValueError(f"{type(self).__name__} power must be a number above 0, got {power!r}")
        if self.rule != RANGE and power != POWER:  # as for any keyword the block does not take
            raise TypeError(f"{type(self).__name__} has no power; only Score has")

    @property
    def scored(self) -> Scored:
        """Its shares in Voters.unscored."""
        return Scored(self.levels, self.rule, power=self.power)

    @property
    def needs(self) -> frozenset[Share]:
        return frozenset({self.scored})

    needs_remaining = needs

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        """The mean part of the top score the voters give each candidate, minus 1:
        minus the part they do not give it (Voters.unscored). That keeps the lead
        between two candidates that nearly all voters give the top score; the order and
        the leads are those of the mean scores themselves. A voter marks the ballot
        once, among all candidates, so the remaining candidates of an elimination keep
        their totals; the others get -1, as if no one gave them a point."""
        totals = -voters.unscored[self.scored]
        return totals if alive is None else np.where(alive, totals, -1.0)

    def __repr__(self):
        power = f", power={self.power!r}" if self.power != POWER else ""
        return f"{type(self).__name__}({self.levels}{power})"


@dataclass(frozen=True, repr=False)  # the repr of Score, with this name
class ScoreAvg(Score):
    """A score from 0 to levels - 1 for every candidate (yeelab.score): the top score for
    the closest, 0 for the farthest, and the middle of the scale for a candidate at the
    mean distance to all candidates; in between, the score in proportion to where the
    distance is between those, rounded to a whole score. Score is this with the middle
    halfway between the closest and the farthest.

    The total is that of Score. ScoreAvg(2) is an approval ballot: the voter approves
    the candidates closer than the mean distance, the best ballot of a voter who knows
    nothing of how the others vote (Weber)."""

    rule = AVG


@dataclass(frozen=True, repr=False)
class ScoreDH(Score):
    """A score from 0 to levels - 1 for every candidate (yeelab.score): the top score for
    the closest, 0 for the farthest, and the levels - 1 steps between them shared out
    among the gaps between neighbours in the order of distance, each step to the gap
    with the largest gap / (its steps + delta). delta = 1 is D'Hondt, which favours the
    large gaps, and 1/2 Sainte-Laguë, which favours neither; the default, score.DELTA, is
    between them. A candidate gets the steps of the gaps below it, so candidates at nearly
    the same distance share a score at any number of levels: with more levels than
    candidates it is not Borda.

    The total is that of Score. ScoreDH(2) is an approval ballot for any delta: the one
    step goes to the largest gap, and the voter approves the candidates above it."""

    delta: float = field(default=DELTA, kw_only=True)
    rule = DHONDT

    def __post_init__(self):
        super().__post_init__()
        delta = self.delta
        if isinstance(delta, bool) or not isinstance(delta, int | float) or not 0 < delta < np.inf:
            raise ValueError(f"{type(self).__name__} delta must be a number above 0, got {delta!r}")

    @property
    def scored(self) -> Scored:
        return Scored(self.levels, self.rule, self.delta)

    def __repr__(self):
        return f"{type(self).__name__}({self.levels}, delta={self.delta!r})"


@dataclass(frozen=True, repr=False)  # the fields and repr of ScoreDH, with this name
class ScoreHybrid(ScoreDH):
    """A score from 0 to levels - 1 for every candidate (yeelab.score): ScoreDH for the
    candidates closer than halfway between the closest and the farthest, and Score for
    the others, each on its own half of the scale. The upper ceil((levels - 1) / 2) steps
    are shared out as by ScoreDH among the gaps between the closer candidates and the gap
    from the last of them to halfway; the farther candidates get the lower
    floor((levels - 1) / 2) in proportion to where their distance is between halfway and
    the farthest. Every closer candidate gets a score at least that of every farther one.

    The total is that of Score. ScoreHybrid(2) is an approval ballot for any delta: the
    voter approves the candidates above the largest gap among those closer than halfway,
    the gap to halfway included."""

    rule = HYBRID


@dataclass(frozen=True, repr=False)
class ScoreCluster(Score):
    """A score from 0 to levels - 1 for every candidate (yeelab.score): the top score for
    the closest, 0 for the farthest, and to the others the scores closest to those of
    Score, never less for a closer candidate, with a cost for each pair of neighbours in
    the order of distance that get different scores although they are less than a step
    apart and form a cluster: a run of candidates much closer to each other than to the
    candidates around them. So a tight cluster keeps one score with few levels, and is
    graded like Score with many. mu is the cost of a split, 0 for Score's ballot; a run of
    candidates is a cluster when its gaps are less than 1 / kappa of the gaps around it.
    The defaults are score.MU and score.KAPPA.

    The total is that of Score. ScoreCluster(2) is an approval ballot: the voter approves
    the candidates Score(2) approves, unless that cut splits a cluster and another gap
    costs less."""

    mu: float = field(default=MU, kw_only=True)
    kappa: float = field(default=KAPPA, kw_only=True)
    rule = CLUSTER

    def __post_init__(self):
        super().__post_init__()
        mu, kappa = self.mu, self.kappa
        if isinstance(mu, bool) or not isinstance(mu, int | float) or not 0 <= mu < np.inf:
            raise ValueError(f"{type(self).__name__} mu must be a number of 0 or more, got {mu!r}")
        if isinstance(kappa, bool) or not isinstance(kappa, int | float) or not 0 < kappa < np.inf:
            raise ValueError(f"{type(self).__name__} kappa must be a number above 0, got {kappa!r}")

    @property
    def scored(self) -> Scored:
        return Scored(self.levels, self.rule, mu=self.mu, kappa=self.kappa)

    def __repr__(self):
        return f"{type(self).__name__}({self.levels}, mu={self.mu!r}, kappa={self.kappa!r})"


@dataclass(frozen=True)
class Mix(Ballot):
    """Two kinds of voters: `share` of the voters of every point mark `second`, the
    others `first`. The tally is that mix of the two tallies, which is linear in the
    voters; share=0 is `first` and share=1 is `second`, and a ballot no one marks is
    not tallied (nor are its shares needed). The points of the two ballots are added up,
    so they must be on one scale, as those of two score ballots are:

        Mix(ScoreDH(2), ScoreAvg(2), share=0.25)

    is a quarter of the voters approving the candidates closer than their mean distance
    and three quarters those above their largest gap."""

    first: Ballot
    second: Ballot
    share: float

    def __post_init__(self):
        _expect(self, self.first, Ballot, "a Ballot")
        _expect(self, self.second, Ballot, "a Ballot")
        if not 0 <= self.share <= 1:
            raise ValueError(f"Mix share must be between 0 and 1, got {self.share!r}")

    @property
    def marked(self) -> tuple[Ballot, ...]:
        """The ballots some of the voters mark."""
        if self.share == 0:
            return (self.first,)
        if self.share == 1:
            return (self.second,)
        return (self.first, self.second)

    @property
    def needs(self) -> frozenset[Share]:
        return frozenset().union(*(ballot.needs for ballot in self.marked))

    @property
    def needs_remaining(self) -> frozenset[Share]:
        return frozenset().union(*(ballot.needs_remaining for ballot in self.marked))

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        tallies = [ballot.tally(voters, alive) for ballot in self.marked]
        if len(tallies) == 1:
            return tallies[0]
        return (1 - self.share) * tallies[0] + self.share * tallies[1]

    def __repr__(self):
        return f"Mix({self.first!r}, {self.second!r}, share={self.share!r})"

# ---------------------------------------------------------------- PairShares


class PairShares(Block):
    """For every pair of candidates, the share of the voters who rank one above the other."""

    kind = "PairShares"
    needs: frozenset[Share]

    def evaluate(self, voters: Voters) -> np.ndarray:
        """d (..., C, C): d[c, e] = share ranking c above e, d[c, c] = 0."""
        raise NotImplementedError


@dataclass(frozen=True)
class Pairwise(PairShares):
    """The pairwise shares of the voters, as they are."""

    needs = frozenset({PAIRWISE})

    def evaluate(self, voters: Voters) -> np.ndarray:
        return voters.pairwise

# ---------------------------------------------------------------- PairDiffs


class PairDiffs(Block):
    """How strongly each candidate beats each other one: s[c, e] = -s[e, c], and c beats
    e where s[c, e] > 0. `transitive` is True if the diffs cannot form a cycle, c beating
    e beating f and f beating c, which is what Unbeaten needs to elect someone always."""

    kind = "PairDiffs"
    needs: frozenset[Share]
    transitive: bool

    def evaluate(self, voters: Voters) -> np.ndarray:
        """s (..., C, C), s[c, c] = 0."""
        raise NotImplementedError

    def unbeaten(self, voters: Voters) -> Result | None:
        """Unbeaten(self) by a formula of the diffs' own, or None to decide on evaluate()."""
        return None


@dataclass(frozen=True)
class ScoreComparisons(PairDiffs):
    """Strict score-ballot preferences: voters giving both candidates the same score abstain."""

    ballot: Score
    transitive = False

    def __post_init__(self):
        _expect(self, self.ballot, Score, "a Score ballot")

    @property
    def needs(self) -> frozenset[Share]:
        return frozenset({ScoredPairwise(self.ballot.scored)})

    def evaluate(self, voters: Voters) -> np.ndarray:
        return voters.scored_pairwise[ScoredPairwise(self.ballot.scored)]

    def __repr__(self):
        return f"ScoreComparisons({self.ballot!r})"


@dataclass(frozen=True)
class Margins(PairDiffs):
    """d - d^T: how much more of the voters rank c above e than e above c. Not transitive,
    the majorities can go round in a cycle."""

    shares: PairShares
    transitive = False

    def __post_init__(self):
        _expect(self, self.shares, PairShares, "PairShares")

    @property
    def needs(self) -> frozenset[Share]:
        return self.shares.needs

    def evaluate(self, voters: Voters) -> np.ndarray:
        d = self.shares.evaluate(voters)
        return d - np.swapaxes(d, -1, -2)

    def __repr__(self):
        return f"Margins({self.shares!r})"


@dataclass(frozen=True)
class StrongestPaths(PairDiffs):
    """p - p^T, where p[c, e] is the strength of the strongest path from c to e over the
    diffs that say a candidate beats another (see _paths). Transitive: c beats e where
    p[c, e] > p[e, c], and that relation has no cycles.

    A candidate c who beats everyone directly is unbeaten on the paths as well: p[c, e] >=
    s[c, e] > 0, and no diff leads into c, so p[e, c] = 0. Unbeaten(self) therefore only
    computes the paths where there is no such candidate."""

    diffs: PairDiffs
    transitive = True

    def __post_init__(self):
        _expect(self, self.diffs, PairDiffs, "PairDiffs")

    @property
    def needs(self) -> frozenset[Share]:
        return self.diffs.needs

    def evaluate(self, voters: Voters) -> np.ndarray:
        return _paths(self.diffs.evaluate(voters))

    def unbeaten(self, voters: Voters) -> Result:
        """Unbeaten(self), with the paths only at the points where no candidate beats
        everyone directly. Elsewhere that candidate wins, with the margin of the diffs
        themselves: positive, at most the margin on the paths, and 0 on the border of
        that region."""
        s = self.diffs.evaluate(voters)
        winner, margin = _unbeaten(s, transitive=False)
        cycle = winner == CYCLE
        if cycle.any():
            winner[cycle], margin[cycle] = _unbeaten(_paths(s[cycle]), transitive=True)
        return winner, margin

    def __repr__(self):
        return f"StrongestPaths({self.diffs!r})"

# ---------------------------------------------------------------- CandidateTotals


class CandidateTotals(Block):
    """A total for each candidate at every point, higher is better."""

    kind = "CandidateTotals"
    needs: frozenset[Share]

    def evaluate(self, voters: Voters) -> np.ndarray:
        """CandidateTotals (..., C)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Tally(CandidateTotals):
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
        """CandidateTotals (..., C) among the candidates marked in alive (..., C), all for None."""
        return self.ballot.tally(voters, alive)

    def __repr__(self):
        return f"Tally({self.ballot!r})"


@dataclass(frozen=True)
class Weakest(CandidateTotals):
    """The weakest diff of a candidate, min_{e != c} s[c, e]: its narrowest win, or,
    if it loses somewhere, minus its worst defeat."""

    diffs: PairDiffs

    def __post_init__(self):
        _expect(self, self.diffs, PairDiffs, "PairDiffs")

    @property
    def needs(self) -> frozenset[Share]:
        return self.diffs.needs

    def evaluate(self, voters: Voters) -> np.ndarray:
        return _off_diagonal(self.diffs.evaluate(voters), np.inf).min(axis=-1)

    def __repr__(self):
        return f"Weakest({self.diffs!r})"

# ---------------------------------------------------------------- Winner


class Winner(Block):
    """Chooses candidates at every point, from the shares in `needs`; a voting method
    chooses one, its winner.

    select() gives the chosen candidates as a mask and the margin. `seats` is how many
    candidates the block chooses at most: 1 for a method, n for Highest(totals, n=n),
    None where that differs from point to point (a Union). No one is chosen in a
    Condorcet cycle, or in a duel that ties (a draw). evaluate() is the method's winner and
    margin, only for one seat. A block defines one of the two; the other follows."""

    kind = "a Winner"
    needs: frozenset[Share]
    seats = 1

    def select(self, voters: Voters) -> Result:
        """(chosen (..., C) bool, margin (...))."""
        winner, margin = self.evaluate(voters)
        return winner[..., None] == np.arange(voters.n_candidates), margin

    def evaluate(self, voters: Voters) -> Result:
        """(winner, margin), both of voters.shape: the chosen candidate, voting.CYCLE
        where no one is chosen."""
        self._one_seat()
        chosen, margin = self.select(voters)
        return np.where(chosen.any(axis=-1), chosen.argmax(axis=-1), CYCLE), margin

    def _one_seat(self):
        """ValueError unless the block chooses one candidate, a winner."""
        if self.seats != 1:
            seats = "a number of candidates" if self.seats is None else f"{self.seats} candidates"
            raise ValueError(f"{self!r} chooses {seats}, not a winner: let a method choose among them, "
                             f"Unbeaten(diffs, among=...)")

    def __or__(self, other):
        return Union(self, other)


def _highest(totals: np.ndarray, n: int) -> Result:
    """(chosen (..., C), margin) of the n highest totals (ties: the lowest index): the
    n-th highest total minus the next, inf with no more than n candidates."""
    if totals.shape[-1] <= n:
        return np.ones(totals.shape, dtype=bool), np.full(totals.shape[:-1], np.inf)
    chosen = np.zeros(totals.shape, dtype=bool)
    rest = np.array(totals)
    for _ in range(n):  # n argmax are faster than sorting a handful of candidates
        top = rest.argmax(axis=-1)[..., None]
        last = np.take_along_axis(rest, top, axis=-1)[..., 0]
        np.put_along_axis(chosen, top, True, axis=-1)
        np.put_along_axis(rest, top, -np.inf, axis=-1)
    return chosen, last - rest.max(axis=-1)


@dataclass(frozen=True)
class Eliminate(Winner):
    """Drops candidates round by round until `until` are left, by default one, the winner:

        how="min"   the lowest total goes (ties: the lowest index); the round's gap is
                    the second lowest total minus the lowest
        how="mean"  every total at most the mean of the remaining ones goes; if that is
                    all of them, all totals are equal and the lowest index stays. The
                    round's gap is the smallest distance of a total from the mean. A
                    round that would leave fewer than `until` keeps the `until` highest
                    instead, and the until-th highest minus the next is a gap too
        how="all"   all but the `until` highest go in one round (ties: the lowest index
                    stays); the gap is the until-th highest total minus the next, inf
                    if no more are left

    "min" and "mean" tally the remaining candidates again each round, so their totals are
    a Tally; "all" reads any totals once. The margin is the smallest gap of all rounds,
    like voting.irv_rounds: the candidates left change only where some round's
    elimination flips, and there the gap is 0. Highest(totals, n) is Eliminate(totals,
    how="all", until=n).
    """

    totals: CandidateTotals
    how: Literal["min", "mean", "all"]
    until: int = 1
    _until = "until"  # the name of `until` in error messages; not a field

    def __post_init__(self):
        name = type(self).__name__
        if self.how not in HOW:
            raise ValueError(f'{name} how must be "min", "mean" or "all", got {self.how!r}')
        if self.how == "all":
            _expect(self, self.totals, CandidateTotals, "CandidateTotals")
        else:
            _expect(self, self.totals, Tally, "a Tally (it tallies the remaining candidates again)")
        until = self.until
        if isinstance(until, bool) or not isinstance(until, int) or until < 1:
            raise ValueError(f"{name} {self._until} must be a whole number of at least 1, got {until!r}")

    @property
    def seats(self) -> int:
        return self.until

    @property
    def needs(self) -> frozenset[Share]:
        return self.totals.needs if self.how == "all" else self.totals.needs_remaining

    def evaluate(self, voters: Voters) -> Result:
        self._one_seat()
        if self.how == "all":
            return _top_two(self.totals.evaluate(voters))
        result = self.totals.ballot.eliminate(voters, self.how)
        return self.rounds(voters) if result is None else result

    def select(self, voters: Voters) -> Result:
        if self.until == 1:
            return super().select(voters)  # from evaluate, which may be the ballot's own formula
        if self.how == "all":
            return _highest(self.totals.evaluate(voters), self.until)
        return self.remaining(voters)

    def rounds(self, voters: Voters) -> Result:
        """evaluate() with one tally per round, whatever formula the ballot has of its
        own (Ballot.eliminate)."""
        alive, margin = self.remaining(voters)
        return alive.argmax(axis=-1), margin

    def remaining(self, voters: Voters) -> Result:
        """select() of "min" and "mean", with one tally per round."""
        n = voters.n_candidates
        alive = np.ones((*voters.shape, n), dtype=bool)
        margin = np.full(voters.shape, np.inf)
        drop = drop_lowest if self.how == "min" else drop_below_mean
        for _ in range(n - self.until):  # every round drops at least one candidate
            totals = self.totals.evaluate(voters, alive)
            gap = drop(totals, alive) if self.until == 1 else self._round(drop, totals, alive)
            np.minimum(margin, gap, out=margin)
            if self.how == "mean" and alive.sum(axis=-1).max() <= self.until:  # often done early
                break
        return alive, margin

    def _round(self, drop, totals: np.ndarray, alive: np.ndarray) -> np.ndarray:
        """drop() on alive (..., C) in place where more than `until` are left; where it
        leaves fewer, the `until` highest totals stay instead. Returns the gap."""
        before = alive.copy()
        done = before.sum(axis=-1) <= self.until
        gap = drop(totals, alive)
        alive[done], gap[done] = before[done], np.inf
        short = alive.sum(axis=-1) < self.until
        if short.any():
            kept, lead = _highest(np.where(before, totals, -np.inf)[short], self.until)
            alive[short], gap[short] = kept, np.minimum(gap[short], lead)
        return gap

    def __repr__(self):
        until = f", until={self.until!r}" if self.until != 1 else ""
        return f'Eliminate({self.totals!r}, how="{self.how}"{until})'


class Highest(Eliminate):
    """The n highest totals, by default one, the winner (ties: the lowest index). The
    margin is the n-th highest total minus the next: for one, the winner's lead over the
    second. Eliminate(totals, how="all", until=n) by its own name."""

    _until = "n"

    def __init__(self, totals: CandidateTotals, n: int = 1):
        super().__init__(totals, "all", n)

    @property
    def n(self) -> int:
        return self.until

    def __repr__(self):
        n = f", n={self.n!r}" if self.n != 1 else ""
        return f"Highest({self.totals!r}{n})"


@dataclass(frozen=True)
class Unbeaten(Winner):
    """The candidate no one beats, or the one beaten the least. beaten[e] = max_{f != e}
    s[f, e] is the strongest diff into e: negative if e beats everyone, positive if
    someone beats e. The winner has the lowest beaten (ties: the lowest index).

    Its margin is the smallest max(beaten[e], 0) over the other candidates e: how far
    the closest of them is from being unbeaten, which is when the winner would change.

    PairDiffs that are not transitive can form a cycle. Then the winner must also beat
    everyone (beaten < 0), or no one wins (voting.CYCLE), and since that flips where the
    winner's beaten crosses 0, |beaten| of the winner is a gap too: the margin is at
    most that. Transitive diffs leave it out: someone is always unbeaten, and on
    strongest paths the winner's beaten is 0 on whole areas, where the widest paths to
    and from another candidate share their weakest diff.

    With `among` only the candidates `among` chooses run, and only against each other
    (the diffs themselves, strongest paths too, are those of all candidates). Two of
    them are a duel: the one who beats the other wins. A tie is a draw, no one, on diffs
    that are not transitive, and the lowest index on transitive ones, as above. Where `among` chooses one, it wins; where no one, no one does. The
    margin is the smaller of among's and the one among them.

    With `against` and `order` (both or neither) only one candidate has to stay unbeaten,
    the king: the winner of `against`. Its challengers are the candidates c who beat it,
    s[c, king] > 0. Without challengers the king wins, otherwise the challenger with the
    highest `order` total does (ties: the lowest index), in one step: no one challenges
    that winner in turn. Where `against` elects no one, the point stays a voting.CYCLE
    with the margin of `against`. Elsewhere the margin is the smallest of three gaps:

        the margin of `against`             the king changes
        min_{c != king} |s[c, king]|        a candidate starts or stops being a challenger
        the best challenger's lead in       another challenger wins; inf with fewer than
        `order` over the second             two challengers
    """

    diffs: PairDiffs
    against: Winner | None = None
    order: CandidateTotals | None = None
    among: Winner | None = None

    def __post_init__(self):
        _expect(self, self.diffs, PairDiffs, "PairDiffs")
        if (self.against is None) != (self.order is None):
            raise ValueError("Unbeaten takes against and order together, or neither")
        if self.against is not None:
            if self.among is not None:
                raise ValueError("Unbeaten takes among or against, not both")
            _expect(self, self.against, Winner, "a Winner as against")
            _expect(self, self.order, CandidateTotals, "CandidateTotals as order")
            if self.against.seats != 1:
                raise ValueError(f"Unbeaten against must choose one candidate, the king, "
                                 f"not {self.against!r}")
        if self.among is not None:
            _expect(self, self.among, Winner, "a Winner as among")

    @property
    def needs(self) -> frozenset[Share]:
        needs = self.diffs.needs
        if self.against is not None:
            needs |= self.against.needs | self.order.needs
        if self.among is not None:
            needs |= self.among.needs
        return needs

    def evaluate(self, voters: Voters) -> Result:
        plain = self.against is None and self.among is None
        result = self.diffs.unbeaten(voters) if plain else None
        return self.decide(voters) if result is None else result

    def decide(self, voters: Voters) -> Result:
        """evaluate() on the diffs as they are, whatever formula they have of their own
        (PairDiffs.unbeaten)."""
        diffs = self.diffs.evaluate(voters)
        if self.against is not None:
            return _challenged(diffs, *self.against.evaluate(voters), self.order.evaluate(voters))
        if self.among is None:
            return _unbeaten(diffs, self.diffs.transitive)
        running, margin = self.among.select(voters)
        winner, gap = _unbeaten(diffs, self.diffs.transitive, running)
        return winner, np.minimum(margin, gap)

    def __repr__(self):
        args = [repr(self.diffs)]
        if self.against is not None:
            args += [f"against={self.against!r}", f"order={self.order!r}"]
        if self.among is not None:
            args.append(f"among={self.among!r}")
        return f"Unbeaten({', '.join(args)})"


@dataclass(frozen=True)
class Fallback(Winner):
    """The candidates `first` chooses, or where it chooses no one (a voting.CYCLE, a
    draw), those of `second`. The margin is first's where it chooses someone, and the
    smaller of the two where not: the choice changes where either one would."""

    first: Winner
    second: Winner

    def __post_init__(self):
        _expect(self, self.first, Winner, "a Winner")
        _expect(self, self.second, Winner, "a Winner")

    @property
    def seats(self) -> int | None:
        return self.first.seats if self.first.seats == self.second.seats else None

    @property
    def needs(self) -> frozenset[Share]:
        return self.first.needs | self.second.needs

    def evaluate(self, voters: Voters) -> Result:
        self._one_seat()
        winner, margin = self.first.evaluate(voters)
        cycle = winner == CYCLE
        if cycle.any():
            other, gap = self.second.evaluate(voters)
            winner = np.where(cycle, other, winner)
            margin = np.where(cycle, np.minimum(margin, gap), margin)
        return winner, margin

    def select(self, voters: Voters) -> Result:
        chosen, margin = self.first.select(voters)
        empty = ~chosen.any(axis=-1)
        if empty.any():
            other, gap = self.second.select(voters)
            chosen = np.where(empty[..., None], other, chosen)
            margin = np.where(empty, np.minimum(margin, gap), margin)
        return chosen, margin

    def __repr__(self):
        return f"Fallback({self.first!r}, {self.second!r})"


@dataclass(frozen=True)
class Union(Winner):
    """first | second: the candidates either one chooses, e.g. the winners of two methods
    as the finalists of a runoff, Unbeaten(diffs, among=koth | irv). The margin is the
    smaller of the two: the candidates change where either one's do."""

    first: Winner
    second: Winner
    seats = None  # one or two winners; not a field

    def __post_init__(self):
        _expect(self, self.first, Winner, "a Winner")
        _expect(self, self.second, Winner, "a Winner")

    @property
    def needs(self) -> frozenset[Share]:
        return self.first.needs | self.second.needs

    def select(self, voters: Voters) -> Result:
        first, margin = self.first.select(voters)
        second, gap = self.second.select(voters)
        return first | second, np.minimum(margin, gap)

    def __repr__(self):
        second = f"({self.second!r})" if isinstance(self.second, Union) else repr(self.second)
        return f"{self.first!r} | {second}"
