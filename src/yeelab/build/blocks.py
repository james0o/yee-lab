"""The blocks voting methods are built from, like in Scratch but as constructors:

    Highest(Tally(BordaCount()))                    Borda count
    Eliminate(Tally(Plurality()), how="min")        instant runoff
    Unbeaten(StrongestPaths(Margins(Pairwise())))   Schulze

Every block has one type of output:

    Ballot           Plurality(), BordaCount(),         points one voter gives a candidate
                     Approval(threshold),
                     GapApproval()
    CandidateTotals  Tally(ballot), Weakest(diffs)      a total of each candidate at a point
    PairShares       Pairwise()                         share of the voters ranking c above e
    PairDiffs        Margins(shares),                   how strongly c beats e (antisymmetric)
                     StrongestPaths(diffs)
    Winner           Highest(totals), Unbeaten(diffs),  winner and margin at every point
                     Eliminate(totals, how=...),
                     Fallback(first, second),
                     Unbeaten(diffs, against=winner, order=totals),
                     Runoff(diffs, first, second)

A ranked ballot gives weight(k, C) points to the candidate at position k (0 = closest)
of C; both count only the remaining candidates, and a higher total is better. An
approval ballot gives one point to each candidate the voter approves, which depends on
how far the candidates are and not only on their order (yeelab.approval): Approval cuts
at a threshold, GapApproval at the largest gap. Each ballot also knows the cheapest
formula for its mean over the voters (Ballot.tally), from the shares in voters.Voters.
Tally averages them over the voters of a point.

A diff s[c, e] = -s[e, c] says that c beats e where it is positive. PairDiffs are
transitive (PairDiffs.transitive) if the candidates that beat each other cannot form a
cycle: Margins are not, StrongestPaths are.

Unbeaten(diffs) elects the candidate no one beats. Unbeaten(diffs, against=winner,
order=totals) only asks that of the winner of `against`, the king: if someone beats it,
the highest `order` total among those who do wins instead. King of the hill is

    Unbeaten(Margins(Pairwise()), against=fptp, order=Tally(Plurality()))

Runoff(diffs, first, second) puts the winners of two methods against each other: the
one who beats the other on the diffs wins.

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

from yeelab.approval import GAP, Cut
from yeelab.build.rounds import drop_below_mean, drop_lowest
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Approved, Share, Voters
from yeelab.voting import CYCLE, irv_rounds

Result = tuple[np.ndarray, np.ndarray]  # (winner, margin) at every point
HOW = ("min", "mean")  # what Eliminate drops each round


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


def _unbeaten(diffs: np.ndarray, transitive: bool) -> Result:
    """Winner and margin of Unbeaten for diffs s (..., C, C) (see there)."""
    n = diffs.shape[-1]
    beaten = _off_diagonal(diffs, -np.inf).max(axis=-2)  # [..., e] = max_{f != e} s[f, e]
    winner = beaten.argmin(axis=-1)
    others = np.where(np.arange(n) == winner[..., None], np.inf, np.maximum(beaten, 0.0))
    margin = others.min(axis=-1)
    if not transitive:
        best = beaten.min(axis=-1)  # the winner's: negative if it beats everyone
        margin = np.minimum(margin, np.abs(best))
        winner = np.where(best < 0, winner, CYCLE)
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
    (a ranked ballot, which has a weight), or by whether the voter approves it."""

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
        of every point, None all of them; the others get 0."""
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


def _approving(voters: Voters, cut: Cut, alive: np.ndarray | None) -> np.ndarray:
    """Tally of an approval ballot: the share of the voters who approve each candidate
    at `cut`. A voter marks the ballot once, among all candidates, so the remaining
    candidates of an elimination keep their shares."""
    approved = voters.approved[cut]
    return approved if alive is None else np.where(alive, approved, 0)


@dataclass(frozen=True)
class Approval(Ballot):
    """One point for every candidate the voter approves: those at least `threshold` of
    the way from the farthest candidate (0) to the closest (1), by squared distance
    (yeelab.approval). 1 approves only the closest candidate, like Plurality; the lower
    the threshold, the more are approved, but never the farthest."""

    threshold: float = 0.5

    def __post_init__(self):
        number = isinstance(self.threshold, (int, float)) and not isinstance(self.threshold, bool)
        if not (number and 0 < self.threshold <= 1):
            raise ValueError(f"Approval threshold must be above 0 and at most 1, got {self.threshold!r}")

    @property
    def needs(self) -> frozenset[Share]:
        return frozenset({Approved(self.threshold)})

    needs_remaining = needs

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        return _approving(voters, self.threshold, alive)


@dataclass(frozen=True)
class GapApproval(Ballot):
    """One point for every candidate the voter approves: those above the largest gap
    between two neighbours when the candidates are in order of squared distance
    (yeelab.approval). For three candidates that is Approval(0.5)."""

    needs = needs_remaining = frozenset({Approved(GAP)})

    def tally(self, voters: Voters, alive: np.ndarray | None) -> np.ndarray:
        return _approving(voters, GAP, alive)

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
    """A voting method: winner and margin at every point, from the shares in `needs`."""

    kind = "a Winner"
    needs: frozenset[Share]

    def evaluate(self, voters: Voters) -> Result:
        """(winner, margin), both of voters.shape."""
        raise NotImplementedError


@dataclass(frozen=True)
class Highest(Winner):
    """The highest total wins (ties: the lowest index); the margin is its lead over the
    second."""

    totals: CandidateTotals

    def __post_init__(self):
        _expect(self, self.totals, CandidateTotals, "CandidateTotals")

    @property
    def needs(self) -> frozenset[Share]:
        return self.totals.needs

    def evaluate(self, voters: Voters) -> Result:
        return _top_two(self.totals.evaluate(voters))

    def __repr__(self):
        return f"Highest({self.totals!r})"


@dataclass(frozen=True)
class Eliminate(Winner):
    """Drops candidates round by round, tallying the remaining ones again each round,
    until one is left:

        how="min"   the lowest total goes (ties: the lowest index); the round's gap is
                    the second lowest total minus the lowest
        how="mean"  every total at most the mean of the remaining ones goes; if that is
                    all of them, all totals are equal and the lowest index wins. The
                    round's gap is the smallest distance of a total from the mean

    The margin is the smallest gap of all rounds, like voting.irv_rounds: the winner
    changes only where some round's elimination flips, and there the gap is 0.
    """

    totals: Tally
    how: Literal["min", "mean"]

    def __post_init__(self):
        _expect(self, self.totals, Tally, "a Tally (it tallies the remaining candidates again)")
        if self.how not in HOW:
            raise ValueError(f'Eliminate how must be "min" or "mean", got {self.how!r}')

    @property
    def needs(self) -> frozenset[Share]:
        return self.totals.needs_remaining

    def evaluate(self, voters: Voters) -> Result:
        result = self.totals.ballot.eliminate(voters, self.how)
        return self.rounds(voters) if result is None else result

    def rounds(self, voters: Voters) -> Result:
        """evaluate() with one tally per round, whatever formula the ballot has of its
        own (Ballot.eliminate)."""
        n = voters.n_candidates
        alive = np.ones((*voters.shape, n), dtype=bool)
        margin = np.full(voters.shape, np.inf)
        drop = drop_lowest if self.how == "min" else drop_below_mean
        for _ in range(n - 1):  # every round drops at least one candidate
            gap = drop(self.totals.evaluate(voters, alive), alive)
            np.minimum(margin, gap, out=margin)
            if self.how == "mean" and alive.sum(axis=-1).max() == 1:  # often done early
                break
        return alive.argmax(axis=-1), margin

    def __repr__(self):
        return f'Eliminate({self.totals!r}, how="{self.how}")'


@dataclass(frozen=True)
class Unbeaten(Winner):
    """The candidate no one beats, or the one beaten the least. beaten[e] = max_{f != e}
    s[f, e] is the strongest diff into e: negative if e beats everyone, positive if
    someone beats e. The winner has the lowest beaten (ties: the lowest index).

    Its margin is the smallest max(beaten[e], 0) over the other candidates e: how far
    the closest of them is from being unbeaten, which is when the winner would change.

    PairDiffs that are not transitive can form a cycle. Then the winner must also beat
    everyone (beaten < 0), or the point is a voting.CYCLE, and since that flips where
    the winner's beaten crosses 0, |beaten| of the winner is a gap too: the margin is at
    most that. Transitive diffs leave it out: someone is always unbeaten, and on
    strongest paths the winner's beaten is 0 on whole areas, where the widest paths to
    and from another candidate share their weakest diff.

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

    def __post_init__(self):
        _expect(self, self.diffs, PairDiffs, "PairDiffs")
        if (self.against is None) != (self.order is None):
            raise ValueError("Unbeaten takes against and order together, or neither")
        if self.against is not None:
            _expect(self, self.against, Winner, "a Winner as against")
            _expect(self, self.order, CandidateTotals, "CandidateTotals as order")

    @property
    def needs(self) -> frozenset[Share]:
        if self.against is None:
            return self.diffs.needs
        return self.diffs.needs | self.against.needs | self.order.needs

    def evaluate(self, voters: Voters) -> Result:
        result = self.diffs.unbeaten(voters) if self.against is None else None
        return self.decide(voters) if result is None else result

    def decide(self, voters: Voters) -> Result:
        """evaluate() on the diffs as they are, whatever formula they have of their own
        (PairDiffs.unbeaten)."""
        diffs = self.diffs.evaluate(voters)
        if self.against is None:
            return _unbeaten(diffs, self.diffs.transitive)
        return _challenged(diffs, *self.against.evaluate(voters), self.order.evaluate(voters))

    def __repr__(self):
        if self.against is None:
            return f"Unbeaten({self.diffs!r})"
        return f"Unbeaten({self.diffs!r}, against={self.against!r}, order={self.order!r})"


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
    where it beats `first` on the diffs, s[second, first] > 0, otherwise `first` does.
    Two candidates cannot form a cycle, so not even diffs that are not transitive leave
    the duel open. Where both elect the same candidate, that candidate wins.

    The margin is the smallest of three gaps: the margins of `first` and of `second` (a
    finalist changes) and |s[first, second]| (the duel flips), which is left out where
    both are the same candidate. Where either one elects no one (voting.CYCLE) there is
    no duel either, and the point stays a CYCLE with the smaller of their two margins."""

    diffs: PairDiffs
    first: Winner
    second: Winner

    def __post_init__(self):
        _expect(self, self.diffs, PairDiffs, "PairDiffs")
        _expect(self, self.first, Winner, "a Winner")
        _expect(self, self.second, Winner, "a Winner")

    @property
    def needs(self) -> frozenset[Share]:
        return self.diffs.needs | self.first.needs | self.second.needs

    def evaluate(self, voters: Voters) -> Result:
        first, margin = self.first.evaluate(voters)
        second, gap = self.second.evaluate(voters)
        diffs = self.diffs.evaluate(voters)
        n = diffs.shape[-1]
        cycle = (first == CYCLE) | (second == CYCLE)
        a, b = np.where(cycle, 0, first), np.where(cycle, 0, second)  # a = b in a cycle: no duel
        pairs = diffs.reshape(*diffs.shape[:-2], n * n)
        duel = np.take_along_axis(pairs, (a * n + b)[..., None], axis=-1)[..., 0]  # s[first, second]
        margin = np.minimum(np.minimum(margin, gap), np.where(a == b, np.inf, np.abs(duel)))
        return np.where(cycle, CYCLE, np.where(duel < 0, second, first)), margin

    def __repr__(self):
        return f"Runoff({self.diffs!r}, {self.first!r}, {self.second!r})"
