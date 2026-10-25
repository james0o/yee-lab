"""The voters a built method is evaluated on: the shares it needs, at every point.

A method's `needs` names the shares it reads (blocks.py); margin/regions.py computes
only those at the points of its grid.

    first       first-choice shares (..., C)
    pairwise    d[..., c, e] = share ranking c above e, and half of those ranking them
                equal, d[..., c, c] = 0, (..., C, C): d[c, e] + d[e, c] = 1
    profile     the whole profile: rankings (R, C), best first, and their shares (..., R);
                with bullet (..., C), the shares of the voters who rank c first and all
                others equal below it (strategic.bulleted), the profile is the other voters
    unscored    {Scored(levels, rule, delta): mean part of the top score that the voters
                do not give each candidate (..., C)}, from 0 to 1, one entry per score
                ballot: its number of levels, its rule (score.RANGE, score.AVG,
                score.DHONDT, score.HYBRID, score.CLUSTER, score.D21), the divisor of
                DHONDT and HYBRID, the mu and kappa of CLUSTER, the power of RANGE and
                D21 and the numbers of votes of D21. The
                mean score is the top score, levels - 1, times 1 minus this. It is
                not kept itself: where nearly all voters give two candidates the top
                score, both mean scores are the top score to rounding, and only the
                parts not given tell them apart
    graded      {Graded(scored): g[..., c, k] = share of the voters who give c the score
                k, (..., C, levels)}: what the median of a score ballot reads
                (Tally(ballot, statistic="median"), judgment.py). The shares of every
                score are kept, not their sums, so that the few voters of a tail stay
                exact as in unscored
    ballots     {Ballots(scored): the score ballot on the voter grid, at the medians
                (margin.shares.ScoreGrid)}, and cells (..., 2), the indices of each point's
                medians: what the score shares of burying and compromising voters come
                from, for the frontrunners of the points (strategic.burying,
                compromising, Voters.compared). The grid has, for points `cells` (P, 2) and frontrunners a, b:
                    taken(a, b, cells)   (P, 2): the part of the top score that the
                        voters who rank b above a give a, and likewise for b
                    added(a, b, cells)   (P, 2): the part of the top score that the
                        voters who rank a above b, but neither first, do not give a,
                        and likewise for b
                    moved(a, b, bury, compromise, cells)   (P, C, C): how much the strict
                        score comparisons move where those shares of the voters bury
                        and compromise, each whatever the other does
    strips      (Striped(angle)): the voters cut into STRIPS strips across a direction,
                each with every other share of its own voters (strips, a Voters each,
                whose shares add up to these, and mass (...,), the share of the voters
                in the strip): what a strategy tilted along that direction reads, where
                the share of the voters who vote by it changes from strip to strip
                (strategic.py)
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.special import logit

STRIPS = 8  # strips of the voters across the direction of a tilt (Striped)
CENTRE = np.array([0.5, 0.5])  # where a tilt leaves the share of strategic voters as it is

from yeelab.score import DELTA, KAPPA, MU, POWER, RANGE, VOTES, Rule


@dataclass(frozen=True)
class Scored:
    """The scores on a ballot with `levels` scores, given by `rule` (score.RANGE,
    score.AVG, score.DHONDT, score.HYBRID, score.CLUSTER or score.D21), for DHONDT and
    HYBRID the divisor `delta`, for CLUSTER the cost of a split `mu` and the cohesion
    `kappa`, for RANGE and D21 the `power` of the part of the way, and for D21 its
    numbers of votes `votes`; the others do not use them."""

    levels: int
    rule: Rule = RANGE
    delta: float = DELTA
    mu: float = MU
    kappa: float = KAPPA
    power: float = POWER
    votes: tuple[int, int, int] = VOTES


@dataclass(frozen=True)
class ScoredPairwise:
    """Pairwise preferences induced by the scores on one cardinal ballot.

    When `tie_weight` is nonzero, the ballot's ordinal preference counts by that share
    only where the pair has equal scores.
    """

    scored: Scored
    tie_weight: float = 0.0

    def __post_init__(self):
        if (isinstance(self.tie_weight, bool) or not isinstance(self.tie_weight, int | float)
                or not np.isfinite(self.tie_weight) or not 0 <= self.tie_weight <= 1):
            raise ValueError(f"Score comparison tie weight must be between 0 and 1, got {self.tie_weight!r}")


@dataclass(frozen=True)
class Ballots:
    """The ballots of the score ballot `scored` on the voter grid: the score shares of
    burying and compromising voters come from them, for the frontrunners of each point."""

    scored: Scored


@dataclass(frozen=True)
class Graded:
    """The share of the voters who give each candidate each score of the score ballot
    `scored`: what its median reads (Tally(ballot, statistic="median"))."""

    scored: Scored


@dataclass(frozen=True)
class Striped:
    """The voters cut into STRIPS strips across the direction at `angle` (radians, from 0
    to pi: a direction and its opposite cut the same strips), each with the shares of its
    own voters: what a strategy tilted along the direction reads (strategic.py). Along
    the direction u, t = u . (x - CENTRE) runs over the unit square from -h to h, h =
    (|cos angle| + |sin angle|) / 2, and the strips cut it into equal parts; the first
    and the last reach beyond the square too, for normal voters. Use Striped.of(tilt)."""

    angle: float

    @staticmethod
    def of(tilt) -> "Striped":
        """The strips across the direction of `tilt`, a nonzero vector: parallel tilts
        have the same strips."""
        angle = round(math.atan2(tilt[1], tilt[0]) % math.pi, 12)
        return Striped(0.0 if angle == round(math.pi, 12) else angle)

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    @property
    def bounds(self) -> np.ndarray:
        """t from -h to h at the borders of the strips, (STRIPS + 1,)."""
        h = (abs(math.cos(self.angle)) + abs(math.sin(self.angle))) / 2
        return np.linspace(-h, h, STRIPS + 1)

    def shares(self, share: float, tilt) -> np.ndarray:
        """The share of the voters of each strip who vote by a strategy of `share` and
        `tilt`, (STRIPS,): the mean over the strip's t of sigma(logit share + l t), l =
        tilt . u, from the logistic's integral (softplus); share itself where the tilt is
        0 or share is 0 or 1. The tilt must be along the strips."""
        u = self.direction
        along = float(np.dot(tilt, u))
        if math.hypot(*(np.asarray(tilt, dtype=np.float64) - along * u)) > 1e-9 * max(1.0, math.hypot(*tilt)):
            raise ValueError(f"tilt {tuple(tilt)} is not along the strips of {self}: "
                             "strategies tilted in different directions at once")
        if along == 0 or not 0 < share < 1:
            return np.full(STRIPS, float(share))
        t, a = self.bounds, logit(share)
        return np.diff(np.logaddexp(0.0, a + along * t)) / (along * np.diff(t))


Share = Literal["first", "pairwise", "profile"] | Scored | ScoredPairwise | Ballots | Graded | Striped
FIRST: Share = "first"
PAIRWISE: Share = "pairwise"
PROFILE: Share = "profile"


@dataclass(frozen=True, eq=False)
class Voters:
    """The shares of the voters at every point, shape (...); None where not computed."""

    first: np.ndarray | None = None
    pairwise: np.ndarray | None = None
    rankings: np.ndarray | None = None
    probs: np.ndarray | None = None
    unscored: dict[Scored, np.ndarray] | None = None
    scored_pairwise: dict[ScoredPairwise, np.ndarray] | None = None
    graded: dict[Graded, np.ndarray] | None = None
    bullet: np.ndarray | None = None
    ballots: dict[Ballots, object] | None = None
    cells: np.ndarray | None = None
    # the strategies against these frontrunners that the score comparisons are not moved by
    # yet (Voters.compared): the shares of the voters who bury and who compromise, each
    # whatever the other does, of the part `kept` of the voters, those who do not bullet
    # vote (strategic.py)
    frontrunners: tuple[int, int] | None = None
    bury: float = 0.0
    compromise: float = 0.0
    kept: float = 1.0
    # the voters of each strip (Striped), whose shares add up to these, and the share of
    # the voters these are of: 1, all of them, or a strip's (...,); a strip's rankings
    # are only those of its voters, the rows `index` of the rankings of all of them
    strips: tuple["Voters", ...] | None = None
    striped: Striped | None = None
    mass: np.ndarray | float = 1.0
    index: np.ndarray | None = None

    def at(self, points: np.ndarray) -> "Voters":
        """The voters of the points marked in `points` (...,) bool, in a row (P,)."""
        def pick(shares):
            if shares is None or isinstance(shares, float):
                return shares
            if isinstance(shares, dict):
                return {key: value[points] for key, value in shares.items()}
            return shares[points]

        strips = None if self.strips is None else tuple(part.at(points) for part in self.strips)
        return replace(self, strips=strips, **{name: pick(getattr(self, name))
                                               for name in ("first", "pairwise", "probs", "unscored",
                                                            "scored_pairwise", "graded", "bullet", "cells", "mass")})

    def compared(self, key: ScoredPairwise) -> np.ndarray:
        """The pairwise preferences of `key`, moved by burying and compromising voters
        against the frontrunners (strategic.burying, compromising), which they only note:
        unlike the scores, the comparisons do not add up over the strategies, so they are
        computed for all of them together from the ballots, once they are known. Voters
        cut into strips add up those of their strips, each moved by its own shares."""
        comparisons = self.scored_pairwise[key]
        if self.frontrunners is None:
            return comparisons
        if self.strips is not None:
            return sum(part.compared(key) for part in self.strips)
        grid = self.grid(key.scored)
        if key.tie_weight:
            moved = grid.moved(*self.frontrunners, self.bury, self.compromise, self.cells.reshape(-1, 2),
                               key.tie_weight)
        else:
            moved = grid.moved(*self.frontrunners, self.bury, self.compromise, self.cells.reshape(-1, 2))
        return comparisons + self.kept * moved.reshape(comparisons.shape)

    def grid(self, scored: Scored):
        """The score ballot on the voter grid (Ballots)."""
        grid = self.ballots.get(Ballots(scored)) if self.ballots else None
        if grid is None or self.cells is None:
            raise ValueError(f"the score shares of strategic voters need the ballots, {Ballots(scored)}")
        return grid

    @property
    def _marked(self) -> np.ndarray:
        """The shares of one of the score ballots, (..., C): the part not given, the share
        of the top score, or how the first candidate compares with each."""
        if self.unscored:
            return next(iter(self.unscored.values()))
        if self.graded:
            return next(iter(self.graded.values()))[..., -1]
        return next(iter(self.scored_pairwise.values()))[..., 0, :]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the points."""
        if self.first is not None:
            return self.first.shape[:-1]
        if self.pairwise is not None:
            return self.pairwise.shape[:-2]
        if self.probs is not None:
            return self.probs.shape[:-1]
        return self._marked.shape[:-1]

    @property
    def n_candidates(self) -> int:
        if self.first is not None:
            return self.first.shape[-1]
        if self.pairwise is not None:
            return self.pairwise.shape[-1]
        if self.rankings is not None:
            return self.rankings.shape[1]
        return self._marked.shape[-1]
