"""The voters a built method is evaluated on: the shares it needs, at every point.

A method's `needs` names the shares it reads (blocks.py); margin/regions.py computes
only those at the points of its grid, the tests take them from a complete profile of
pixels/.

    first       first-choice shares (..., C)
    pairwise    d[..., c, e] = share ranking c above e, d[..., c, c] = 0, (..., C, C)
    profile     the whole profile: rankings (R, C), best first, and their shares (..., R)
    unscored    {Scored(levels, rule, delta): mean part of the top score that the voters
                do not give each candidate (..., C)}, from 0 to 1, one entry per score
                ballot: its number of levels, its rule (score.RANGE, score.AVG,
                score.DHONDT, score.HYBRID) and the divisor of DHONDT and HYBRID. The
                mean score is the top score, levels - 1, times 1 minus this. It is
                not kept itself: where nearly all voters give two candidates the top
                score, both mean scores are the top score to rounding, and only the
                parts not given tell them apart
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yeelab.score import DELTA, RANGE, Rule


@dataclass(frozen=True)
class Scored:
    """The scores on a ballot with `levels` scores, given by `rule` (score.RANGE,
    score.AVG, score.DHONDT or score.HYBRID) and, for DHONDT and HYBRID, the divisor
    `delta`; the others do not use it."""

    levels: int
    rule: Rule = RANGE
    delta: float = DELTA


Share = Literal["first", "pairwise", "profile"] | Scored
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

    @property
    def _marked(self) -> np.ndarray:
        """The shares of one of the score ballots, (..., C)."""
        return next(iter(self.unscored.values()))

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
