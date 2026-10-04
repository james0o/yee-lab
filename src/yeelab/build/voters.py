"""The voters a built method is evaluated on: the shares it needs, at every point.

A method's `needs` names the shares it reads (blocks.py); margin/regions.py computes
only those at the points of its grid, the tests take them from a complete profile of
pixels/.

    first       first-choice shares (..., C)
    pairwise    d[..., c, e] = share ranking c above e, d[..., c, c] = 0, (..., C, C)
    profile     the whole profile: rankings (R, C), best first, and their shares (..., R)
    unapproved  {cut: share not approving each candidate (..., C)}, one entry per cut of
                the approval ballots (approval.HALF, approval.GAP, approval.AVG); a method names each
                as Approved(cut). The share approving is 1 minus this. It is not kept
                itself: where nearly all voters approve two candidates, both shares are
                1 to rounding, and only the shares not approving tell them apart
    unscored    {levels: mean part of the top score that the voters do not give each
                candidate (..., C)}, from 0 to 1, one entry per number of levels of the
                score ballots (yeelab.score); a method names each as Scored(levels). The
                mean score is the top score, levels - 1, times 1 minus this. Kept this way
                for the same reason as `unapproved`
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yeelab.approval import Cut


@dataclass(frozen=True)
class Approved:
    """The approval shares at one cut: approval.HALF, approval.GAP or approval.AVG."""

    cut: Cut


@dataclass(frozen=True)
class Scored:
    """The scores on a ballot with `levels` scores."""

    levels: int


Share = Literal["first", "pairwise", "profile"] | Approved | Scored
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
    unapproved: dict[Cut, np.ndarray] | None = None
    unscored: dict[int, np.ndarray] | None = None

    @property
    def _marked(self) -> np.ndarray:
        """The shares of one of the approval or score ballots, (..., C)."""
        shares = self.unapproved if self.unapproved is not None else self.unscored
        return next(iter(shares.values()))

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
