"""The voters a built method is evaluated on: the shares it needs, at every point.

A method's `needs` names the shares it reads (blocks.py); margin/regions.py computes
only those and interpolates them to its grid, the tests take them from a complete
profile of pixels/.

    first       first-choice shares (..., C)
    pairwise    d[..., c, e] = share ranking c above e, d[..., c, c] = 0, (..., C, C)
    profile     the whole profile: rankings (R, C), best first, and their shares (..., R)
    approved    {cut: share approving each candidate (..., C)}, one entry per cut of the
                approval ballots (approval.HALF, approval.GAP); a method names each as
                Approved(cut)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from yeelab.approval import Cut


@dataclass(frozen=True)
class Approved:
    """The approval shares at one cut: approval.HALF or approval.GAP."""

    cut: Cut


Share = Literal["first", "pairwise", "profile"] | Approved
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
    approved: dict[Cut, np.ndarray] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the points."""
        if self.first is not None:
            return self.first.shape[:-1]
        if self.pairwise is not None:
            return self.pairwise.shape[:-2]
        if self.probs is not None:
            return self.probs.shape[:-1]
        return next(iter(self.approved.values())).shape[:-1]

    @property
    def n_candidates(self) -> int:
        if self.first is not None:
            return self.first.shape[-1]
        if self.pairwise is not None:
            return self.pairwise.shape[-1]
        if self.rankings is not None:
            return self.rankings.shape[1]
        return next(iter(self.approved.values())).shape[-1]
