"""The voters a built method is evaluated on: the shares it needs, at every point.

A method's `needs` names the shares it reads (blocks.py); margin/regions.py computes
only those at the points of its grid.

    first       first-choice shares (..., C)
    pairwise    d[..., c, e] = share ranking c above e, and half of those ranking them
                equal, d[..., c, c] = 0, (..., C, C): d[c, e] + d[e, c] = 1
    profile     the whole profile: rankings (R, C), best first, and their shares (..., R);
                with bullet (..., C), the shares of the voters who rank c first and all
                others equal below it (Voters.bulleted), the profile is the other voters
    unscored    {Scored(levels, rule, delta): mean part of the top score that the voters
                do not give each candidate (..., C)}, from 0 to 1, one entry per score
                ballot: its number of levels, its rule (score.RANGE, score.AVG,
                score.DHONDT, score.HYBRID, score.CLUSTER), the divisor of DHONDT and
                HYBRID, the mu and kappa of CLUSTER and the power of RANGE. The
                mean score is the top score, levels - 1, times 1 minus this. It is
                not kept itself: where nearly all voters give two candidates the top
                score, both mean scores are the top score to rounding, and only the
                parts not given tell them apart
    buried      {Buried(scored): b[..., c, e] = mean part of the top score that the voters
                who rank e above c give c on that score ballot (..., C, C)}: what c loses
                where those voters bury it (Voters.burying)
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from yeelab.score import DELTA, KAPPA, MU, POWER, RANGE, Rule


@dataclass(frozen=True)
class Scored:
    """The scores on a ballot with `levels` scores, given by `rule` (score.RANGE,
    score.AVG, score.DHONDT, score.HYBRID or score.CLUSTER), for DHONDT and HYBRID the
    divisor `delta`, for CLUSTER the cost of a split `mu` and the cohesion `kappa`, and
    for RANGE the `power` of the part of the way; the others do not use them."""

    levels: int
    rule: Rule = RANGE
    delta: float = DELTA
    mu: float = MU
    kappa: float = KAPPA
    power: float = POWER


@dataclass(frozen=True)
class ScoredPairwise:
    """Strict pairwise preferences induced by the scores on one cardinal ballot."""

    scored: Scored


@dataclass(frozen=True)
class Buried:
    """What each candidate loses on the score ballot `scored` where the voters who rank
    another above it bury it, giving it 0."""

    scored: Scored


Share = Literal["first", "pairwise", "profile"] | Scored | ScoredPairwise | Buried
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
    bullet: np.ndarray | None = None
    buried: dict[Buried, np.ndarray] | None = None

    def at(self, points: np.ndarray) -> "Voters":
        """The voters of the points marked in `points` (...,) bool, in a row (P,)."""
        def pick(shares):
            if shares is None:
                return None
            if isinstance(shares, dict):
                return {key: value[points] for key, value in shares.items()}
            return shares[points]

        return Voters(pick(self.first), pick(self.pairwise), self.rankings, pick(self.probs), pick(self.unscored),
                      pick(self.scored_pairwise), pick(self.bullet), pick(self.buried))

    def burying(self, share: float, a: int, b: int) -> "Voters":
        """These voters with `share` of them, at every point, burying whichever of the
        frontrunners a and b they rank lower: they rank it last, and on a score ballot
        give it 0. The rest, and the bullet ballots, which rank it among the last already,
        vote as before; share 0 is these voters themselves. No first choice changes: no one buries
        their first choice. With s the share:

            profile   the rankings keep (1 - s) of their shares, and each also comes
                      buried, with s of them
            pairwise  d + s (the shares of the buried rankings minus those of the
                      rankings, from the profile): only the pairs of a or b with a third
                      candidate change
            unscored  u[a] + s buried[a, b] and u[b] + s buried[b, a]: a burying voter
                      takes back what they gave the frontrunner they rank lower

        The pairwise shares need the profile and the score shares `buried`. Score
        comparisons are not buried: they are left out."""
        if share == 0:
            return self
        changed = {"scored_pairwise": None}  # not buried, so left out (blocks.Burial: no STAR)
        if self.probs is not None:
            rankings = np.asarray(self.rankings)
            position = np.argsort(rankings, axis=1)
            lower = np.where(position[:, a] > position[:, b], a, b)  # the one each ranking buries
            keep = rankings != lower[:, None]
            buried = np.column_stack([rankings[keep].reshape(len(rankings), -1), lower]).astype(rankings.dtype)
            changed["rankings"] = np.concatenate([rankings, buried])
            changed["probs"] = np.concatenate([(1 - share) * self.probs, share * self.probs], axis=-1)
        if self.pairwise is not None:
            if self.probs is None:
                raise ValueError("burial on pairwise shares needs the profile")
            moved = _prefers(changed["rankings"][len(rankings):]) - _prefers(rankings)
            change = (self.probs @ moved.astype(self.probs.dtype)).reshape(self.pairwise.shape)
            changed["pairwise"] = self.pairwise + share * change
        if self.unscored:
            unscored = {}
            for key, u in self.unscored.items():
                taken = self.buried.get(Buried(key)) if self.buried else None
                if taken is None:
                    raise ValueError(f"burial on score shares needs what burying takes back, {Buried(key)}")
                u = np.array(u)
                u[..., a] += share * taken[..., a, b]
                u[..., b] += share * taken[..., b, a]
                unscored[key] = u
            changed["unscored"] = unscored
        return replace(self, **changed)

    def bulleted(self, share: float) -> "Voters":
        """These voters with `share` of them, at every point, bullet voting: they rank
        their first choice first and all other candidates equal below it, and on a score
        ballot give it the top score and all others 0. The rest vote as before; share 0
        is these voters themselves. With f the first-choice shares, which do not change,
        and s the share:

            pairwise         (1 - s) d[c, e] + s (1 + f[c] - f[e]) / 2: a voter who ranks
                             c and e equal counts half for each
            profile          the rankings keep (1 - s) of their shares, and s of the
                             voters of each first choice go to bullet
            unscored         (1 - s) u[c] + s (the sum of f over the others): a bullet
                             voter gives c none of the top score unless c is their first
            scored_pairwise  (1 - s) x[c, e] + s (f[c] - f[e])

        The pairwise and score shares need `first`; the profile has its own."""
        if share == 0:
            return self
        changed = {}
        if self.probs is not None:
            n = self.rankings.shape[1]
            firsts = self.probs @ np.eye(n, dtype=self.probs.dtype)[self.rankings[:, 0]]
            changed["probs"] = (1 - share) * self.probs
            changed["bullet"] = share * firsts + (0.0 if self.bullet is None else self.bullet)
        if self.pairwise is not None or self.unscored or self.scored_pairwise:
            f = self.first
            if f is None:
                raise ValueError("bullet voting on pairwise or score shares needs the first-choice shares")
            n = f.shape[-1]
            if self.pairwise is not None:
                d = (1 - share) * self.pairwise + share * (1 + f[..., :, None] - f[..., None, :]) / 2
                d[..., range(n), range(n)] = 0.0
                changed["pairwise"] = d
            if self.unscored:  # a sum of the others, not 1 - f: exact where f is nearly 1
                others = np.stack([np.delete(f, c, axis=-1).sum(axis=-1) for c in range(n)], axis=-1)
                changed["unscored"] = {key: (1 - share) * u + share * others for key, u in self.unscored.items()}
            if self.scored_pairwise:
                lead = f[..., :, None] - f[..., None, :]
                changed["scored_pairwise"] = {key: (1 - share) * x + share * lead
                                              for key, x in self.scored_pairwise.items()}
        if self.buried:  # a bullet voter gives no one but their first choice a point
            changed["buried"] = {key: (1 - share) * b for key, b in self.buried.items()}
        return replace(self, **changed)

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


def _prefers(rankings: np.ndarray) -> np.ndarray:
    """P[r, c * C + e] = 1 where ranking r (R, C) ranks c above e, else 0, (R, C * C)."""
    position = np.argsort(rankings, axis=1)
    return (position[:, :, None] < position[:, None, :]).reshape(len(rankings), -1).astype(np.int8)
