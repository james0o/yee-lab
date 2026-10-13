"""Strategic voting: the methods of blocks.py where some of the voters vote strategically.
Each strategy is a block that wraps a method:

    Bullet(method, share=b)       b of the voters bullet vote: their first choice first
                                  and all other candidates equal below it, on a score
                                  ballot the top score for it and 0 for all others
    Burial(method, share=q)       q of the voters bury: of the two frontrunners a poll
                                  names at every point, they rank the one they like less
                                  last, and give it 0 on a score ballot
    Compromise(method, share=y)   y of the voters compromise: if neither frontrunner is
                                  their first choice, they rank the one they like more
                                  first, and give it the top score on a score ballot

The others vote sincerely, and share 0 is the method itself. The strategies combine:
Bullet(Burial(Compromise(method, y), q), b) is the method where b of the voters bullet
vote, and of the others q bury and y compromise, each whatever the other does, so q y of
them do both. Compromise goes innermost, so that every poll reads the sincere first
choices.

A strategy changes the voters, not the method: bulleted(), burying() and compromising()
give the shares (voters.Voters) of the voters of whom that share vote by it. Bullet
voting depends on the voter alone. Burial and compromise depend on the frontrunners,
which change from point to point, so the method runs on the points of each pair of
frontrunners apart, and the poll's margin is a gap too (_Frontrunners). Their score
shares come from the ballots of the voter grid, for each pair at its points only
(margin.shares.ScoreGrid). Runoff comparisons do not add up over the strategies like the
scores, so the strategies only note them, and Voters.compared counts all of them together
when the method reads them.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from yeelab.build.blocks import Highest, Plurality, Result, Tally, Winner, _expect
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Ballots, Scored, ScoredPairwise, Share, Voters

POLL = Highest(Tally(Plurality()), n=2)  # the default frontrunners: the two with the most first choices

# ---------------------------------------------------------------- Voters


def bulleted(voters: Voters, share: float) -> Voters:
    """The voters with `share` of them, at every point, bullet voting: they rank their
    first choice first and all other candidates equal below it, and on a score ballot
    give it the top score and all others 0. The rest vote as before; share 0 is the
    voters themselves. With f the first-choice shares, which do not change, and s the
    share:

        pairwise         (1 - s) d[c, e] + s (1 + f[c] - f[e]) / 2: a voter who ranks c
                         and e equal counts half for each
        profile          the rankings keep (1 - s) of their shares, and s of the voters
                         of each first choice go to bullet
        unscored         (1 - s) u[c] + s (the sum of f over the others): a bullet voter
                         gives c none of the top score unless c is their first
        scored_pairwise  (1 - s) x[c, e] + s (f[c] - f[e]): bullet voters also rank
                         every candidate below their first choice equal

    The pairwise and score shares need `first`; the profile has its own. The bullet
    voters neither bury nor compromise: `kept`, the part of the voters who may, shrinks."""
    if share == 0:
        return voters
    changed = {}
    if voters.probs is not None:
        n = voters.rankings.shape[1]
        firsts = voters.probs @ np.eye(n, dtype=voters.probs.dtype)[voters.rankings[:, 0]]
        changed["probs"] = (1 - share) * voters.probs
        changed["bullet"] = share * firsts + (0.0 if voters.bullet is None else voters.bullet)
    if voters.pairwise is not None or voters.unscored or voters.scored_pairwise:
        f = voters.first
        if f is None:
            raise ValueError("bullet voting on pairwise or score shares needs the first-choice shares")
        n = f.shape[-1]
        if voters.pairwise is not None:
            d = (1 - share) * voters.pairwise + share * (1 + f[..., :, None] - f[..., None, :]) / 2
            d[..., range(n), range(n)] = 0.0
            changed["pairwise"] = d
        if voters.unscored:  # a sum of the others, not 1 - f: exact where f is nearly 1
            others = np.stack([np.delete(f, c, axis=-1).sum(axis=-1) for c in range(n)], axis=-1)
            changed["unscored"] = {key: (1 - share) * u + share * others for key, u in voters.unscored.items()}
        if voters.scored_pairwise:
            lead = f[..., :, None] - f[..., None, :]
            changed["scored_pairwise"] = {key: (1 - share) * x + share * lead
                                          for key, x in voters.scored_pairwise.items()}
    changed["kept"] = voters.kept * (1 - share)
    return replace(voters, **changed)


def burying(voters: Voters, share: float, a: int, b: int) -> Voters:
    """The voters with `share` of them, at every point, burying whichever of the
    frontrunners a and b they rank lower: they rank it last, and on a score ballot give
    it 0. The rest, and the bullet ballots, which rank it among the last already, vote
    as before; share 0 is the voters themselves. No first choice changes: no one buries
    their first choice. With s the share:

        profile   the rankings keep (1 - s) of their shares, and each also comes
                  buried, with s of them
        pairwise  d + s (the shares of the buried rankings minus those of the rankings,
                  from the profile): only the pairs of a or b with a third candidate
                  change
        unscored  u[a] + s taken[a] and u[b] + s taken[b] (Voters.ballots): a burying
                  voter takes back what they gave the frontrunner they rank lower

    The pairwise shares need the profile and the score shares the ballots. The score
    comparisons are only noted to move (Voters.compared)."""
    if share == 0:
        return voters
    changed = _against(voters, a, b, bury=share) if voters.scored_pairwise else {}
    if voters.probs is not None:
        rankings = np.asarray(voters.rankings)
        position = np.argsort(rankings, axis=1)
        lower = np.where(position[:, a] > position[:, b], a, b)  # the one each ranking buries
        rest = rankings[rankings != lower[:, None]].reshape(len(rankings), -1)
        buried = np.column_stack([rest, lower]).astype(rankings.dtype)
        changed.update(_moved_profile(voters, share, rankings, buried))
    elif voters.pairwise is not None:
        raise ValueError("burial on pairwise shares needs the profile")
    if voters.unscored:
        changed["unscored"] = _moved_scores(voters, a, b, share, "taken")
    return replace(voters, **changed)


def compromising(voters: Voters, share: float, a: int, b: int) -> Voters:
    """The voters with `share` of them, at every point, compromising on whichever of the
    frontrunners a and b they rank higher, if neither is their first choice: they rank
    it first, and on a score ballot give it the top score. The rest of the ballot stays.
    The others, those with a or b first, and the bullet ballots vote as before; share 0
    is the voters themselves. With s the share:

        profile   the rankings keep (1 - s) of their shares, and each also comes
                  compromised, with s of them
        first     f + s (the first choices of the compromised rankings minus those of
                  the rankings, from the profile): s of the first choices of the others
                  go to a and b
        pairwise  likewise from the profile: only the pairs of a or b with a third
                  candidate change
        unscored  u[a] - s added[a] and u[b] - s added[b] (Voters.ballots): a
                  compromising voter gives the frontrunner they rank higher what they
                  did not

    The first-choice and pairwise shares need the profile: without it the first choices
    are left out, the pairwise shares cannot be. The score shares need the ballots; the
    score comparisons are only noted to move (Voters.compared). Burying and compromising
    move different candidates, so a voter can do both, in either order."""
    if share == 0:
        return voters
    changed = _against(voters, a, b, compromise=share) if voters.scored_pairwise else {}
    if voters.probs is not None:
        rankings = np.asarray(voters.rankings)
        position = np.argsort(rankings, axis=1)
        # the one each ranking compromises on, first already where a or b is: no change there
        upper = np.where(position[:, a] < position[:, b], a, b)
        rest = rankings[rankings != upper[:, None]].reshape(len(rankings), -1)
        raised = np.column_stack([upper, rest]).astype(rankings.dtype)
        changed.update(_moved_profile(voters, share, rankings, raised))
        if voters.first is not None:
            firsts = np.eye(rankings.shape[1], dtype=voters.probs.dtype)
            changed["first"] = voters.first + share * (voters.probs @ (firsts[raised[:, 0]] - firsts[rankings[:, 0]]))
    elif voters.pairwise is not None:
        raise ValueError("compromise on pairwise shares needs the profile")
    else:
        changed["first"] = None  # the profile tells who moves their first choice
    if voters.unscored:
        changed["unscored"] = _moved_scores(voters, a, b, share, "added")
    return replace(voters, **changed)


def _moved_profile(voters: Voters, share: float, rankings: np.ndarray, moved: np.ndarray) -> dict:
    """The profile where `share` of the voters of each ranking mark it `moved`, and the
    pairwise shares that follow: d + share (the shares of the moved rankings minus those
    of the rankings)."""
    changed = {"rankings": np.concatenate([rankings, moved]),
               "probs": np.concatenate([(1 - share) * voters.probs, share * voters.probs], axis=-1)}
    if voters.pairwise is not None:
        change = voters.probs @ (_prefers(moved) - _prefers(rankings)).astype(voters.probs.dtype)
        changed["pairwise"] = voters.pairwise + share * change.reshape(voters.pairwise.shape)
    return changed


def _moved_scores(voters: Voters, a: int, b: int, share: float, points: str) -> dict:
    """The part not given of every score ballot, moved by `share` of the voters who do
    not bullet vote: taken back by burying ("taken") or added by compromising
    ("added")."""
    unscored = {}
    for key, u in voters.unscored.items():
        change = getattr(voters.grid(key), points)(a, b, voters.cells.reshape(-1, 2)).reshape(*u.shape[:-1], 2)
        change *= share * voters.kept * (1 if points == "taken" else -1)
        u = np.array(u)
        u[..., a] += change[..., 0]
        u[..., b] += change[..., 1]
        unscored[key] = u
    return unscored


def _against(voters: Voters, a: int, b: int, bury: float = 0.0, compromise: float = 0.0) -> dict:
    """The strategies against the frontrunners a and b noted for the score comparisons,
    with `bury` and `compromise` more of the voters doing each."""
    if voters.frontrunners is not None and set(voters.frontrunners) != {a, b}:
        raise ValueError(f"strategies against frontrunners {voters.frontrunners} and {(a, b)} at once")
    return {"frontrunners": (min(a, b), max(a, b)), "bury": voters.bury + bury - voters.bury * bury,
            "compromise": voters.compromise + compromise - voters.compromise * compromise}


def _prefers(rankings: np.ndarray) -> np.ndarray:
    """P[r, c * C + e] = 1 where ranking r (R, C) ranks c above e, else 0, (R, C * C)."""
    position = np.argsort(rankings, axis=1)
    return (position[:, :, None] < position[:, None, :]).reshape(len(rankings), -1).astype(np.int8)

# ---------------------------------------------------------------- Blocks


@dataclass(frozen=True)
class Bullet(Winner):
    """`method` where `share` of the voters of every point bullet vote and the others
    vote sincerely (bulleted). A bullet voter still marks a complete ballot: their first
    choice first and all other candidates equal below it, and on a score ballot the top
    score for their first choice and 0 for all others. Where a method gives points by
    position, equal candidates share the mean of the points of the positions they take
    (BordaCount: (C - 2) / 2 each); in a pair they abstain, and in an elimination a
    ballot whose first choice is out counts equally for all the remaining candidates.
    Share 0 is the method itself, and at share 1 every method elects the candidate with
    the most first choices."""

    method: Winner
    share: float

    def __post_init__(self):
        _expect(self, self.method, Winner, "a Winner")
        if not 0 <= self.share <= 1:
            raise ValueError(f"Bullet share must be between 0 and 1, got {self.share!r}")

    @property
    def seats(self) -> int | None:
        return self.method.seats

    @property
    def needs(self) -> frozenset[Share]:
        """The method's, and the first-choice shares the bullet voters' pairwise and score
        shares come from; the profile has its own."""
        needs = self.method.needs
        if self.share > 0 and any(share == PAIRWISE or isinstance(share, Scored | ScoredPairwise)
                                  for share in needs):
            needs |= {FIRST}
        return needs

    def select(self, voters: Voters) -> Result:
        return self.method.select(bulleted(voters, self.share))

    def evaluate(self, voters: Voters) -> Result:
        return self.method.evaluate(bulleted(voters, self.share))

    def __repr__(self):
        return f"Bullet({self.method!r}, share={self.share!r})"


def _ballots(needs: frozenset[Share]) -> frozenset[Share]:
    """The ballots of every score ballot in `needs`, of its totals or comparisons: what the
    score shares of burying and compromising voters come from (Voters.ballots)."""
    return frozenset(Ballots(share) for share in needs if isinstance(share, Scored)) | frozenset(
        Ballots(share.scored) for share in needs if isinstance(share, ScoredPairwise))


@dataclass(frozen=True)
class _Frontrunners(Winner):
    """A strategy against the two frontrunners a poll names at every point (Burial,
    Compromise): `method` where `share` of the voters vote by it and the others
    sincerely. Who the voters vote against changes where the poll names other
    frontrunners, so the method runs on the points of each pair of frontrunners apart,
    and the poll's margin (for POLL the second most first choices minus the third) is a
    gap too: the method's margin is the smaller of the two. Share 0 is the method
    itself."""

    method: Winner
    share: float
    poll: Winner = field(default=POLL, kw_only=True)

    def __post_init__(self):
        name = type(self).__name__
        _expect(self, self.method, Winner, "a Winner")
        if not 0 <= self.share <= 1:
            raise ValueError(f"{name} share must be between 0 and 1, got {self.share!r}")
        _expect(self, self.poll, Winner, "a Winner as poll")
        if self.poll.seats != 2:
            raise ValueError(f"{name} poll must choose two candidates, the frontrunners, not {self.poll!r}")

    @property
    def seats(self) -> int | None:
        return self.method.seats

    def vote(self, voters: Voters, a: int, b: int) -> Voters:
        """The voters where `share` of them vote by the strategy against a and b."""
        raise NotImplementedError

    def select(self, voters: Voters) -> Result:
        return self._by_poll(voters, self.method.select)

    def evaluate(self, voters: Voters) -> Result:
        return self._by_poll(voters, self.method.evaluate)

    def _by_poll(self, voters: Voters, decide) -> Result:
        """decide() on the voters of the points of each pair of frontrunners, voting by
        the strategy against them."""
        if self.share == 0:
            return decide(voters)
        chosen, gap = self.poll.select(voters)
        n = voters.n_candidates
        pairs = chosen @ (1 << np.arange(n))
        results = None
        for pair in np.unique(pairs):
            at = pairs == pair
            a, b = np.flatnonzero((int(pair) >> np.arange(n)) & 1)
            part = decide(self.vote(voters.at(at), a, b))
            if results is None:
                results = [np.empty(pairs.shape + value.shape[1:], dtype=value.dtype) for value in part]
            for result, value in zip(results, part):
                result[at] = value
        first, margin = results
        return first, np.minimum(margin, gap)

    def __repr__(self):
        poll = f", poll={self.poll!r}" if self.poll != POLL else ""
        return f"{type(self).__name__}({self.method!r}, share={self.share!r}{poll})"


@dataclass(frozen=True, repr=False)
class Burial(_Frontrunners):
    """`method` where `share` of the voters of every point bury (burying): of the two
    frontrunners, they rank the one they rank lower last, and on a score ballot give it
    0. The rest of the ballot stays, so no first choice changes, nor the head-to-head of
    the two frontrunners."""

    @property
    def needs(self) -> frozenset[Share]:
        """The method's and the poll's, the profile the buried pairwise shares come from,
        and the ballots of every score ballot, which what burying takes back comes from."""
        needs = self.method.needs
        if self.share == 0:
            return needs
        if PAIRWISE in needs:
            needs |= {PROFILE}
        return needs | self.poll.needs | _ballots(needs)

    def vote(self, voters: Voters, a: int, b: int) -> Voters:
        return burying(voters, self.share, a, b)


@dataclass(frozen=True, repr=False)
class Compromise(_Frontrunners):
    """`method` where `share` of the voters of every point compromise (compromising): if
    neither frontrunner is their first choice, they rank the one they rank higher first,
    and on a score ballot give it the top score. The rest of the ballot stays, so the
    head-to-head of the two frontrunners does not change; the first choices do. The
    poll reads the first choices of the voters it gets: inside Burial or Bullet it gets
    the sincere ones, Bullet(Burial(Compromise(method, ...), ...), ...), and outside them
    it would poll the compromised ones. Where all voters compromise, every first choice
    is a frontrunner: FPTP, the two-round system and IRV elect the one of them who beats
    the other head to head."""

    @property
    def needs(self) -> frozenset[Share]:
        """The method's and the poll's, the profile the moved first-choice and pairwise
        shares come from, and the ballots of every score ballot, which what compromising
        adds comes from."""
        needs = self.method.needs
        if self.share == 0:
            return needs
        if needs & {FIRST, PAIRWISE, PROFILE}:
            needs |= {PROFILE}
        return needs | self.poll.needs | _ballots(needs)

    def vote(self, voters: Voters, a: int, b: int) -> Voters:
        return compromising(voters, self.share, a, b)
