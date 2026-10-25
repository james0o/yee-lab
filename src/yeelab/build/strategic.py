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

The shares of every score that the median of a score ballot reads (Voters.graded,
Tally(ballot, statistic="median")) add up over the voters like the scores: a strategy
moves the voters it changes from their score of a candidate to 0 or the top score.

A strategy can be tilted, `tilt=u`: then the share of the voters at x who vote by it is

    s(x) = sigma(logit share + u . (x - CENTRE)),   CENTRE = (0.5, 0.5)

the share itself at the centre of the square, more of the voters in the direction of u
and fewer opposite it, as many across it; the odds of the two walls across u = (l, 0)
differ e^l times. The voters are cut into strips across u (Striped), each with the mean
of s over it, and the strategy moves the shares of each strip by its own share
(Voters.strips); the method reads their sums. The strategies of one method may be tilted
along one direction only.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from yeelab.build.blocks import Highest, Plurality, Result, Tally, Winner, _expect
from yeelab.build.voters import (FIRST, PAIRWISE, PROFILE, Ballots, Graded, Scored, ScoredPairwise, Share, Striped,
                                 Voters)

POLL = Highest(Tally(Plurality()), n=2)  # the default frontrunners: the two with the most first choices
NO_TILT = (0.0, 0.0)  # the same share of strategic voters everywhere


# ---------------------------------------------------------------- Voters


def bulleted(voters: Voters, share: float, tilt=NO_TILT) -> Voters:
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
        graded           (1 - s) g[c] + s (f[c] at the top score, the sum of f over the
                         others at 0)

    The pairwise and score shares need `first`; the profile has its own. The bullet
    voters neither bury nor compromise: `kept`, the part of the voters who may, shrinks.
    With a tilt, each strip of the voters by its own share (_by_strips)."""
    return _by_strips(voters, share, tilt, _bullet)


def _bullet(voters: Voters, share: float) -> Voters:
    """bulleted() of voters of one share; those of a strip have its mass where all have
    1 (the 1 of the pairwise shares)."""
    changed = {}
    if voters.probs is not None:
        n = voters.rankings.shape[1]
        firsts = voters.probs @ np.eye(n, dtype=voters.probs.dtype)[voters.rankings[:, 0]]
        changed["probs"] = (1 - share) * voters.probs
        changed["bullet"] = share * firsts + (0.0 if voters.bullet is None else voters.bullet)
    if voters.pairwise is not None or voters.unscored or voters.scored_pairwise or voters.graded:
        f = voters.first
        if f is None:
            raise ValueError("bullet voting on pairwise or score shares needs the first-choice shares")
        n = f.shape[-1]
        if voters.pairwise is not None:
            mass = np.asarray(voters.mass)[..., None, None]
            d = (1 - share) * voters.pairwise + share * (mass + f[..., :, None] - f[..., None, :]) / 2
            d[..., range(n), range(n)] = 0.0
            changed["pairwise"] = d
        # a sum of the others, not 1 - f: exact where f is nearly 1
        others = np.stack([np.delete(f, c, axis=-1).sum(axis=-1) for c in range(n)], axis=-1)
        if voters.unscored:
            changed["unscored"] = {key: (1 - share) * u + share * others for key, u in voters.unscored.items()}
        if voters.graded:
            changed["graded"] = {}
            for key, g in voters.graded.items():
                bullet = np.zeros_like(g)
                bullet[..., 0], bullet[..., -1] = others, f
                changed["graded"][key] = (1 - share) * g + share * bullet
        if voters.scored_pairwise:
            lead = f[..., :, None] - f[..., None, :]
            changed["scored_pairwise"] = {key: (1 - share) * x + share * lead
                                          for key, x in voters.scored_pairwise.items()}
    changed["kept"] = voters.kept * (1 - share)
    return replace(voters, **changed)


def burying(voters: Voters, share: float, a: int, b: int, tilt=NO_TILT) -> Voters:
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
        graded    of a, s of the voters who rank b above it go from their score to 0,
                  and likewise of b (Voters.ballots)

    The pairwise shares need the profile and the score shares the ballots. The score
    comparisons are only noted to move (Voters.compared). With a tilt, each strip of
    the voters by its own share (_by_strips)."""
    return _by_strips(voters, share, tilt, lambda part, s: _bury(part, s, a, b))


def _bury(voters: Voters, share: float, a: int, b: int) -> Voters:
    """burying() of voters of one share."""
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
    if voters.graded:
        changed["graded"] = _moved_grades(voters, a, b, share, bury=True)
    return replace(voters, **changed)


def compromising(voters: Voters, share: float, a: int, b: int, tilt=NO_TILT) -> Voters:
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
        graded    of a, s of the voters who rank it above b, neither first, go from
                  their score to the top score, and likewise of b (Voters.ballots)

    The first-choice and pairwise shares need the profile: without it the first choices
    are left out, the pairwise shares cannot be. The score shares need the ballots; the
    score comparisons are only noted to move (Voters.compared). Burying and compromising
    move different candidates, so a voter can do both, in either order. With a tilt,
    each strip of the voters by its own share (_by_strips)."""
    return _by_strips(voters, share, tilt, lambda part, s: _compromise(part, s, a, b))


def _compromise(voters: Voters, share: float, a: int, b: int) -> Voters:
    """compromising() of voters of one share."""
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
    if voters.graded:
        changed["graded"] = _moved_grades(voters, a, b, share, bury=False)
    return replace(voters, **changed)


def _by_strips(voters: Voters, share: float, tilt, vote) -> Voters:
    """vote(voters, share), the strategy at one share; for voters cut into strips
    (Voters.strips), vote(strip, its share of `share` and `tilt`) for every strip, and
    the voters with their sums. Share 0 is the voters themselves."""
    if share == 0:
        return voters
    if voters.strips is None:
        if any(tilt):
            raise ValueError(f"a strategy tilted by {tuple(tilt)} needs the voters cut into strips, {Striped.of(tilt)}")
        return vote(voters, share)
    return _summed(voters, tuple(vote(part, s) for part, s in zip(voters.strips, voters.striped.shares(share, tilt))))


def _summed(voters: Voters, parts: tuple[Voters, ...]) -> Voters:
    """The voters cut into the strips `parts`: every share the sum of theirs, and the
    frontrunners theirs. Each strip has some of the rankings of all voters, the rows
    `index` of them, and every ranking is in one strip at least."""
    def total(name):
        values = [getattr(part, name) for part in parts]
        if values[0] is None:
            return None
        if isinstance(values[0], dict):
            return {key: sum(value[key] for value in values) for key in values[0]}
        return sum(values)

    changed = {name: total(name) for name in ("first", "pairwise", "unscored", "scored_pairwise", "graded", "bullet")}
    if parts[0].probs is not None:
        rows = 1 + max(int(part.index.max()) for part in parts)
        rankings = np.empty((rows, parts[0].rankings.shape[1]), dtype=parts[0].rankings.dtype)
        probs = np.zeros((*parts[0].probs.shape[:-1], rows), dtype=parts[0].probs.dtype)
        for part in parts:
            rankings[part.index] = part.rankings
            probs[..., part.index] += part.probs
        changed.update(rankings=rankings, probs=probs)
    return replace(voters, strips=parts, frontrunners=parts[0].frontrunners, **changed)


def _moved_grades(voters: Voters, a: int, b: int, share: float, bury: bool) -> dict:
    """The shares of every score of every score ballot, where `share` of the voters who
    do not bullet vote, of those that burying (bury) or compromising moves, go from
    their score of a and of b to 0, or to the top score (ScoreGrid.moved_grades).
    Burying and compromising move a candidate's score for different voters, so each
    takes those it moves out of what the other left; a share is never below 0."""
    graded = {}
    for key, g in voters.graded.items():
        moved = voters.grid(key.scored).moved_grades(a, b, voters.cells.reshape(-1, 2))[0 if bury else 1]
        moved = moved.reshape(*g.shape[:-2], 2, g.shape[-1]) * (share * voters.kept)
        g = np.array(g)
        for side, c in enumerate((a, b)):
            g[..., c, :] = np.maximum(g[..., c, :] - moved[..., side, :], 0.0)
            g[..., c, 0 if bury else -1] += moved[..., side, :].sum(axis=-1)
        graded[key] = g
    return graded


def _moved_profile(voters: Voters, share: float, rankings: np.ndarray, moved: np.ndarray) -> dict:
    """The profile where `share` of the voters of each ranking mark it `moved`, and the
    pairwise shares that follow: d + share (the shares of the moved rankings minus those
    of the rankings). Of the voters of a strip, row r of the rankings of all voters is
    row 2 r of theirs after, and its moved ranking row 2 r + 1 (_summed)."""
    changed = {"rankings": np.concatenate([rankings, moved]),
               "probs": np.concatenate([(1 - share) * voters.probs, share * voters.probs], axis=-1)}
    if voters.index is not None:
        changed["index"] = np.concatenate([2 * voters.index, 2 * voters.index + 1])
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
    the most first choices. With a `tilt`, `share` is that of the voters at the centre of
    the square (bulleted, Striped)."""

    method: Winner
    share: float
    tilt: tuple[float, float] = field(default=NO_TILT, kw_only=True)

    def __post_init__(self):
        _expect(self, self.method, Winner, "a Winner")
        if not 0 <= self.share <= 1:
            raise ValueError(f"Bullet share must be between 0 and 1, got {self.share!r}")
        _check_tilt(self)

    @property
    def seats(self) -> int | None:
        return self.method.seats

    @property
    def needs(self) -> frozenset[Share]:
        """The method's, and the first-choice shares the bullet voters' pairwise and score
        shares come from; the profile has its own."""
        needs = self.method.needs
        if self.share > 0 and any(share == PAIRWISE or isinstance(share, Scored | ScoredPairwise | Graded)
                                  for share in needs):
            needs |= {FIRST}
        return _striped(needs, self)

    def select(self, voters: Voters) -> Result:
        return self.method.select(bulleted(voters, self.share, self.tilt))

    def evaluate(self, voters: Voters) -> Result:
        return self.method.evaluate(bulleted(voters, self.share, self.tilt))

    def __repr__(self):
        return f"Bullet({self.method!r}, share={self.share!r}{_tilt_repr(self)})"


def _check_tilt(block):
    """The tilt of a strategy block as two floats, or ValueError; and its strips one
    direction with those of the strategies inside it (_striped)."""
    name = type(block).__name__
    try:
        tilt = tuple(float(value) for value in block.tilt)
    except (TypeError, ValueError):
        tilt = ()
    if len(tilt) != 2 or not np.isfinite(tilt).all():
        raise ValueError(f"{name} tilt must be two finite numbers, got {block.tilt!r}")
    object.__setattr__(block, "tilt", tilt)
    block.needs  # noqa: B018 -- raises for strategies tilted in different directions


def _striped(needs: frozenset[Share], block) -> frozenset[Share]:
    """`needs`, and the strips of the voters across the tilt of `block` where it is
    tilted (Striped); all the strategies of a method must be tilted along one direction."""
    if block.share > 0 and any(block.tilt):
        needs |= {Striped.of(block.tilt)}
    striped = sorted(share.angle for share in needs if isinstance(share, Striped))
    if len(striped) > 1:
        raise ValueError(f"strategies tilted in different directions at once, at angles {striped}: "
                         "the voters are cut into strips across one direction only")
    return needs


def _tilt_repr(block) -> str:
    return f", tilt={block.tilt!r}" if any(block.tilt) else ""


def _ballots(needs: frozenset[Share]) -> frozenset[Share]:
    """The ballots of every score ballot in `needs`, of its totals, comparisons or shares
    of every score: what the score shares of burying and compromising voters come from
    (Voters.ballots)."""
    return frozenset(Ballots(share) for share in needs if isinstance(share, Scored)) | frozenset(
        Ballots(share.scored) for share in needs if isinstance(share, ScoredPairwise | Graded))


@dataclass(frozen=True)
class _Frontrunners(Winner):
    """A strategy against the two frontrunners a poll names at every point (Burial,
    Compromise): `method` where `share` of the voters vote by it and the others
    sincerely. Who the voters vote against changes where the poll names other
    frontrunners, so the method runs on the points of each pair of frontrunners apart,
    and the poll's margin (for POLL the second most first choices minus the third) is a
    gap too: the method's margin is the smaller of the two. Share 0 is the method
    itself. With a `tilt`, `share` is that of the voters at the centre of the square
    (burying, compromising, Striped)."""

    method: Winner
    share: float
    poll: Winner = field(default=POLL, kw_only=True)
    tilt: tuple[float, float] = field(default=NO_TILT, kw_only=True)

    def __post_init__(self):
        name = type(self).__name__
        _expect(self, self.method, Winner, "a Winner")
        if not 0 <= self.share <= 1:
            raise ValueError(f"{name} share must be between 0 and 1, got {self.share!r}")
        _expect(self, self.poll, Winner, "a Winner as poll")
        if self.poll.seats != 2:
            raise ValueError(f"{name} poll must choose two candidates, the frontrunners, not {self.poll!r}")
        _check_tilt(self)

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
        return f"{type(self).__name__}({self.method!r}, share={self.share!r}{poll}{_tilt_repr(self)})"


@dataclass(frozen=True, repr=False)
class Burial(_Frontrunners):
    """`method` where `share` of the voters of every point bury (burying): of the two
    frontrunners, they rank the one they rank lower last, and on a score ballot give it
    0. The rest of the ballot stays, so no first choice changes, nor the head-to-head of
    the two frontrunners."""

    @property
    def needs(self) -> frozenset[Share]:
        """The method's and the poll's, the profile the buried pairwise shares come from,
        and the ballots of every score ballot, which what burying takes back comes from:
        the voters bury all their shares, the poll's too."""
        needs = self.method.needs
        if self.share == 0:
            return needs
        needs |= self.poll.needs
        if PAIRWISE in needs:
            needs |= {PROFILE}
        return _striped(needs | _ballots(needs), self)

    def vote(self, voters: Voters, a: int, b: int) -> Voters:
        return burying(voters, self.share, a, b, self.tilt)


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
        adds comes from: the voters compromise in all their shares, the poll's too, but
        the first choices only the poll reads it has read already (compromising)."""
        needs = self.method.needs
        if self.share == 0:
            return needs
        if needs & {FIRST, PAIRWISE, PROFILE} or PAIRWISE in self.poll.needs:
            needs |= {PROFILE}
        needs |= self.poll.needs
        return _striped(needs | _ballots(needs), self)

    def vote(self, voters: Voters, a: int, b: int) -> Voters:
        return compromising(voters, self.share, a, b, self.tilt)
