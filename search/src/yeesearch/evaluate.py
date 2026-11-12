"""The objective of the search: how far a method's winners are from the nearest candidate,
sincerely and where 30 % of the voters bury, over a set of configurations; lower is better.

    J = trimmed weighted mean over configurations k of (SINCERE e_k(m) + BURIED b_k(m)) + NODE_COST size(m)

the weight of a configuration 1 but for the configurations of the web UI the search is to
get right (configs.focus_configs), which are never trimmed either. On those a method must
bury no worse than a rival (Config.rival), a hard rule: with

    behind(m) = sum over them of max(0, b_k(m) / b_k(rival) - 1),

a method behind its rivals gets RIVAL_HARD + behind(m) added to J, so that it ranks below
every method that is not, and those behind among themselves by how far (the genetic
programming keeps a slope to climb); the report shows only those with behind(m) = 0.

size(m) the complexity of m (grammar.complexity): its distinct blocks, a method used
twice (in a condition and in a branch) paid for once, but not the totals and unions every
method needs, more for Conditionals nested in Conditionals, and one more for every among=
each time it occurs.

e_k is the error of the method over the error of a random winner, REGION of it that on
the focus region of the Yee diagram of k (around the candidates) and WHOLE of it that on
the whole diagram (yeelab.margin.metrics.voronoi_agreement with cycles="worst": a point
without a winner counts as won by the farthest candidate), each point of the voter grid
drawn at the geometric median of its voters as the web UI starts (PIXEL_MEDIAN), and b_k
that error where 30 % of the voters bury against the frontrunners of Burial's default
poll, the two with the most first choices: the burial of the web UI. The mean leaves out
the TRIM of the configurations the method does best on and the TRIM it does worst on, so
that a few configurations do not decide it. A method that fails or takes too long scores
inf (yeesearch.pool)."""

from collections import OrderedDict
from copy import copy
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from itertools import count

import numpy as np

from yeelab.build import Burial, Winner
from yeelab.build.blocks import Block, Condition
from yeelab.build.strategic import POLL, burying
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Ballots, Voters
from yeelab.margin import regions, shares
from yeelab.margin.metrics import FOCUS_ROUND, _area_weights, focus_distance

from yeesearch.configs import Config
from yeesearch.grammar import complexity, needs as grammar_needs

NEEDS = grammar_needs()  # the shares cached for every configuration
# a configuration is read on every score ballot of the grammar in turns: each one's table is
# built once if all of them are kept (yeelab keeps fewer, which built each several times)
shares.keep_ballots(max(shares.BALLOT_TABLES, 2 + sum(isinstance(share, Ballots) for share in NEEDS)))

SHARE = 0.3  # of the voters who bury
SCENARIOS = ("sincere", "poll")  # "poll": burying against the frontrunners of Burial's default poll
SINCERE, BURIED = 0.25, 0.75  # the burial error matters most (2026-10-28; 0.4 and 0.6 before)
NODE_COST = 0.0001  # per block: 15 blocks cost 0.0015, a tenth of a typical J (0.0003 before 2026-10-28)
TRIM = 0.05  # of the configurations left out of J's mean at each end, for each method
PIXEL_MEDIAN = "geometric"  # where a point of the voter grid is drawn: as the web UI starts (Scorer)
REGION, WHOLE = 0.8, 0.2  # of the error around the candidates and on the whole diagram (2026-10-28; 1 and 0 before)
SCORER = f"{PIXEL_MEDIAN}-whole{WHOLE:g}"  # what the errors depend on: caches of errors are kept apart by it
RIVAL_HARD = 1.0  # added to J of a method burying worse than a rival anywhere: more than any J of one that does not
RIVAL_TOLERANCE = 1e-9  # behind at most this is none: the same winners as the rival, up to rounding
MEMO_BYTES = 256 * 2**20  # of the results of subtrees a process remembers (Memo)


def scenarios(method, share: float = SHARE) -> dict:
    """{name: method as the voters vote} of SCENARIOS: sincerely, and with `share` of the
    voters burying against the frontrunners of the default poll."""
    return {"sincere": method, "poll": Burial(method, share)}


class Scorer:
    """The loss of every candidate at every point of the grid of one configuration, and
    what each point weighs, to score many methods' winners fast: relative(winner) is
    REGION voronoi_agreement(..., cycles="worst")["region"] error over random and WHOLE
    the same of "diagram", the sum over the points of the loss times `scale`."""

    def __init__(self, config: Config, size: int, pixel_median: str = PIXEL_MEDIAN, whole: float = WHOLE):
        """`pixel_median` "geometric": each point of the voter grid drawn at the geometric
        median of its voters, as the web UI starts (regions.geometric_grid), the points it
        does not keep weighing 0; "marginal": at the median of each coordinate, the
        library's default and the search's before 2026-10-28. `whole`: the part of the error
        on the whole diagram, the rest around the candidates."""
        candidates = config.array
        coords = regions.grid(size, config.model.pixels)[0]
        keep = np.ones(len(coords), dtype=bool)
        if pixel_median == "geometric":
            keep, points = regions.geometric_grid(config.model, size)
        else:
            points = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1)
        pairwise = np.linalg.norm(candidates[:, None] - candidates[None], axis=-1)[np.triu_indices(len(candidates), 1)]
        pairwise = pairwise[pairwise > 0]
        scale = float(np.median(pairwise)) if len(pairwise) else 1.0
        distances = np.linalg.norm(points[..., None, :] - candidates, axis=-1)
        regret = distances - distances.min(axis=-1, keepdims=True)
        weights = _area_weights(points)
        focus = weights * (focus_distance(points, candidates, config.deviation) <= FOCUS_ROUND * config.deviation)
        kept = np.ix_(keep, keep)
        self.every = np.zeros((len(coords), len(coords), len(candidates)))  # (G, G, C)
        self.every[kept] = regret / (regret + scale)
        self.worst = self.every.max(axis=-1)
        self.scale = np.zeros((len(coords), len(coords)))  # what each point's loss adds to relative()
        for part, area in ((1 - whole, focus if focus.sum() > 0 else weights), (whole, weights)):
            if part:
                spread = np.zeros((len(coords), len(coords)))
                spread[kept] = area / area.sum()
                self.scale += part * spread / float((self.every.mean(axis=-1) * spread).sum())

    def relative(self, winner: np.ndarray) -> float:
        chosen = np.take_along_axis(self.every, np.maximum(winner, 0)[..., None], axis=-1)[..., 0]
        loss = np.where(winner >= 0, chosen, self.worst)
        return float((loss * self.scale).sum())


def restricted(voters: Voters, needs: frozenset) -> Voters:
    """The voters with only the shares in `needs`: strategic voters move every share they
    have, so a method is evaluated fastest on the shares it reads."""
    def keep(shares):
        kept = {key: value for key, value in (shares or {}).items() if key in needs}
        return kept or None

    profile = PROFILE in needs
    ballots = keep(voters.ballots)
    return replace(voters, first=voters.first if FIRST in needs else None,
                   pairwise=voters.pairwise if PAIRWISE in needs else None,
                   rankings=voters.rankings if profile else None, probs=voters.probs if profile else None,
                   unscored=keep(voters.unscored), scored_pairwise=keep(voters.scored_pairwise),
                   graded=keep(voters.graded), ballots=ballots, cells=voters.cells if ballots else None)


class MemoGrid:
    """A score ballot on the voter grid (margin.shares.ScoreGrid) that remembers what it
    gives: the score shares that burying voters move depend on the configuration, the
    frontrunners and the points they are frontrunners at, which a poll names alike for
    every method, and not on the method; ScoreGrid computes them again on every call.
    Copies are handed out, since the callers may change them in place."""

    def __init__(self, grid):
        self.grid = grid
        self.memo = {}

    def _remembered(self, name: str, cells: np.ndarray, *args, **options):
        key = (name, args, tuple(sorted(options.items())), cells.shape, cells.tobytes())
        if key not in self.memo:
            self.memo[key] = getattr(self.grid, name)(*args, cells, **options)
        return self.memo[key].copy()

    def taken(self, a, b, cells, mask=None):
        return self._remembered("taken", cells, a, b, mask=mask) if mask is None else self.grid.taken(a, b, cells, mask)

    def added(self, a, b, cells, mask=None):
        return self._remembered("added", cells, a, b, mask=mask) if mask is None else self.grid.added(a, b, cells, mask)

    def moved_grades(self, a, b, cells, mask=None):
        if mask is not None:
            return self.grid.moved_grades(a, b, cells, mask)
        return self._remembered("moved_grades", cells, a, b)

    def moved(self, a, b, bury, compromise, cells, tie_weight=0.0, mask=None):
        if mask is not None:
            return self.grid.moved(a, b, bury, compromise, cells, tie_weight, mask)
        return self._remembered("moved", cells, a, b, bury, compromise, tie_weight=tie_weight)

    def __getattr__(self, name):
        return getattr(self.grid, name)


class Memo:
    """What the Winner subtrees of methods select and elect, by their repr and the voters
    they ran on (`context`: a configuration, sincerely or buried at one pair of frontrunners),
    so that a method shares the work of the subtrees it has in common with those before it:
    a genetic program's children keep most of their parents, an enumeration most of the
    smaller methods. The least recently used go once they take more than `limit` bytes.
    A child block runs on the voters its parent got (yeelab.build.blocks), so the context
    of the root names those of every subtree."""

    def __init__(self, limit: int = MEMO_BYTES):
        self.entries: OrderedDict = OrderedDict()
        self.limit, self.bytes, self.context = limit, 0, None
        self.hits = self.misses = 0

    def get(self, key, compute) -> tuple:
        full = (self.context, key)
        value = self.entries.get(full)
        if value is None:
            self.misses += 1
            value = tuple(np.array(part) for part in compute())
            self.entries[full] = value
            self.bytes += sum(part.nbytes for part in value)
            while self.bytes > self.limit and self.entries:
                self.bytes -= sum(part.nbytes for part in self.entries.popitem(last=False)[1])
        else:
            self.hits += 1
            self.entries.move_to_end(full)
        return tuple(part.copy() for part in value)  # the caller may change it in place


MEMO = Memo()  # of this process


@dataclass(frozen=True, eq=False)
class _Remembered(Winner):
    """A Winner whose select() and evaluate() go through a Memo."""

    inner: Winner
    key: str = field(repr=False)
    memo: Memo = field(repr=False)

    @property
    def seats(self):
        return self.inner.seats

    @property
    def needs(self):
        return self.inner.needs

    def select(self, voters):
        return self.memo.get(("select", self.key), lambda: self.inner.select(voters))

    def evaluate(self, voters):
        return self.memo.get(("evaluate", self.key), lambda: self.inner.evaluate(voters))

    def __repr__(self):
        return repr(self.inner)


@dataclass(frozen=True, eq=False)
class _RememberedCondition(Condition):
    """A Condition whose evaluate() goes through a Memo: CoarseTrim's has_winner sweeps
    every trim, and the children of a genetic program keep their parents' conditions."""

    inner: Condition
    key: str = field(repr=False)
    memo: Memo = field(repr=False)

    @property
    def needs(self):
        return self.inner.needs

    def evaluate(self, voters):
        return self.memo.get(("condition", self.key), lambda: self.inner.evaluate(voters))

    def __repr__(self):
        return repr(self.inner)


def remembering(block, memo: Memo):
    """The block with every Winner and Condition in it (and itself, if one) going through
    `memo`, by its repr."""
    key = repr(block)
    changes = {item.name: remembering(value, memo) for item in fields(block)
               if isinstance(value := getattr(block, item.name), Block)}
    if changes:  # a copy with the fields set, as the constructors of some (Highest) do not take them all
        block = copy(block)
        for name, value in changes.items():
            object.__setattr__(block, name, value)
    if isinstance(block, Winner):
        return _Remembered(block, key, memo)
    return _RememberedCondition(block, key, memo) if isinstance(block, Condition) else block


@dataclass
class Prepared:
    """A configuration on a grid: the cached voters, with every share of the grammar, and
    its scorer."""

    config: Config
    size: int
    voters: Voters
    scorer: Scorer
    _buried: dict = field(default_factory=dict, repr=False)  # of buried(): by the share

    @classmethod
    def of(cls, config: Config, size: int, bury: float | None = None) -> "Prepared":
        """With `bury`, the voters burying by that share too (buried), now, while the
        tables of the configuration's score ballots are still kept (shares.keep_ballots)."""
        voters = regions.voters(NEEDS, config.array, config.model, size)
        voters = replace(voters, ballots={key: MemoGrid(grid) for key, grid in voters.ballots.items()})
        prepared = cls(config, size, voters, Scorer(config, size))
        if bury is not None:
            prepared.buried(bury)
        return prepared

    def buried(self, share: float = SHARE) -> list[tuple[np.ndarray, Voters]]:
        """(points, voters) of every pair of frontrunners (pairs): the voters of its points,
        in a row, with every share of the grammar where `share` of them bury against the
        pair, the score comparisons moved too (settled). The same for every method, so a
        Batch evaluates the methods on them."""
        if share not in self._buried:
            self._buried[share] = [(at, settled(burying(self.voters.at(at), share, a, b)))
                                   for _, a, b, at in self.pairs]
        return self._buried[share]

    def winner(self, method) -> np.ndarray:
        """The method's winner at every point: on the cached voters where they have every
        share it reads (restricted), else on its own."""
        if method.needs <= NEEDS:
            return method.evaluate(restricted(self.voters, method.needs))[0]
        return regions.winners(method, self.config.array, self.config.model, self.size)[1]

    @cached_property
    def pairs(self) -> list[tuple[int, int, int, np.ndarray]]:
        """(key, a, b, points) of every pair of frontrunners a, b of Burial's default poll
        and the points it names them at: the same for every method."""
        chosen, _ = POLL.select(restricted(self.voters, POLL.needs))
        n = chosen.shape[-1]
        pairs = chosen @ (1 << np.arange(n))
        return [(int(pair), *np.flatnonzero((int(pair) >> np.arange(n)) & 1), pairs == pair)
                for pair in np.unique(pairs)]

    @cached_property
    def rival_burial(self) -> float | None:
        """The burial error of the configuration's rival (Config.rival), None without one."""
        if self.config.rival is None:
            return None
        from yeelab.build.methods import METHODS
        return self.scorer.relative(self.winners(METHODS[self.config.rival])["poll"])

    def winners(self, method, share: float = SHARE, memo: Memo = MEMO) -> dict[str, np.ndarray]:
        """{scenario: winner} of SCENARIOS, as winner() of each of scenarios(method, share)
        but remembering the subtrees (Memo): sincerely, and burying at every pair of
        frontrunners as Burial does."""
        variant = Burial(method, share)
        if not variant.needs <= NEEDS:
            return {name: self.winner(each) for name, each in scenarios(method, share).items()}
        remembered = remembering(method, memo)
        memo.context = (self.config.name, self.size, "sincere")
        sincere = remembered.evaluate(restricted(self.voters, method.needs))[0]
        voters = restricted(self.voters, variant.needs)
        buried = np.empty_like(sincere)
        for pair, a, b, at in self.pairs:
            memo.context = (self.config.name, self.size, share, pair)
            buried[at] = remembered.evaluate(burying(voters.at(at), share, a, b))[0]
        return {"sincere": sincere, "poll": buried}


def settled(voters: Voters) -> Voters:
    """Strategic voters with nothing left to move: their score comparisons moved now
    (Voters.compared, which moves them when a method asks), and without the ballots those
    came from."""
    compared = voters.scored_pairwise and {key: voters.compared(key) for key in voters.scored_pairwise}
    return replace(voters, scored_pairwise=compared, frontrunners=None, bury=0.0, compromise=0.0, ballots=None,
                   cells=None)


def in_a_row(voters: Voters) -> Voters:
    """The voters of a grid of points with the points in a row, (G, G, ...) to (G * G, ...),
    without the ballots of strategic voters."""
    points = len(voters.shape)

    def flat(values):
        if values is None:
            return None
        if isinstance(values, dict):
            return {key: flat(value) for key, value in values.items()}
        return values.reshape(-1, *values.shape[points:])

    return replace(voters, ballots=None, cells=None,
                   **{name: flat(getattr(voters, name))
                      for name in ("first", "pairwise", "probs", "unscored", "scored_pairwise", "graded")})


def joined(parts: list[Voters]) -> Voters:
    """Voters of points in a row, of the same candidates in number, one after another as the
    voters of one row of points. A method decides every point on its own, so it elects at
    each what it elects there alone. The rankings are those of any part, each once, and a
    part's voters have none of the others'."""
    def stacked(name):
        values = [getattr(part, name) for part in parts]
        if values[0] is None:
            return None
        if isinstance(values[0], dict):
            return {key: np.concatenate([value[key] for value in values]) for key in values[0]}
        return np.concatenate(values)

    changed = {name: stacked(name) for name in ("first", "pairwise", "unscored", "scored_pairwise", "graded")}
    if parts[0].probs is not None:
        column = {}  # of each ranking
        columns = [[column.setdefault(tuple(ranking), len(column)) for ranking in part.rankings.tolist()]
                   for part in parts]
        probs = np.zeros((sum(len(part.probs) for part in parts), len(column)), dtype=parts[0].probs.dtype)
        start = 0
        for part, own in zip(parts, columns):
            rows = probs[start:start + len(part.probs)]
            for k, c in enumerate(own):  # a ranking can be there twice: buried, it is another one's
                rows[:, c] += part.probs[:, k]
            start += len(part.probs)
        changed.update(rankings=np.array(list(column), dtype=parts[0].rankings.dtype), probs=probs)
    return replace(parts[0], **changed)


@dataclass
class _Group:
    """The configurations of a Batch with the same number of candidates: their voters in
    one row of points, sincere and buried, and of each configuration (index among the
    prepared, where its sincere points are, the shape of its grid, where its buried points
    are, the points of the grid those are of, in their order)."""

    sincere: Voters
    buried: Voters
    members: list[tuple[int, slice, tuple[int, ...], slice, np.ndarray]]


class Batch:
    """Prepared configurations as a few rows of points, to evaluate a method on all of
    them in a few calls: most of the time of a method on the small grids of the search is
    not the arithmetic but calling it, a hundred times for 20 configurations (each sincerely
    and at every pair of frontrunners). The configurations with the same number of
    candidates are one group (joined): its voters voting sincerely are one row, and those
    burying, at every pair of frontrunners of every configuration (Prepared.buried), another.
    A method decides each point on its own, so the winners are those of Prepared.winners."""

    _ids = count()
    _kept: dict = {}  # of(): by the configurations and the share

    def __init__(self, prepared: list[Prepared], share: float = SHARE):
        self.prepared, self.share, self.key = list(prepared), share, next(Batch._ids)
        by_size: dict[int, list[int]] = {}
        for index, p in enumerate(self.prepared):
            by_size.setdefault(p.config.n_candidates, []).append(index)
        self.groups = []
        for indices in by_size.values():
            sincere, buried, members = [], [], []
            start = buried_start = 0
            for index in indices:
                p = self.prepared[index]
                voters = in_a_row(p.voters)
                sincere.append(voters)
                parts = p.buried(share)
                buried += [part for _, part in parts]
                points = np.concatenate([np.flatnonzero(at) for at, _ in parts])
                members.append((index, slice(start, start + len(voters.first)), p.voters.shape,
                                slice(buried_start, buried_start + len(points)), points))
                start += len(voters.first)
                buried_start += len(points)
            self.groups.append(_Group(joined(sincere), joined(buried), members))
        for p in self.prepared:  # the rows hold them now
            p._buried.pop(share, None)

    @classmethod
    def of(cls, prepared: list[Prepared], share: float = SHARE) -> "Batch":
        """The batch of these configurations, made once (a few at most are kept)."""
        key = (tuple(id(p) for p in prepared), share)
        batch = cls._kept.get(key)
        if batch is None:
            while len(cls._kept) >= 4:
                del cls._kept[next(iter(cls._kept))]
            batch = cls._kept[key] = cls(prepared, share)  # holds the configurations, so their ids stay theirs
        return batch

    def winners(self, method, memo: Memo = MEMO) -> list[dict[str, np.ndarray]]:
        """{scenario: winner} of SCENARIOS on every configuration, as Prepared.winners gives
        them, remembering the subtrees (Memo) by the group."""
        remembered = remembering(method, memo)
        out: list[dict] = [{} for _ in self.prepared]
        for g, group in enumerate(self.groups):
            memo.context = ("batch", self.key, g, "sincere")
            sincere = remembered.evaluate(group.sincere)[0]
            memo.context = ("batch", self.key, g, self.share)
            buried = remembered.evaluate(group.buried)[0]
            for index, span, shape, buried_span, points in group.members:
                out[index]["sincere"] = sincere[span].reshape(shape)
                poll = np.empty(shape, dtype=buried.dtype)
                poll.ravel()[points] = buried[buried_span]
                out[index]["poll"] = poll
        return out


def errors(method, prepared: list[Prepared], share: float = SHARE) -> dict[str, list[float]]:
    """{scenario: [relative error on each configuration]}, "weights": [Config.weight of
    each] where one is not 1, and "rivals": [burial error of each one's rival, None
    without] where one has a rival (objective). The winners come from a Batch of the
    configurations where the cached voters have every share the method reads."""
    out = {name: [] for name in SCENARIOS}
    if Burial(method, share).needs <= NEEDS:
        found = Batch.of(prepared, share).winners(method)
    else:
        found = [p.winners(method, share) for p in prepared]
    for p, winners in zip(prepared, found):
        for name in SCENARIOS:
            out[name].append(p.scorer.relative(winners[name]))
    weights = [p.config.weight for p in prepared]
    if any(weight != 1 for weight in weights):
        out["weights"] = weights
    rivals = [p.rival_burial for p in prepared]
    if any(rival is not None for rival in rivals):
        out["rivals"] = rivals
    return out


def warm(prepared: list[Prepared]):
    """Makes the Batch of the configurations, with the voters of each burying at every pair
    of frontrunners, so that the first method with a time limit does not pay for it."""
    Batch.of(prepared)


def trimmed(values, trim: float = TRIM, weights=None) -> np.ndarray:
    """The indices of the values but the int(trim n) lowest and as many highest (as
    scipy.stats.trim_mean cuts); with `weights`, of the values of weight 1 only, those of
    another weight always kept."""
    values = np.asarray(values)
    plain = np.arange(len(values)) if weights is None else np.flatnonzero(np.asarray(weights) == 1)
    cut = int(trim * len(plain))
    kept = plain[np.argsort(values[plain], kind="stable")[cut:len(plain) - cut]]
    return kept if weights is None else np.sort(np.concatenate([kept, np.flatnonzero(np.asarray(weights) != 1)]))


def objective(method, scenario_errors: dict[str, list[float]]) -> dict:
    """J and its parts from errors(): the sincere and the burial error, means over the
    configurations J keeps (trimmed: those of the TRIM lowest and highest SINCERE e_k +
    BURIED b_k left out, of weight 1), each counting as its weight (Config.weight), so that
    J = SINCERE sincere + BURIED burial + NODE_COST size, and RIVAL_HARD + behind more for
    a method whose burial error is above a rival's (Config.rival): behind how far, 0
    without rivals."""
    sincere = np.array(scenario_errors["sincere"])
    burial = np.array(scenario_errors["poll"])
    weights = scenario_errors.get("weights")
    kept = trimmed(SINCERE * sincere + BURIED * burial, weights=weights)
    w = np.ones(len(sincere)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w[kept] / w[kept].sum()
    nodes = complexity(method)
    sincere_mean, burial_mean = float(w @ sincere[kept]), float(w @ burial[kept])
    behind = sum(max(0.0, b / rival - 1) for b, rival in zip(burial, scenario_errors.get("rivals") or [])
                 if rival is not None and rival > 0)
    behind = float(behind) if behind > RIVAL_TOLERANCE else 0.0
    J = SINCERE * sincere_mean + BURIED * burial_mean + NODE_COST * nodes + (RIVAL_HARD + behind if behind else 0.0)
    return {"J": J, "sincere": sincere_mean, "burial": burial_mean, "size": nodes, "behind": behind}
