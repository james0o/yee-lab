"""The objective of the search: how far a method's winners are from the nearest candidate,
sincerely and where 30 % of the voters bury, over a set of configurations; lower is better.

    J = trimmed mean over configurations k of (SINCERE e_k(m) + BURIED b_k(m)) + NODE_COST size(m)

size(m) the complexity of m (grammar.complexity): its distinct blocks, a method used
twice (in a condition and in a branch) paid for once, but not the totals and unions every
method needs, and more for Conditionals nested in Conditionals.

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

import numpy as np

from yeelab.build import Burial, Highest, ScoreComparisons, Tally, Unbeaten, Winner
from yeelab.build.blocks import Block
from yeelab.build.strategic import POLL, burying
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Voters
from yeelab.margin import regions
from yeelab.margin.metrics import FOCUS_ROUND, _area_weights, focus_distance

from yeesearch.configs import Config
from yeesearch.grammar import D20, SCORE_BALLOTS, TIE_WEIGHTS, complexity, needs as grammar_needs

NEEDS = grammar_needs()  # the shares cached for every configuration

SHARE = 0.3  # of the voters who bury
SCENARIOS = ("sincere", "poll")  # "poll": burying against the frontrunners of Burial's default poll
SINCERE, BURIED = 0.25, 0.75  # the burial error matters most (2026-10-28; 0.4 and 0.6 before)
NODE_COST = 0.0001  # per block: 15 blocks cost 0.0015, a tenth of a typical J (0.0003 before 2026-10-28)
TRIM = 0.05  # of the configurations left out of J's mean at each end, for each method
PIXEL_MEDIAN = "geometric"  # where a point of the voter grid is drawn: as the web UI starts (Scorer)
REGION, WHOLE = 0.8, 0.2  # of the error around the candidates and on the whole diagram (2026-10-28; 1 and 0 before)
SCORER = f"{PIXEL_MEDIAN}-whole{WHOLE:g}"  # what the errors depend on: caches of errors are kept apart by it
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


def remembering(block, memo: Memo):
    """The block with every Winner in it (and itself, if one) going through `memo`, by its repr."""
    key = repr(block)
    changes = {item.name: remembering(value, memo) for item in fields(block)
               if isinstance(value := getattr(block, item.name), Block)}
    if changes:  # a copy with the fields set, as the constructors of some (Highest) do not take them all
        block = copy(block)
        for name, value in changes.items():
            object.__setattr__(block, name, value)
    return _Remembered(block, key, memo) if isinstance(block, Winner) else block


@dataclass
class Prepared:
    """A configuration on a grid: the cached voters, with every share of the grammar, and
    its scorer."""

    config: Config
    size: int
    voters: Voters
    scorer: Scorer

    @classmethod
    def of(cls, config: Config, size: int) -> "Prepared":
        voters = regions.voters(NEEDS, config.array, config.model, size)
        voters = replace(voters, ballots={key: MemoGrid(grid) for key, grid in voters.ballots.items()})
        return cls(config, size, voters, Scorer(config, size))

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


def errors(method, prepared: list[Prepared], share: float = SHARE) -> dict[str, list[float]]:
    """{scenario: [relative error on each configuration]}."""
    out = {name: [] for name in SCENARIOS}
    for p in prepared:
        for name, winner in p.winners(method, share).items():
            out[name].append(p.scorer.relative(winner))
    return out


def warm(prepared: list[Prepared]):
    """Scores a method of every way a score ballot is read, so that the MemoGrids of the
    configurations remember what burial moves before the first method with a time limit."""
    for ballot in SCORE_BALLOTS.values():
        for method in (Highest(Tally(ballot)), Highest(Tally(ballot, statistic="median")),
                       *(Unbeaten(ScoreComparisons(ballot, tie_weight=weight)) for weight in TIE_WEIGHTS)):
            errors(method, prepared)
    errors(D20, prepared)


def trimmed(values, trim: float = TRIM) -> np.ndarray:
    """The indices of the values but the int(trim n) lowest and as many highest (as
    scipy.stats.trim_mean cuts)."""
    values = np.asarray(values)
    cut = int(trim * len(values))
    return np.argsort(values, kind="stable")[cut:len(values) - cut]


def objective(method, scenario_errors: dict[str, list[float]]) -> dict:
    """J and its parts from errors(): the sincere and the burial error, means over the
    configurations J keeps (trimmed: those of the TRIM lowest and highest SINCERE e_k +
    BURIED b_k left out), so that J = SINCERE sincere + BURIED burial + NODE_COST size."""
    sincere = np.array(scenario_errors["sincere"])
    burial = np.array(scenario_errors["poll"])
    kept = trimmed(SINCERE * sincere + BURIED * burial)
    nodes = complexity(method)
    return {"J": float((SINCERE * sincere[kept] + BURIED * burial[kept]).mean() + NODE_COST * nodes),
            "sincere": float(sincere[kept].mean()), "burial": float(burial[kept].mean()), "size": nodes}
