"""The search of Conditionals over the library of primitive methods (yeesearch.primitives):
Conditional(c, A, B) for every condition c and every two primitives A != B, scored from
their maps without running a block, then a level deeper for the best of them.

The loss of a method at a point depends only on its winner there, so with l_X[p] the part
of J that the winner of X at the point p adds (p a point of a configuration, sincerely or
buried: SINCERE or BURIED over the number of configurations, times the weight of the
point in the focus region over the error of a random winner, times the loss of the
winner, evaluate.Scorer),

    J(Conditional(c, A, B)) = NODE_COST (1 + |c|) + f_c(A) + g_c(B)
    f_c(A) = sum_p c_p l_A[p] + NODE_COST |A|,  g_c(B) = sum_p (1 - c_p) l_B[p] + NODE_COST |B|

where |X| is the number of blocks. The best two primitives of a condition are the best A
of f_c and the best B of g_c, and the sums c . l_A of all conditions with all primitives
are a matrix product: every condition with every pair at once. The conditions, each true
or false at every point, sincerely and buried:

    elects  X elects someone (not a cycle), X a primitive
    above   Top or Lead of a Tally of the mean > t, at every threshold of the grammar
    same    X == Y, X and Y primitives; X != Y is the same with A and B swapped
    and     two of the best `above` and `same` conditions, each maybe negated (below,
            differ): with A and B swapped that is every & and | of them

The same products rank the Conditionals without the cost of their size too ("raw"), to
see whether a Conditional is better at all or only too large; and the primitives alone
are ranked too ("alone"), the best of them being new methods themselves. Two levels of
Conditionals of the best conditions, A1, A2 and B free, are searched apart (Nested): the
J of Conditional(c1, A1, Conditional(c2, A2, B)) is three sums apart as well.

J's mean is trimmed (evaluate.objective), which no sum over points is: the search sums
over the configurations of middle difficulty (Search) and scores its best exactly.

A method is a tree: a primitive's source, or ("if", condition, method, method); a
condition ("elects", method), ("above" | "below", measure, threshold), ("same" |
"differ", method, method) or ("and", condition, condition), the measure an index of
primitives.MEASURES. Many Conditionals fit the training configurations by chance, so
the best of the training are scored on the validation set, and the best of those on the
test set by running their blocks (yeesearch.report)."""

import hashlib
import json
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from yeelab.build import Conditional
from yeelab.build.blocks import CombinedCondition, Threshold, WinnerComparison

from yeesearch.evaluate import BURIED, NODE_COST, PIXEL_MEDIAN, SINCERE, Scorer, objective, trimmed
from yeesearch.grammar import NEST, THRESHOLDS, complexity, size
from yeesearch.pool import parse
from yeesearch.primitives import MEASURES, MapStore, enumerated, found, known, side

console = Console(width=150)
TRAIN_SET, TRAIN_SIZE = "wide", 16
VALIDATION_SET, VALIDATION_SIZE = "validation", 24
COSTS = {"J": NODE_COST, "raw": 0.0}  # the rankings of Conditionals: by J, and without the cost of the size
RANKINGS = COSTS | {"alone": NODE_COST, "nested": NODE_COST, "nested_raw": 0.0}  # and the primitives alone, Nested
# methods whose agreements with others go into Nested's conditions: burial moves neither fptp's first choices nor
# two_round's head-to-head of the frontrunners, and D20 is weak alone but may tell where others are right
AGREEING = ("fptp", "two_round", "d20")
NESTED_TOP, NESTED_EACH = 1400, 200  # conditions of Nested: the best of all, and of the agreements with each AGREEING
BATCH = 1024  # conditions in one matrix product
BEST = 6  # of f and of g of a condition, whose pairs are tried
KEEP = 4  # pairs of a condition kept
POOL = 50_000  # the best Conditionals kept over all conditions, of each ranking
BASELINES = ("star", "star_two_round", "mj", "search_1", "search_2", "search_3", "search_4", "d20")
EXTRA = ()  # methods of yeelab.build.methods outside the grammar taken as primitives (D20 is in it since 2026-10-28)

# ---------------------------------------------------------------- trees


def block(tree):
    """The yeelab block of a method or condition tree."""
    if isinstance(tree, str):
        return parse(tree)
    kind = tree[0]
    if kind == "if":
        return Conditional(block(tree[1]), block(tree[2]), block(tree[3]))
    if kind == "elects":
        return block(tree[1])
    if kind in ("above", "below"):
        return Threshold(MEASURES[tree[1]], tree[2], above=kind == "above")
    if kind in ("same", "differ"):
        return WinnerComparison(block(tree[1]), block(tree[2]), equal=kind == "same")
    if kind == "and":
        return CombinedCondition(block(tree[1]), block(tree[2]), "and")
    raise ValueError(f"not a tree: {tree!r}")


def leaves(tree) -> list[str]:
    """The primitives a tree reads."""
    if isinstance(tree, str):
        return [tree]
    return [leaf for part in tree[1:] if isinstance(part, str | tuple) for leaf in leaves(part)]


def negated(condition):
    """The condition true where `condition` is false (`below` is `above`'s but where the
    measure equals the threshold)."""
    flip = {"above": "below", "below": "above", "same": "differ", "differ": "same"}
    return (flip[condition[0]], *condition[1:])


def as_tree(value):
    """A tree back from JSON, where its tuples are lists."""
    return tuple(as_tree(part) for part in value) if isinstance(value, list) else value

# ---------------------------------------------------------------- a set's maps and losses


class Space:
    """The maps of methods on the configurations of a set: of the `items` (trees),
    (P, 2, K, G, G) int8, sincerely and buried, and of MEASURES; and what each winner
    loses at every point (evaluate.Scorer), to score any tree of them."""

    def __init__(self, store: MapStore, items: list, maps: np.ndarray | None = None,
                 pixel_median: str = PIXEL_MEDIAN):
        """`pixel_median`: where each point of the voter grid is drawn, which the maps do
        not depend on (evaluate.Scorer)."""
        self.store, self.configs, self.side = store, store.configs, side(store.grid)
        self.pixel_median = pixel_median
        self.items = list(items)
        self.index = {item: k for k, item in enumerate(self.items)}
        if maps is None:
            maps, ok = store.stacked(self.items)
            if not ok.all():
                raise ValueError(f"maps failed or missing: {[s for s, o in zip(self.items, ok) if not o][:3]}")
        self.maps = maps
        self.measures = store.measures()
        scorers = [Scorer(config, store.grid, pixel_median) for config in self.configs]
        width = max(config.n_candidates for config in self.configs) + 1
        self.cycle = width - 1  # the column of the loss of a cycle
        unit = np.zeros((len(scorers), self.side, self.side, width))
        for k, scorer in enumerate(scorers):
            unit[k, ..., :scorer.every.shape[-1]] = scorer.every
            unit[k, ..., -1] = scorer.worst
            unit[k] *= scorer.scale[..., None]
        self.unit = unit  # the relative error each winner adds at each point (Scorer.relative)

    def extended(self, trees: list) -> "Space":
        """The space with more items, trees of its items."""
        new = [tree for tree in trees if tree not in self.index]
        maps = np.concatenate([self.maps, np.array([self.map_of(tree) for tree in new]).reshape(
            -1, *self.maps.shape[1:])])
        return Space(self.store, self.items + new, maps, self.pixel_median)

    def losses(self, maps: np.ndarray) -> np.ndarray:
        """What the winners of maps (..., 2, K, G, G) add to the relative error of their
        configuration at every point, (..., 2, K, G, G)."""
        index = np.where(maps < 0, self.cycle, maps).astype(np.intp)
        k, i, j = np.ogrid[:len(self.configs), :self.side, :self.side]
        return self.unit[k, i, j, index]

    def errors(self, maps: np.ndarray) -> np.ndarray:
        """The relative errors (..., 2, K) of maps (..., 2, K, G, G): Scorer.relative of
        each scenario and configuration."""
        return self.losses(maps).sum(axis=(-2, -1))

    @cached_property
    def sizes(self) -> np.ndarray:
        return np.array([complexity(block(item)) for item in self.items])

    def map_of(self, tree) -> np.ndarray:
        """The winners (2, K, G, G) of a method tree."""
        if tree in self.index:
            return self.maps[self.index[tree]]
        if isinstance(tree, str):
            raise KeyError(f"no maps of {tree}")
        return np.where(self.mask_of(tree[1]), self.map_of(tree[2]), self.map_of(tree[3]))

    def mask_of(self, condition) -> np.ndarray:
        """Where (2, K, G, G) a condition tree is true."""
        kind = condition[0]
        if kind == "elects":
            return self.map_of(condition[1]) >= 0
        if kind in ("above", "below"):
            value = self.measures[condition[1]]
            return value > condition[2] if kind == "above" else value < condition[2]
        if kind in ("same", "differ"):
            same = self.map_of(condition[1]) == self.map_of(condition[2])
            return same if kind == "same" else ~same
        if kind == "and":
            return self.mask_of(condition[1]) & self.mask_of(condition[2])
        raise ValueError(f"not a condition: {condition!r}")

    def score(self, tree) -> dict:
        """evaluate.objective() of a method tree from the maps, with its errors on every
        configuration."""
        errors = self.errors(self.map_of(tree))
        result = objective(block(tree), {"sincere": list(errors[0]), "poll": list(errors[1])})
        return result | {"errors": {"sincere": errors[0].round(6).tolist(), "poll": errors[1].round(6).tolist()}}

    def behaviour(self, tree) -> bytes:
        return hashlib.blake2b(self.map_of(tree).tobytes(), digest_size=16).digest()

# ---------------------------------------------------------------- the library


def library(runs: Path, max_size: int) -> list[str]:
    """The sources of the primitives: the grammar enumerated, the known methods (and
    EXTRA, which the grammar cannot build), the best of the GP runs."""
    from yeelab.build.methods import METHODS

    extra = [repr(METHODS[name]) for name in EXTRA]
    return list(dict.fromkeys(enumerated(max_size) + list(known()) + extra + found(runs)))


def distinct(store: MapStore, sources: list[str]) -> tuple[list[str], dict[str, str]]:
    """The sources that compute on every configuration, of those with the same maps
    (sincere and buried, everywhere) only the smallest; and {source left out: the one
    kept in its place}."""
    maps, ok = store.stacked(sources)
    sizes = [complexity(parse(s)) for s in sources]
    order = sorted((k for k in range(len(sources)) if ok[k]), key=lambda k: (sizes[k], len(sources[k]), sources[k]))
    kept, by_key, alias = [], {}, {}
    for k in order:
        key = hashlib.blake2b(maps[k].tobytes(), digest_size=16).digest()
        if key in by_key:
            alias[sources[k]] = by_key[key]
        else:
            by_key[key] = sources[k]
            kept.append(sources[k])
    return kept, alias

# ---------------------------------------------------------------- the search


@dataclass
class Pool:
    """The best Conditionals of a ranking: J (float32, before the exact score), the
    condition (an index of the conditions searched) and the items A and B."""

    J: np.ndarray
    condition: np.ndarray
    a: np.ndarray
    b: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))

    def merged(self, other: "Pool", limit: int = POOL) -> "Pool":
        joined = Pool(*(np.concatenate([getattr(self, name), getattr(other, name)]) for name in ("J", "condition",
                                                                                               "a", "b")))
        if len(joined.J) > limit:
            keep = np.argpartition(joined.J, limit)[:limit]
            joined = Pool(joined.J[keep], joined.condition[keep], joined.a[keep], joined.b[keep])
        return joined

    def sorted(self) -> "Pool":
        order = np.argsort(self.J, kind="stable")
        return Pool(self.J[order], self.condition[order], self.a[order], self.b[order])


class Search:
    """The items of a Space as the A and B of Conditionals, on the points where they do
    not all lose alike (active): elsewhere every Conditional adds the same, `constant`.

    J's trimmed mean leaves out other configurations for every method, which no sum over
    the points can; the search sums over the configurations kept by their difficulty, the
    median over the items of SINCERE e_k + BURIED b_k (a method's best and worst
    configurations are mostly the easiest and the hardest for all), and the best of it
    are scored exactly (Space.score)."""

    def __init__(self, space: Space):
        self.space = space
        flat = space.maps.reshape(len(space.items), -1)
        errors = np.concatenate([space.errors(space.maps[start:start + 256])
                                 for start in range(0, len(space.items), 256)])  # (P, 2, K)
        self.difficulty = np.median(SINCERE * errors[:, 0] + BURIED * errors[:, 1], axis=0)
        kept = trimmed(self.difficulty)
        share = np.zeros(len(space.configs))
        share[kept] = 1 / len(kept)
        self.weight = np.array([SINCERE, BURIED])[:, None] * share  # (2, K): of each configuration's error in J
        weighted = np.broadcast_to((space.unit.sum(axis=-1) > 0) & (share > 0)[:, None, None],
                                   space.maps.shape[1:]).reshape(-1)
        self.active = weighted & (flat.min(axis=0) != flat.max(axis=0))
        self.columns = np.flatnonzero(self.active)
        self.L = np.empty((len(space.items), len(self.columns)), dtype=np.float32)
        for start in range(0, len(space.items), 256):
            part = self.losses(space.maps[start:start + 256]).reshape(-1, flat.shape[1])
            self.L[start:start + 256] = part[:, self.columns]
        self.constant = float(self.losses(space.maps[:1]).reshape(-1)[~self.active].sum())
        self.total = self.L.sum(axis=1, dtype=np.float64).astype(np.float32)
        self.flat = np.ascontiguousarray(flat[:, self.columns])  # the maps on the active points
        self.measures = space.measures.reshape(len(MEASURES), -1)[:, self.columns]

    def losses(self, maps: np.ndarray) -> np.ndarray:
        """What the winners of maps (..., 2, K, G, G) add to the search's J at every point."""
        return self.space.losses(maps) * self.weight[..., None, None]

    def alone(self, cost: float = NODE_COST) -> np.ndarray:
        """The search's J of every item alone."""
        return self.constant + self.total.astype(np.float64) + cost * self.space.sizes

    def mask(self, condition) -> np.ndarray:
        """Where on the active points a condition tree is true."""
        kind = condition[0]
        if kind == "elects":
            return self.flat[self.space.index[condition[1]]] >= 0
        if kind in ("same", "differ"):
            same = self.flat[self.space.index[condition[1]]] == self.flat[self.space.index[condition[2]]]
            return same if kind == "same" else ~same
        if kind in ("above", "below"):
            value = self.measures[condition[1]]
            return value > condition[2] if kind == "above" else value < condition[2]
        return self.mask(condition[1]) & self.mask(condition[2])

    def run(self, conditions, log=None, label: str = "") -> tuple[dict[str, Pool], np.ndarray]:
        """The best Conditionals of the conditions (a Conditions) over every two items
        A != B, of each ranking of COSTS, by the matrix product, BATCH conditions at a
        time; and the best J of every condition."""
        count = len(self.space.items)
        top = min(BEST, count)
        pools = {name: Pool.empty() for name in COSTS}
        best = np.full(len(conditions), np.inf)
        start = time.perf_counter()
        for first in range(0, len(conditions), BATCH):
            index = np.arange(first, min(first + BATCH, len(conditions)))
            M = conditions.masks(index).astype(np.float32) @ self.L.T  # each item's loss on the true points
            for name, cost in COSTS.items():
                item_cost = (cost * self.space.sizes).astype(np.float32)
                f = M + item_cost
                g = (self.total - M) + item_cost
                fa = np.argpartition(f, top - 1, axis=1)[:, :top]
                gb = np.argpartition(g, top - 1, axis=1)[:, :top]
                J = np.take_along_axis(f, fa, axis=1)[:, :, None] + np.take_along_axis(g, gb, axis=1)[:, None, :]
                J[fa[:, :, None] == gb[:, None, :]] = np.inf
                J = J.reshape(len(index), -1) + (self.constant + cost * (1 + conditions.sizes[index]))[:, None]
                keep = np.argsort(J, axis=1)[:, :KEEP]
                values = np.take_along_axis(J, keep, axis=1)
                if name == "J":
                    best[index] = values[:, 0]
                a = np.take_along_axis(fa, keep // top, axis=1)
                b = np.take_along_axis(gb, keep % top, axis=1)
                finite = np.isfinite(values)
                rows = np.broadcast_to(index[:, None], values.shape)
                pools[name] = pools[name].merged(Pool(values[finite], rows[finite], a[finite], b[finite]))
            if log is not None and ((first // BATCH) % 100 == 0 or first + BATCH >= len(conditions)):
                done = index[-1] + 1
                elapsed = time.perf_counter() - start
                log(f"  {label}: {done}/{len(conditions)} conditions, {elapsed:.0f} s, about "
                    f"{elapsed / done * (len(conditions) - done):.0f} s to go, best J {np.min(best[:done]):.5f}")
        return {name: pool.sorted() for name, pool in pools.items()}, best

# ---------------------------------------------------------------- conditions


class Conditions:
    """A list of condition trees with their sizes (blocks) and masks on the active points
    of a Search, by index."""

    def __init__(self, search: Search, trees: list, sizes):
        self.search, self.trees, self.sizes = search, list(trees), np.asarray(sizes, dtype=np.int64)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, k):
        return self.trees[k]

    def masks(self, index) -> np.ndarray:
        return np.array([self.search.mask(self.trees[k]) for k in index]).reshape(len(index), -1)

    def unique(self) -> "Conditions":
        """Of the conditions alike on the active points only the smallest, and none true
        or false everywhere there (its Conditional is one of its branches, larger)."""
        return self.subset(_unique(self, self.sizes))

    def subset(self, kept) -> "Conditions":
        return Conditions(self.search, [self.trees[k] for k in kept], self.sizes[kept])

    def tree(self, condition: int, a: int, b: int):
        """The Conditional of a condition and two items."""
        items = self.search.space.items
        return "if", self[condition], items[a], items[b]


class SamePairs(Conditions):
    """The conditions X == Y of every two items, of at most `max_size` blocks (with the
    block of the comparison; None: any), built a batch at a time: there are millions."""

    def __init__(self, search: Search, max_size: int | None = None, pairs: np.ndarray | None = None):
        sizes = search.space.sizes
        if pairs is None:
            order = np.argsort(sizes, kind="stable")
            parts = []
            for i, x in enumerate(order):
                ys = order[i + 1:]
                if max_size is not None:
                    ys = ys[sizes[ys] <= max_size - 1 - sizes[x]]
                parts.append(np.column_stack([np.full(len(ys), x), ys]))
            pairs = np.concatenate(parts).astype(np.int32) if parts else np.zeros((0, 2), np.int32)
        self.search, self.max_size, self.pairs = search, max_size, pairs
        self.sizes = 1 + sizes[pairs[:, 0]] + sizes[pairs[:, 1]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        items = self.search.space.items
        return "same", items[self.pairs[k, 0]], items[self.pairs[k, 1]]

    @property
    def trees(self):
        return [self[k] for k in range(len(self))]

    def masks(self, index) -> np.ndarray:
        flat = self.search.flat
        return flat[self.pairs[index, 0]] == flat[self.pairs[index, 1]]

    def subset(self, kept) -> "SamePairs":
        return SamePairs(self.search, self.max_size, self.pairs[kept])


def _unique(conditions: Conditions, sizes: np.ndarray) -> list[int]:
    order = np.argsort(sizes, kind="stable")
    kept, seen = [], set()
    for first in range(0, len(order), 4096):
        part = order[first:first + 4096]
        masks = conditions.masks(part)
        counts = masks.sum(axis=1)
        packed = np.packbits(masks, axis=1)
        for row, k in enumerate(part):
            if 0 < counts[row] < masks.shape[1]:
                key = hashlib.blake2b(packed[row].tobytes(), digest_size=16).digest()
                if key not in seen:
                    seen.add(key)
                    kept.append(int(k))
    return sorted(kept)


def condition_size(condition, sizes: dict) -> int:
    """Blocks of a condition tree, `sizes` those of the methods it reads."""
    kind = condition[0]
    if kind == "elects":
        return sizes[condition[1]]
    if kind in ("above", "below"):
        return 1 + complexity(MEASURES[condition[1]])
    if kind in ("same", "differ"):
        return 1 + sizes[condition[1]] + sizes[condition[2]]
    return 1 + condition_size(condition[1], sizes) + condition_size(condition[2], sizes)


def simple(search: Search) -> Conditions:
    """The conditions `elects` of every item and `above` of every measure and threshold."""
    items = search.space.items
    trees = [("elects", item) for item in items] + [("above", m, t) for m in range(len(MEASURES))
                                                    for t in THRESHOLDS]
    sizes = dict(zip(items, search.space.sizes))
    return Conditions(search, trees, [condition_size(tree, sizes) for tree in trees])


def combined(search: Search, best: list) -> Conditions:
    """c1 & c2 of every two of the conditions `best` (above and same), each maybe
    negated: with A and B swapped, every & and | of them."""
    sizes = dict(zip(search.space.items, search.space.sizes))
    trees = []
    for i, first in enumerate(best):
        for second in best[i + 1:]:
            for one in (first, negated(first)):
                for other in (second, negated(second)):
                    trees.append(("and", one, other))
    return Conditions(search, trees, [condition_size(tree, sizes) for tree in trees])

class Nested:
    """Conditional(c1, A1, Conditional(c2, A2, B)) of every two conditions c1 != c2 of a
    Conditions and the best items A1, A2 != B for them. J is the sum of c1 . l_A1, (not c1
    and c2) . l_A2 and (not c1 and not c2) . l_B with the costs, each of one item, so one
    matrix product a pair of conditions gives the best of all three: the last is not c1 .
    l_B, of the condition alone, less the second."""

    def __init__(self, conditions: Conditions):
        self.conditions, self.search = conditions, conditions.search
        first, second = np.meshgrid(np.arange(len(conditions)), np.arange(len(conditions)), indexing="ij")
        apart = first != second
        self.pairs = np.column_stack([first[apart], second[apart]])

    def __len__(self):
        return len(self.pairs)

    def tree(self, row: int, a: int, b: int):
        """The tree of a pair, A1 = a and A2, B = divmod(b, the number of items)."""
        items = self.search.space.items
        i, j = self.pairs[row]
        a2, b2 = divmod(b, len(items))
        return "if", self.conditions[int(i)], items[a], ("if", self.conditions[int(j)], items[a2], items[b2])

    def run(self, log=None) -> dict[str, Pool]:
        """The best of each pair of conditions, of each ranking of COSTS, BATCH pairs at a time."""
        search = self.search
        L, sizes, count = search.L, search.space.sizes, len(search.space.items)
        masks = self.conditions.masks(np.arange(len(self.conditions)))
        true = masks.astype(np.float32) @ L.T  # (C, P): c . l of every item
        false = search.total[None, :] - true
        costs = {name: (cost * sizes).astype(np.float32) for name, cost in COSTS.items()}
        first_best = {name: (true + costs[name]).min(axis=1) for name in COSTS}
        first_item = {name: (true + costs[name]).argmin(axis=1) for name in COSTS}
        pools = {name: Pool.empty() for name in COSTS}
        start, best = time.perf_counter(), np.inf
        for begin in range(0, len(self), BATCH):
            index = np.arange(begin, min(begin + BATCH, len(self)))
            i, j = self.pairs[index].T
            second = (~masks[i] & masks[j]).astype(np.float32) @ L.T  # (not c1 and c2) . l
            rest = false[i] - second  # (not c1 and not c2) . l
            for name, cost in COSTS.items():
                f2, f3 = second + costs[name], rest + costs[name]
                o2, o3 = np.argpartition(f2, 1, axis=1)[:, :2], np.argpartition(f3, 1, axis=1)[:, :2]
                v2, v3 = np.take_along_axis(f2, o2, axis=1), np.take_along_axis(f3, o3, axis=1)
                alike = o2[:, 0] == o3[:, 0]  # A2 == B: one of them the second best
                other2, other3 = v2[:, 1] + v3[:, 0], v2[:, 0] + v3[:, 1]
                a2 = np.where(alike & (other2 < other3), o2[:, 1], o2[:, 0])
                b = np.where(alike & (other2 >= other3), o3[:, 1], o3[:, 0])
                tail = np.where(alike, np.minimum(other2, other3), v2[:, 0] + v3[:, 0])
                J = (search.constant + cost * (2 + NEST + self.conditions.sizes[i] + self.conditions.sizes[j])
                     + first_best[name][i] + tail)
                if name == "J":
                    best = min(best, float(J.min()))
                pools[name] = pools[name].merged(Pool(J.astype(np.float32), index, first_item[name][i],
                                                      a2 * count + b))
            if log is not None and ((begin // BATCH) % 300 == 0 or begin + BATCH >= len(self)):
                done, elapsed = index[-1] + 1, time.perf_counter() - start
                log(f"  nested: {done}/{len(self)}, {elapsed:.0f} s, about {elapsed / done * (len(self) - done):.0f} s "
                    f"to go, best J {best:.5f}")
        return {name: pool.sorted() for name, pool in pools.items()}


def nested_conditions(search: Search, stages: dict, alias: dict) -> Conditions:
    """The conditions of Nested: the NESTED_TOP best of the stages and the NESTED_EACH best
    agreements X == Y of every method of AGREEING, alike on the active points only once."""
    from yeelab.build.methods import METHODS

    trees = _best_conditions(stages, NESTED_TOP)
    same, _, best = stages["same"]
    for name in AGREEING:
        k = search.space.index[alias.get(repr(METHODS[name]), repr(METHODS[name]))]
        mine = np.flatnonzero((same.pairs == k).any(axis=1))
        trees += [same[int(m)] for m in mine[np.argsort(best[mine], kind="stable")[:NESTED_EACH]]]
    trees = list(dict.fromkeys(trees))
    sizes = dict(zip(search.space.items, search.space.sizes))
    return Conditions(search, trees, [condition_size(tree, sizes) for tree in trees]).unique()


# ---------------------------------------------------------------- the run


def _json(value):
    if isinstance(value, tuple):
        return [_json(part) for part in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def candidates(space: Space, conditions: Conditions, pool: Pool, count: int, cost: float) -> list[dict]:
    """The `count` best Conditionals of a pool, scored exactly, of those alike on every
    point (sincerely and buried) only the best, by J with the cost `cost` of a block. The
    pool's J pays for every block, the exact one for every distinct block once, so ten
    times `count` are scored."""
    out, seen = [], set()
    for k in range(len(pool.J)):
        tree = conditions.tree(int(pool.condition[k]), int(pool.a[k]), int(pool.b[k]))
        key = space.behaviour(tree)
        if key in seen:
            continue
        seen.add(key)
        result = space.score(tree)
        out.append({"tree": tree, "J": result["J"], "rank": result["J"] - (NODE_COST - cost) * result["size"],
                    "sincere": result["sincere"], "burial": result["burial"], "size": result["size"],
                    "screened": float(pool.J[k])})
        if len(out) >= 10 * count:
            break
    out.sort(key=lambda entry: entry["rank"])
    return out[:count]


def run(out: Path, runs: Path = Path("runs"), max_size: int = 8, same_size: int | None = None, combine: int = 1000,
        deeper: int = 300, deeper_conditions: int = 5000, top_train: int = 200, top_test: int = 20,
        workers: int = 4, estimate: bool = False, pixel_median: str = PIXEL_MEDIAN, maps: Path | None = None):
    """The search, written to `out`: the maps of the library (maps/), the best of the
    training (train.json), of the validation (validation.json), the test report and a
    summary (summary.md). With `estimate`, only the numbers of conditions and Conditionals
    and how long they take. `pixel_median` as evaluate.Scorer; `maps` the directory of the
    maps (out/maps), which every pixel median shares."""
    log = console.print
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    maps = out / "maps" if maps is None else maps
    store = MapStore(maps / f"{TRAIN_SET}-{TRAIN_SIZE}", TRAIN_SET, TRAIN_SIZE)
    sources = library(runs, max_size)
    store.compute(sources, workers, log=log)
    items, alias = distinct(store, sources)
    space = Space(store, items, pixel_median=pixel_median)
    search = Search(space)
    log(f"library: {len(sources)} methods, {len(items)} of them distinct on {len(store.configs)} configurations "
        f"of {TRAIN_SET} (grid {TRAIN_SIZE}); {len(search.columns)} of {search.active.size} points differ")
    alone = search.alone()
    (out / "library.json").write_text(json.dumps({"sources": len(sources), "distinct": items, "alias": alias,
                                                  "J": dict(zip(items, alone.round(6).tolist()))}, indent=1))
    pairs = len(items) * (len(items) - 1)
    first = simple(search).unique()
    same = SamePairs(search, same_size)
    timing = same.subset(np.arange(min(len(same), 2 * BATCH)))
    tick = time.perf_counter()
    search.run(timing)
    per_condition = (time.perf_counter() - tick) / max(len(timing), 1)
    log(f"conditions: {len(first)} elects and above, {len(same)} same"
        f"{'' if same_size is None else f' (at most {same_size} blocks)'}, "
        f"{combine * (combine - 1) * 2} and; {pairs:,} pairs A != B; {per_condition * 1e3:.2f} ms a condition "
        f"({per_condition / pairs * 1e9:.3f} ns a Conditional, against ~0.35 s a method evaluated)")
    estimates = {"same": len(same) * per_condition, "simple": len(first) * per_condition,
                 "combined": combine * (combine - 1) * 2 * per_condition}
    log(f"estimated: {sum(estimates.values()) / 60:.0f} min for "
        f"{(len(first) + len(same) + combine * (combine - 1) * 2) * pairs:,.0f} Conditionals before the level deeper")
    if estimate:
        return
    tick = time.perf_counter()
    same = same.unique()
    log(f"  {len(same)} distinct same conditions ({time.perf_counter() - tick:.0f} s)")
    stages, record = {}, {"library": len(sources), "distinct": len(items), "points": int(search.active.size),
                          "active": len(search.columns), "ms_per_condition": per_condition * 1e3}
    for label, conditions in (("simple", first), ("same", same)):
        tick = time.perf_counter()
        pools, best = search.run(conditions, log=log, label=label)
        stages[label] = (conditions, pools, best)
        record[label] = _stage(conditions, best, pairs, time.perf_counter() - tick)
    # the best above and same conditions, combined
    best_conditions = _best_conditions(stages, combine, but="elects")
    tick = time.perf_counter()
    both = combined(search, best_conditions).unique()
    pools, best = search.run(both, log=log, label="and")
    stages["and"] = (both, pools, best)
    record["and"] = _stage(both, best, pairs, time.perf_counter() - tick)
    # one level deeper: the best Conditionals as items, with the best conditions
    found_first = {name: [] for name in COSTS}
    for name, cost in COSTS.items():
        for label, (conditions, pools, _) in stages.items():
            found_first[name] += candidates(space, conditions, pools[name], deeper, cost)
        found_first[name] = _best_distinct(space, found_first[name], deeper)
    deeper_items = [entry["tree"] for name in COSTS for entry in found_first[name]]
    deep_space = space.extended(deeper_items)
    deep_search = Search(deep_space)
    deep_trees = _best_conditions(stages, deeper_conditions)
    sizes = dict(zip(deep_space.items, deep_space.sizes))
    deep_conditions = Conditions(deep_search, deep_trees, [condition_size(tree, sizes) for tree in deep_trees])
    tick = time.perf_counter()
    deep_pools, best = deep_search.run(deep_conditions, log=log, label="deeper")
    count = len(deep_space.items)
    record["deeper"] = _stage(deep_conditions, best, count * (count - 1), time.perf_counter() - tick)
    nested = Nested(nested_conditions(search, stages, alias))
    log(f"nested: {len(nested.conditions)} conditions, {len(nested):,} pairs of them")
    tick = time.perf_counter()
    nested_pools = nested.run(log=log)
    record["nested"] = {"conditions": len(nested.conditions), "pairs": len(nested),
                        "conditionals": len(nested) * len(items) ** 3, "seconds": round(time.perf_counter() - tick, 1)}
    (out / "search.json").write_text(json.dumps(record, indent=1))
    train = {}
    for name, cost in COSTS.items():
        entries = found_first[name] + candidates(deep_space, deep_conditions, deep_pools[name], top_train, cost)
        train[name] = _best_distinct(deep_space, entries, top_train)
    train["alone"] = [{"tree": items[k]} | {key: value for key, value in space.score(items[k]).items() if key != "errors"}
                      for k in np.argsort(alone)[:top_train]]
    for entry in train["alone"]:
        entry["rank"] = entry["J"]
    for name in ("nested", "nested_raw"):
        train[name] = candidates(space, nested, nested_pools["raw" if name == "nested_raw" else "J"], top_train,
                                 RANKINGS[name])
    (out / "train.json").write_text(json.dumps({name: [_entry(e) for e in entries] for name, entries in train.items()},
                                               indent=1))
    log(f"search done in {(time.perf_counter() - started) / 60:.1f} min; best on {TRAIN_SET}: "
        + ", ".join(f"{name} {train[name][0]['rank']:.5f}" for name in RANKINGS))
    validated = validate(out, train, workers, log, maps, pixel_median)
    report(out, validated, top_test, workers, log, maps, pixel_median)


def _stage(conditions: Conditions, best: np.ndarray, pairs: int, seconds: float) -> dict:
    return {"conditions": len(conditions), "conditionals": len(conditions) * pairs, "seconds": round(seconds, 1),
            "best_J": float(best.min()) if len(best) else None}


def _best_conditions(stages: dict, count: int, but: str | None = None) -> list:
    """The `count` conditions of the stages of the least best J, none of the kind `but`."""
    labels = list(stages)
    best = np.concatenate([stages[label][2] for label in labels])
    owner = np.concatenate([np.full(len(stages[label][2]), k) for k, label in enumerate(labels)])
    within = np.concatenate([np.arange(len(stages[label][2])) for label in labels])
    out = []
    for k in np.argsort(best, kind="stable"):
        tree = stages[labels[owner[k]]][0][int(within[k])]
        if tree[0] != but:
            out.append(tree)
            if len(out) == count:
                break
    return out


def _best_distinct(space: Space, entries: list[dict], count: int) -> list[dict]:
    out, seen = [], set()
    for entry in sorted(entries, key=lambda e: e["rank"]):
        key = space.behaviour(entry["tree"])
        if key not in seen:
            seen.add(key)
            out.append(entry)
    return out[:count]


def _entry(entry: dict) -> dict:
    return {key: _json(value) for key, value in entry.items()} | {"method": repr(block(entry["tree"]))}


def validate(out: Path, train: dict[str, list[dict]], workers: int, log, maps: Path,
             pixel_median: str) -> dict[str, list[dict]]:
    """The best of the training scored on the validation set, from the maps of their
    primitives there, and the baselines (BASELINES) too."""
    from yeelab.build.methods import METHODS

    store = MapStore(maps / f"{VALIDATION_SET}-{VALIDATION_SIZE}", VALIDATION_SET, VALIDATION_SIZE)
    baselines = {name: repr(METHODS[name]) for name in BASELINES}
    needed = list(dict.fromkeys([leaf for entries in train.values() for entry in entries
                                 for leaf in leaves(entry["tree"])] + list(baselines.values())))
    store.compute(needed, workers, log=log)
    space = Space(store, needed, pixel_median=pixel_median)
    validated = {}
    for name, cost in RANKINGS.items():
        entries = []
        for entry in train[name]:
            result = space.score(entry["tree"])
            entries.append(entry | {"validation": result["J"],
                                    "validation_rank": result["J"] - (NODE_COST - cost) * result["size"],
                                    "validation_sincere": result["sincere"], "validation_burial": result["burial"]})
        entries.sort(key=lambda entry: entry["validation_rank"])
        validated[name] = entries
    validated["baselines"] = [{"name": name, "method": source} | space.score(source)
                              for name, source in baselines.items()]
    (out / "validation.json").write_text(json.dumps(
        {name: [_entry(e) if "tree" in e else e for e in entries] for name, entries in validated.items()}, indent=1))
    return validated


TEST_SET, TEST_SIZE = "test", 32


def report(out: Path, validated: dict, top: int, workers: int, log, maps: Path, pixel_median: str):
    """The best of the validation of each ranking and the baselines on the test set, from
    the maps of their primitives there (test.json), and a summary against Star Two Round.
    The maps are the blocks run on the voters (as yeesearch.report does) and serve every
    pixel median."""
    from yeelab.build.methods import METHODS

    chosen = _chosen(validated, top)
    trees = {repr(block(as_tree(entry["tree"]))): as_tree(entry["tree"])
             for name in RANKINGS for entry in validated[name][:top]}
    baselines = {repr(METHODS[name]): name for name in BASELINES}
    trees |= {source: source for source in baselines}
    store = MapStore(maps / f"{TEST_SET}-{TEST_SIZE}", TEST_SET, TEST_SIZE)
    needed = list(dict.fromkeys(leaf for tree in trees.values() for leaf in leaves(tree)))
    store.compute(needed, workers, log=log)
    space = Space(store, needed, pixel_median=pixel_median)
    scores = {source: space.score(tree) for source, tree in trees.items()}
    (out / "test.json").write_text(json.dumps({"pixel_median": pixel_median, "chosen": chosen, "scores": scores},
                                              indent=1))
    summary(out, scores, chosen, validated, pixel_median)


def _chosen(validated: dict, top: int) -> dict[str, str]:
    """{source: label} of the best of the validation of each ranking."""
    chosen = {}
    for name in RANKINGS:
        for k, entry in enumerate(validated[name][:top]):
            chosen.setdefault(repr(block(as_tree(entry["tree"]))), f"{name} {k + 1}")
    return chosen


def summary_again(out: Path, top: int):
    """summary() from the files of a finished search."""
    validated = json.loads((out / "validation.json").read_text())
    for name in RANKINGS:
        for entry in validated[name]:
            entry["tree"] = as_tree(entry["tree"])
    test = json.loads((out / "test.json").read_text())
    summary(out, test["scores"], test["chosen"], validated, test["pixel_median"])


def summary(out: Path, scores: dict[str, dict], chosen: dict[str, str], validated: dict, pixel_median: str):
    """summary.md: J, sincere and burial error on the test set, and on how many test
    configurations the burial error is lower and higher than Star Two Round's."""
    from yeelab.build.methods import METHODS

    base = np.array(scores[repr(METHODS["star_two_round"])]["errors"]["poll"])
    named = {repr(METHODS[name]): name for name in BASELINES}
    lines = [f"test set, grid {TEST_SIZE}, pixel median {pixel_median}, burial 30 % (the default poll)", "",
             "| method | J test | J − size | sincere | burial | burial better than STR | worse | size | J validation |",
             "|---|---|---|---|---|---|---|---|---|"]
    table = Table(title=f"test set, grid {TEST_SIZE}, pixel median {pixel_median} (burial: 30 %, the default poll); "
                        "against Star Two Round (STR)")
    for column in ("", "J", "J − size", "sincere", "burial", "better", "worse", "size", "J valid."):
        table.add_column(column, justify="left" if not column else "right")
    validation = {repr(block(e["tree"])): e["validation"] for name in RANKINGS for e in validated[name]}
    validation |= {e["method"]: e["J"] for e in validated["baselines"]}
    for source, score in sorted(scores.items(), key=lambda item: item[1]["J"]):
        mine = np.array(score["errors"]["poll"])
        better, worse = np.mean(mine < base - 1e-9), np.mean(mine > base + 1e-9)
        label = " = ".join(name for name in (named.get(source), chosen.get(source)) if name)
        cells = [label, f"{score['J']:.4f}", f"{score['J'] - NODE_COST * score['size']:.4f}",
                 f"{score['sincere']:.4f}", f"{score['burial']:.4f}", f"{better:.0%}", f"{worse:.0%}",
                 str(score["size"]), f"{validation.get(source, float('nan')):.4f}"]
        table.add_row(*cells)
        lines.append("| " + " | ".join(cells) + " |")
    console.print(table)
    lines.append("")
    for source, score in sorted(scores.items(), key=lambda item: item[1]["J"]):
        if source in chosen:
            lines.append(f"- **{chosen[source]}**: `{source}`")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
