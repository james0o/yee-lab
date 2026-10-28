"""The library of primitive methods that the search of Conditionals (yeesearch.conditional)
composes: every method of the grammar up to MAX_SIZE blocks, enumerated, the methods of
yeelab.build.methods the grammar can build and the best methods of the GP runs. For each
one, its map: the winner at every point of every configuration of a set, sincerely and
where 30 % of the voters bury (evaluate.scenarios), CYCLE (-1) where no one wins. And the
measures Top and Lead that the threshold conditions read, sincerely and buried.

Burial is pointwise: its poll names the frontrunners at every point from the sincere first
choices, whatever the method, and the method then runs on the buried voters of the point.
So Burial(Conditional(c, A, B)) elects where(c buried, A buried, B buried), and the maps of
the primitives give the maps of any Conditional of them, without running a block.

The maps of a set are kept in a directory (MapStore), a file for every configuration and
batch of methods, so that a stopped computation goes on where it stopped; a worker holds
the voters of one configuration at a time."""

import itertools
import json
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pebble import ProcessExpired

from yeelab.build import AntiPlurality, BordaCount, Lead, Plurality, Tally, Top, Winner
from yeelab.build.blocks import Measure
from yeelab.build.methods import METHODS
from yeelab.margin import regions

from yeesearch.configs import SETS
from yeesearch.evaluate import SHARE, Prepared, scenarios
from yeesearch.grammar import RULES, SCORE_BALLOTS, Invalid, Node, build_method, from_block, size
from yeesearch.pool import Workers, parse

MAX_SIZE = 8  # blocks of the enumerated methods: 8368 methods; 9 would add 58549
OWN = {name: 0 if name == "elects" else 1 if rule.children else size(rule.build(tuple(v[0] for v in rule.params), []))
       for name, rule in RULES.items()}  # blocks of a rule itself: of a leaf, all it builds (Margins(Pairwise()), D20)
BALLOTS = (Plurality(), AntiPlurality(), BordaCount(), *SCORE_BALLOTS.values())  # of the grammar's "mean"
MEASURES = tuple(kind(Tally(ballot)) for kind in (Top, Lead) for ballot in BALLOTS)
SCENARIO_NAMES = ("sincere", "poll")  # the second axis of the maps
TASK_TIMEOUT = 3600.0  # seconds for every method of the library on one configuration

# ---------------------------------------------------------------- the methods


def _compositions(total: int, parts: int):
    """Every way to write `total` as `parts` positive numbers, in order."""
    if parts == 0:
        if total == 0:
            yield ()
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield first, *rest


def trees(slot: str, blocks: int, memo: dict | None = None) -> list[Node]:
    """Every tree of the grammar for a slot of type `slot` whose block has `blocks`
    blocks (grammar.size), whether the constructors accept it or not."""
    memo = {} if memo is None else memo
    if (slot, blocks) in memo:
        return memo[slot, blocks]
    out = []
    for name, rule in RULES.items():
        rest = blocks - OWN[name]
        if slot not in rule.fills or rest < 0 or (rest > 0) != bool(rule.children):
            continue
        params = list(itertools.product(*rule.params))
        if not rule.children:
            out += [Node(name, values) for values in params]
            continue
        for sizes in _compositions(rest, len(rule.children)):
            children = [trees(t, k, memo) for t, k in zip(rule.children, sizes)]
            out += [Node(name, values, combo) for combo in itertools.product(*children) for values in params]
    memo[slot, blocks] = out
    return out


def enumerated(max_size: int = MAX_SIZE) -> list[str]:
    """The sources of every method of the grammar of at most `max_size` blocks, the
    smallest first."""
    memo, sources = {}, {}
    for blocks in range(1, max_size + 1):
        for node in trees("method", blocks, memo):
            try:
                sources.setdefault(repr(build_method(node)), None)
            except Invalid:
                pass
    return list(sources)


def known() -> dict[str, str]:
    """{source: name} of the methods of yeelab.build.methods the grammar can build."""
    return {repr(method): name for name, method in METHODS.items() if from_block(method) is not None}


def found(runs: Path, top: int = 50) -> list[str]:
    """The `top` methods of the hall of fame of every GP run in `runs`, by validation J."""
    sources = []
    for hall in sorted(runs.glob("*/hall.json")):
        entries = json.loads(hall.read_text())
        entries.sort(key=lambda e: (e.get("validation") is None, e.get("validation") or 0, e["J"]))
        sources += [entry["method"] for entry in entries[:top]]
    return list(dict.fromkeys(sources))

# ---------------------------------------------------------------- maps


def side(grid: int) -> int:
    """Points along a side of a grid: its cells' centres and both walls (regions.grid)."""
    return len(regions.grid(grid, 1)[0])


@dataclass(frozen=True, eq=False)
class Measured(Winner):
    """A measure in the place of a winner, so that Burial(Measured(measure), share) gives
    the measure on the buried voters of every point, polled as for any method."""

    measure: Measure

    @property
    def needs(self):
        return self.measure.needs

    def evaluate(self, voters):
        value = self.measure.evaluate(voters)
        return value, np.full(value.shape, np.inf)


def winners(prepared: Prepared, method) -> np.ndarray:
    """The winners of the method (2, G, G) on a prepared configuration: sincerely and
    buried (SCENARIO_NAMES); for a Measured, the measure."""
    variants = scenarios(method, SHARE)
    return np.array([prepared.winner(variants[name]) for name in SCENARIO_NAMES])


def config_maps(set_name: str, index: int, grid: int, sources: list[str], measures: bool) -> dict:
    """The maps (S, 2, G, G) int8 of the sources on one configuration of a set (G =
    side(grid)), the
    indices of those that fail, and with `measures` those of MEASURES (M, 2, G, G): one
    task of a worker."""
    prepared = Prepared.of(SETS[set_name]()[index], grid)
    maps = np.full((len(sources), 2, side(grid), side(grid)), -2, dtype=np.int8)
    failed = []
    for k, source in enumerate(sources):
        try:
            maps[k] = winners(prepared, parse(source))
        except Exception:  # noqa: BLE001 -- a method that fails anywhere is left out of the library
            failed.append(k)
    out = {"maps": maps, "failed": failed}
    if measures:
        out["measures"] = np.array([winners(prepared, Measured(measure)) for measure in MEASURES])
    return out


class MapStore:
    """The maps of methods on the configurations of a set on a grid, in a directory:
    <configuration>.<batch>.npz with the sources and their maps, <configuration>.measures.npy;
    `count` only the first configurations of the set."""

    def __init__(self, path: Path, set_name: str, grid: int, count: int | None = None):
        self.path, self.set_name, self.grid = path, set_name, grid
        path.mkdir(parents=True, exist_ok=True)
        self.configs = SETS[set_name]()[:count]
        self.rows: list[dict[str, int]] = []  # of each configuration: {source: row of maps}, -1 failed
        self.maps: list[np.ndarray] = []
        for config in self.configs:
            rows, parts = {}, []
            for file in sorted(path.glob(f"{config.name}.batch-*.npz")):
                with np.load(file) as data:
                    failed = set(json.loads(str(data["failed"])))
                    offset = sum(len(part) for part in parts)
                    for k, source in enumerate(data["sources"]):
                        rows[str(source)] = -1 if k in failed else offset + k
                    parts.append(data["maps"])
            self.rows.append(rows)
            self.maps.append(np.concatenate(parts) if parts else np.zeros((0, 2, side(grid), side(grid)), np.int8))

    def missing(self, sources: list[str]) -> dict[int, list[str]]:
        return {k: [source for source in sources if source not in rows] for k, rows in enumerate(self.rows)}

    def _measures_path(self, index: int) -> Path:
        return self.path / f"{self.configs[index].name}.measures.npy"

    def add(self, index: int, sources: list[str], result: dict):
        name = self.configs[index].name
        batch = len(list(self.path.glob(f"{name}.batch-*.npz")))
        written = self.path / f"{name}.batch-{batch:04d}.tmp"  # not matched by the glob of __init__
        with written.open("wb") as file:
            np.savez(file, sources=np.array(sources), maps=result["maps"], failed=json.dumps(result["failed"]))
        written.rename(self.path / f"{name}.batch-{batch:04d}.npz")
        offset = len(self.maps[index])
        failed = set(result["failed"])
        self.rows[index].update({source: -1 if k in failed else offset + k for k, source in enumerate(sources)})
        self.maps[index] = np.concatenate([self.maps[index], result["maps"]])
        if "measures" in result:
            np.save(self._measures_path(index), result["measures"])

    def compute(self, sources: list[str], workers: int = 4, log=print):
        """Computes the maps of the sources the store does not have, in the workers, one
        configuration a task."""
        missing = self.missing(sources)
        todo = {k: chunk for k, chunk in missing.items() if chunk or not self._measures_path(k).exists()}
        if not todo:
            return
        log(f"maps of {max(map(len, todo.values()))} methods on {len(todo)} configurations of {self.set_name} "
            f"(grid {self.grid}) to compute")
        pool = Workers(workers)
        try:
            futures = {k: pool.pool.schedule(config_maps, (self.set_name, k, self.grid, chunk,
                                                           not self._measures_path(k).exists()),
                                             timeout=TASK_TIMEOUT) for k, chunk in todo.items()}
            for done, (k, future) in enumerate(futures.items()):
                try:
                    result = future.result()
                except (FutureTimeout, ProcessExpired) as error:
                    raise RuntimeError(f"maps of {self.configs[k].name}: {error!r}") from error
                self.add(k, todo[k], result)
                log(f"  configuration {done + 1}/{len(todo)} {self.configs[k].name}")
        finally:
            pool.close()

    def stacked(self, sources: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(maps (P, 2, K, G, G) int8 of the sources, ok (P,) bool: computed and
        nowhere failed); the store must have every source."""
        rows = np.array([[rows[source] for rows in self.rows] for source in sources], dtype=np.int64).reshape(
            len(sources), len(self.configs))
        ok = (rows >= 0).all(axis=1)
        maps = np.stack([self.maps[k][np.maximum(rows[:, k], 0)] for k in range(len(self.configs))], axis=2)
        return maps, ok

    def measures(self) -> np.ndarray:
        """The measures (M, 2, K, G, G)."""
        return np.stack([np.load(self._measures_path(k)) for k in range(len(self.configs))], axis=2)
