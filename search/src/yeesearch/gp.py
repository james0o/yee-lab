"""Strongly typed genetic programming over the grammar: the baseline the neural generator
is to beat.

The first population is the methods of yeelab.build.methods the grammar can build and
random trees (ramped half-and-half). Each generation keeps the ELITE best and fills the
rest with children of parents chosen by tournaments: crossover swaps subtrees of the same
type, a mutation regrows a subtree, redraws one node or hoists a subtree into the place of
its ancestor of the same type. A child already in the population is drawn again, one
already scored anywhere in the run is not scored again (pool.Archive). The best methods
(the hall of fame) are scored on the validation set too."""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from yeelab.build import BordaCount, Conditional, Highest, Margins, Pairwise, Tally, Unbeaten
from yeelab.build.methods import METHODS

from yeesearch import grammar
from yeesearch.grammar import RULES, Invalid, Node, build_method, from_block, positions, random_tree, replaced
from yeesearch.pool import Archive, Workers, parse

TRAIN_SIZE = 16
VALIDATION_SIZE = 24
ELITE = 5
TOURNAMENT = 5
OPERATORS = {"crossover": 0.7, "subtree": 0.15, "point": 0.1, "hoist": 0.05}
HALL = 50  # methods kept in the hall of fame
RETRIES = 50  # draws of a child before taking a parent as it is
console = Console()


def seeds() -> list[Node]:
    """The methods of yeelab.build.methods the grammar can build; Black's method with
    "the Condorcet winner, where there is one" as a method that elects someone."""
    condorcet = Unbeaten(Margins(Pairwise()))
    black = Conditional(condorcet, condorcet, Highest(Tally(BordaCount())))
    trees = [from_block(method) for method in [*METHODS.values(), black]]
    return list({repr(tree): tree for tree in trees if tree is not None}.values())


def source(node: Node) -> str:
    return repr(build_method(node))


@dataclass
class Individual:
    node: Node
    source: str
    J: float


def tournament(rng: random.Random, population: list[Individual]) -> Individual:
    return min(rng.sample(population, min(TOURNAMENT, len(population))), key=lambda one: one.J)


def crossover(rng: random.Random, first: Node, second: Node) -> Node:
    """first with a random subtree replaced by a subtree of second that fills its type."""
    path, _, slot = rng.choice(list(positions(first)))
    donors = [node for _, node, _ in positions(second) if slot in RULES[node.rule].fills]
    return replaced(first, path, rng.choice(donors)) if donors else first


def subtree_mutation(rng: random.Random, node: Node) -> Node:
    path, _, slot = rng.choice(list(positions(node)))
    return replaced(node, path, random_tree(rng, slot, rng.randint(1, 4)))


def point_mutation(rng: random.Random, node: Node) -> Node:
    """One node redrawn: another rule of the same children's types that fills its slot,
    or other parameters."""
    path, old, slot = rng.choice(list(positions(node)))
    rule = RULES[old.rule]
    alike = [other for other in RULES.values() if slot in other.fills and other.children == rule.children]
    new = rng.choice(alike)
    params = tuple(rng.choice(values) for values in new.params)
    return replaced(node, path, Node(new.name, params, old.children))


def hoist(rng: random.Random, node: Node) -> Node:
    """A subtree moved into the place of an ancestor whose slot it fills: a smaller method."""
    path, old, slot = rng.choice(list(positions(node)))
    below = [inner for inner_path, inner, _ in positions(old) if inner_path and slot in RULES[inner.rule].fills]
    return replaced(node, path, rng.choice(below)) if below else node


def child(rng: random.Random, population: list[Individual], taken: set[str]) -> tuple[Node, str]:
    """A new valid method not among `taken`, by a random operator."""
    for _ in range(RETRIES):
        operator = rng.choices(list(OPERATORS), weights=list(OPERATORS.values()))[0]
        parent = tournament(rng, population).node
        if operator == "crossover":
            node = crossover(rng, parent, tournament(rng, population).node)
        elif operator == "subtree":
            node = subtree_mutation(rng, parent)
        elif operator == "point":
            node = point_mutation(rng, parent)
        else:
            node = hoist(rng, parent)
        if grammar.valid(node):
            try:
                text = source(node)
            except Invalid:
                continue
            if text not in taken:
                return node, text
    node = random_tree(rng, "method", 5)
    while not grammar.valid(node):
        node = random_tree(rng, "method", 5)
    return node, source(node)


def run(out: Path, hours: float, population: int = 1000, workers: int = 4, seed: int = 0, timeout: float = 60.0,
        generations: int | None = None):
    """Evolve for `hours` (or `generations`), writing to the directory `out`: evals.jsonl
    every method scored on the training set, validation.jsonl the hall of fame on the
    validation set, generations.jsonl the progress, hall.json the hall of fame after every
    generation. A run started again in the same directory starts from the best methods it
    has scored, and does not score any of them again."""
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    archive, validation = Archive(out / "evals.jsonl"), Archive(out / "validation.jsonl")
    workers_ = Workers(workers, timeout, prepared=(("train", TRAIN_SIZE), ("validation", VALIDATION_SIZE)))
    progress = (out / "generations.jsonl").open("a")
    deadline = time.monotonic() + 3600 * hours
    stopped = False
    try:
        nodes = _resumed(archive, population) or seeds()
        nodes += grammar.ramped(rng, max(population - len(nodes), 0))
        current = _scored(workers_, archive, nodes)
        generation = 0
        while time.monotonic() < deadline and (generations is None or generation < generations):
            generation += 1
            start = time.monotonic()
            current.sort(key=lambda one: one.J)
            taken = {one.source for one in current[:ELITE]}
            children = []
            while len(children) < population - ELITE:
                node, text = child(rng, current, taken)
                taken.add(text)
                children.append(node)
            fresh = sum(source(node) not in archive for node in children)
            current = current[:ELITE] + _scored(workers_, archive, children)
            hall = _hall(archive)
            workers_.methods([entry["method"] for entry in hall], "validation", VALIDATION_SIZE, validation)
            _write_hall(out / "hall.json", hall, validation)
            best = min(current, key=lambda one: one.J)
            finite = sorted(one.J for one in current if one.J < float("inf"))
            median = finite[len(finite) // 2] if finite else float("inf")
            record = {"generation": generation, "best": best.J, "median": median,
                      "failed": len(current) - len(finite), "fresh": fresh, "scored": len(archive.results),
                      "seconds": round(time.monotonic() - start, 1), "method": best.source}
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            console.print(f"gen {generation:4d}  best J {best.J:.4f}  median {median:.4f}  "
                          f"failed {record['failed']:3d}  new {fresh:3d}  scored {len(archive.results):6d}  "
                          f"{record['seconds']:6.1f} s\n           {best.source}")
    except KeyboardInterrupt:
        stopped = True
        console.print("stopped; run again with the same --out to go on")
    finally:
        progress.close()
        workers_.close(stop=stopped)
        archive.close()
        validation.close()


def _resumed(archive: Archive, population: int) -> list[Node]:
    """The trees of the `population` best methods of the archive (none in a new run)."""
    trees = [from_block(parse(entry["method"])) for entry in _hall(archive, population)]
    return [tree for tree in trees if tree is not None]


def _write_hall(path: Path, hall: list[dict], validation: Archive):
    path.write_text(json.dumps(
        [{key: entry[key] for key in ("method", "J", "sincere", "burial", "size")}
         | {"validation": validation[entry["method"]]["J"] if entry["method"] in validation else None}
         for entry in hall], indent=1))


def _scored(workers: Workers, archive: Archive, nodes: list[Node]) -> list[Individual]:
    sources = [source(node) for node in nodes]
    results = workers.methods(sources, "train", TRAIN_SIZE, archive)
    return [Individual(node, text, results[text]["J"]) for node, text in zip(nodes, sources)]


def _hall(archive: Archive, count: int = HALL) -> list[dict]:
    """The `count` best methods scored so far on the training set, of those that err
    alike on every configuration only the first, the smallest (its J is the lowest)."""
    scored = sorted((entry for entry in archive.results.values() if entry["J"] < float("inf")),
                    key=lambda entry: entry["J"])
    hall, seen = [], set()
    for entry in scored:
        key = json.dumps(entry["errors"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            hall.append(entry)
            if len(hall) == count:
                break
    return hall
