"""Strongly typed genetic programming over the grammar: the baseline the neural generator
is to beat.

The first population is the methods of yeelab.build.methods the grammar can build and
random trees (ramped half-and-half). Each generation keeps the ELITE best and fills the
rest with IMMIGRANTS of new random trees and children of parents chosen by tournaments:
crossover swaps subtrees of the same type, a mutation regrows a subtree, redraws one node
or hoists a subtree into the place of its ancestor of the same type. A child already in
the population is drawn again, one already scored anywhere in the run is not scored again
(pool.Archive). Of the methods that err alike on every configuration only the best stays
in the population, the others are replaced by random trees, so that it does not fill
with copies of one method; and where the best has not improved for STALL generations the
whole population starts again from random trees, a new cycle (the best are kept in the
archive and the hall of fame). The best methods (the hall of fame) are scored on the validation set
too."""

import fcntl
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

from yeelab.build import BordaCount, Conditional, Highest, Margins, Pairwise, Tally, Unbeaten
from yeelab.build.methods import METHODS

from yeesearch import grammar
from yeesearch.grammar import RULES, Invalid, Node, build_method, from_block, positions, random_tree, replaced
from yeesearch.pool import Archive, Workers, append_jsonl, parse, read_jsonl, synced

TRAIN_SIZE = 16
VALIDATION_SIZE = 24
ELITE = 5
TOURNAMENT = 5
OPERATORS = {"crossover": 0.5, "subtree": 0.3, "point": 0.15, "hoist": 0.05}
IMMIGRANTS = 0.25  # of a generation: new random trees
RANDOM_DEPTHS = range(3, 9)  # of the random trees, deep enough for Conditionals
REFILLS = 3  # rounds of random trees replacing the copies in a population
STALL = 30  # generations without a better method in the cycle before the population starts again
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
    behaviour: str  # its errors on every configuration: methods that err alike are copies


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
        generations: int | None = None, until: datetime | None = None):
    """Evolve for `hours`, or until the time `until`, or `generations`, writing to the
    directory `out`: evals.jsonl every method scored on the training set, validation.jsonl
    the hall of fame on the validation set, generations.jsonl the progress, hall.json the
    hall of fame after every generation. A run started again in the same directory, by hand
    or after a power cut (resume.sh), starts from the best methods it has scored and does
    not score any of them again: what it has written is on the disk after every
    generation. One run at a time works in a directory (lock)."""
    out.mkdir(parents=True, exist_ok=True)
    seconds = (until - datetime.now()).total_seconds() if until is not None else 3600 * hours
    if seconds <= 0:
        console.print(f"the run in {out} was to end at {until:%Y-%m-%d %H:%M}: nothing to do")
        return
    lock = (out / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run is going on in {out}") from None
    deadline = time.monotonic() + seconds
    done = len(read_jsonl(out / "generations.jsonl"))  # generations of the run before it stopped
    rng = random.Random(seed + done)  # another draw when it goes on
    archive, validation = Archive(out / "evals.jsonl"), Archive(out / "validation.jsonl")
    workers_ = Workers(workers, timeout, prepared=(("train", TRAIN_SIZE), ("validation", VALIDATION_SIZE)))
    progress = append_jsonl(out / "generations.jsonl")
    stopped = False
    try:
        nodes = _resumed(archive, population) or seeds()
        nodes += _random(rng, population - len(nodes))
        current = _distinct(rng, workers_, archive, _scored(workers_, archive, nodes), population)
        generation, stalled = done, 0
        best_so_far = cycle_best = min(one.J for one in current)  # cycle: since the last restart
        while time.monotonic() < deadline and (generations is None or generation - done < generations):
            generation += 1
            start = time.monotonic()
            restart = stalled >= STALL
            if restart:
                nodes, stalled, cycle_best = _random(rng, population), 0, float("inf")
            else:
                current.sort(key=lambda one: one.J)
                taken = {one.source for one in current[:ELITE]}
                nodes = _random(rng, round(IMMIGRANTS * population), taken)
                taken |= {source(node) for node in nodes}
                while len(nodes) < population - ELITE:
                    node, text = child(rng, current, taken)
                    taken.add(text)
                    nodes.append(node)
            fresh = sum(source(node) not in archive for node in nodes)
            scored = _scored(workers_, archive, nodes)
            current = _distinct(rng, workers_, archive, scored if restart else current[:ELITE] + scored, population)
            best = min(current, key=lambda one: one.J)
            if best.J < cycle_best - 1e-12:
                cycle_best, stalled = best.J, 0
            else:
                stalled += 1
            best_so_far = min(best_so_far, best.J)
            hall = _hall(archive)
            workers_.methods([entry["method"] for entry in hall], "validation", VALIDATION_SIZE, validation)
            _write_hall(out / "hall.json", hall, validation)
            checked = [entry for entry in hall if entry["method"] in validation]
            best_validated = min(checked, key=lambda entry: validation[entry["method"]]["J"]) if checked else None
            finite = sorted(one.J for one in current if one.J < float("inf"))
            median = finite[len(finite) // 2] if finite else float("inf")
            record = {"generation": generation, "best": best.J, "best_ever": best_so_far, "median": median,
                      "failed": len(current) - len(finite), "distinct": len({one.behaviour for one in current}),
                      "restart": restart, "fresh": fresh, "scored": len(archive.results),
                      "seconds": round(time.monotonic() - start, 1), "method": best.source,
                      "validation_best": validation[best_validated["method"]]["J"] if best_validated else None,
                      "validation_method": best_validated["method"] if best_validated else None}
            progress.write(json.dumps(record) + "\n")
            for file in (archive, validation):
                file.sync()
            synced(progress)
            console.print(f"gen {generation:4d}{' restart' if restart else ''}  best J {best.J:.4f} "
                          f"(ever {best_so_far:.4f})  median {median:.4f}  failed {record['failed']:3d}  "
                          f"new {fresh:3d}  scored {len(archive.results):6d}  {record['seconds']:6.1f} s\n"
                          f"           train best: {best.source}\n"
                          f"           validation best ({record['validation_best'] or float('nan'):.4f}): "
                          f"{record['validation_method']}")
    except KeyboardInterrupt:
        stopped = True
        console.print("stopped; run again with the same --out to go on")
    finally:
        synced(progress)
        progress.close()
        workers_.close(stop=stopped)
        archive.close()
        validation.close()
        lock.close()


def _random(rng: random.Random, count: int, taken: set[str] = frozenset()) -> list[Node]:
    """`count` new random methods (grammar.ramped) not among `taken`."""
    trees, seen = [], set(taken)
    while len(trees) < count:
        for tree in grammar.ramped(rng, count - len(trees), RANDOM_DEPTHS):
            text = source(tree)
            if text not in seen:
                seen.add(text)
                trees.append(tree)
    return trees


def _distinct(rng: random.Random, workers: Workers, archive: Archive, individuals: list[Individual],
              population: int) -> list[Individual]:
    """The individuals with only the best of those that err alike, made up to `population`
    by new random trees that err unlike them, in at most REFILLS rounds."""
    kept, seen = [], set()
    for one in sorted(individuals, key=lambda one: one.J):
        if one.behaviour not in seen:
            seen.add(one.behaviour)
            kept.append(one)
    for _ in range(REFILLS):
        missing = population - len(kept)
        if missing <= 0:
            break
        for one in _scored(workers, archive, _random(rng, missing, {one.source for one in kept})):
            if one.behaviour not in seen:
                seen.add(one.behaviour)
                kept.append(one)
    return kept


def _resumed(archive: Archive, population: int) -> list[Node]:
    """The trees of the `population` best methods of the archive (none in a new run)."""
    trees = [from_block(parse(entry["method"])) for entry in _hall(archive, population)]
    return [tree for tree in trees if tree is not None]


def _write_hall(path: Path, hall: list[dict], validation: Archive):
    """hall.json, whole or as it was: written beside it and moved over it."""
    written = path.with_suffix(".tmp")
    with written.open("w") as file:
        json.dump([{key: entry[key] for key in ("method", "J", "sincere", "burial", "size")}
                   | {"validation": validation[entry["method"]]["J"] if entry["method"] in validation else None}
                   for entry in hall], file, indent=1)
        synced(file)
    os.replace(written, path)


def _scored(workers: Workers, archive: Archive, nodes: list[Node]) -> list[Individual]:
    sources = [source(node) for node in nodes]
    results = workers.methods(sources, "train", TRAIN_SIZE, archive)
    return [Individual(node, text, results[text]["J"], _behaviour(results[text]))
            for node, text in zip(nodes, sources)]


def _behaviour(result: dict) -> str:
    """What a method errs on every configuration of the training set; all that fail alike."""
    return json.dumps(result.get("errors"), sort_keys=True)


def _hall(archive: Archive, count: int = HALL) -> list[dict]:
    """The `count` best methods scored so far on the training set, of those that err
    alike on every configuration only the first, the smallest (its J is the lowest)."""
    scored = sorted((entry for entry in archive.results.values() if entry["J"] < float("inf")),
                    key=lambda entry: entry["J"])
    hall, seen = [], set()
    for entry in scored:
        key = _behaviour(entry)
        if key not in seen:
            seen.add(key)
            hall.append(entry)
            if len(hall) == count:
                break
    return hall
