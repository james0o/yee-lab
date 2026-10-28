"""The data of search.typ, written to data.json: the configurations of every set, the
empty strip of the geometric diagram by deviation, the complexity of the methods of the
web UI's rows, and their scores on the test set (from the test maps of a run's directory).

    cd search && uv run python docs/data.py runs/2026-10-27-conditional/maps
"""

import json
import sys
from pathlib import Path

import numpy as np

from yeelab.build.methods import FOUND, METHODS

from yeesearch.conditional import Space, leaves
from yeesearch.configs import SETS, empty_strip, inside
from yeesearch.evaluate import NODE_COST
from yeesearch.grammar import complexity, distinct_size, size
from yeesearch.primitives import MapStore

SHOWN = ("star", "star_two_round", "mj", "two_round", "irv", "fptp", *FOUND)
EXAMPLES = ("fptp", "star", "star_two_round", "search_1", "search_5", "search_6", "adhoc", "con_fptp")


def kind(name: str) -> str:
    if name.startswith(("random", "ood-")):
        return "ood" if name.startswith("ood-") else "random"
    return "hand"


def configurations() -> dict:
    out = {}
    for set_name, make in SETS.items():
        out[set_name] = [{"name": c.name, "kind": kind(c.name), "deviation": c.deviation,
                          "candidates": [list(point) for point in c.candidates]} for c in make()]
    return out


def dropped() -> dict:
    """The configurations the wall rule leaves out, by set: drawn again without it."""
    import yeesearch.configs as configs

    keep = configs._inside
    configs._inside = lambda found: found
    try:
        every = {name: make() for name, make in SETS.items()}
    finally:
        configs._inside = keep
    return {name: [c.name for c in found if not inside(c)] for name, found in every.items()}


def scores(maps: Path) -> dict:
    store = MapStore(maps / "test-32", "test", 32)
    sources = {name: repr(METHODS[name]) for name in SHOWN}
    store.compute(list(sources.values()))
    space = Space(store, list(sources.values()))
    reference = np.array(space.score(sources["star_two_round"])["errors"]["poll"])
    out = {}
    for name, source in sources.items():
        score = space.score(source)
        burial = np.array(score["errors"]["poll"])
        out[name] = {key: score[key] for key in ("J", "sincere", "burial", "size")} | {
            "raw": score["J"] - NODE_COST * score["size"],
            "better": float(np.mean(burial < reference - 1e-9)), "worse": float(np.mean(burial > reference + 1e-9))}
    return out


if __name__ == "__main__":
    maps = Path(sys.argv[1])
    data = {"sets": configurations(), "dropped": dropped(),
            "strip": {f"{d:g}": round(empty_strip(d), 3) for d in (0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4)},
            "complexity": {name: {"blocks": size(METHODS[name]), "distinct": distinct_size(METHODS[name]),
                                  "complexity": complexity(METHODS[name])} for name in EXAMPLES},
            "methods": {name: repr(METHODS[name]) for name in SHOWN},
            "scores": scores(maps)}
    Path(__file__).with_name("data.json").write_text(json.dumps(data, indent=1))
    print("data.json:", {name: len(found) for name, found in data["sets"].items()}, "dropped",
          {name: len(found) for name, found in data["dropped"].items()})
