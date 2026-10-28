import itertools

import numpy as np
import pytest

from yeelab.build import Burial
from yeelab.build.methods import METHODS

from yeesearch.conditional import (COSTS, Conditions, SamePairs, Search, Space, block, combined, condition_size,
                                   distinct, simple)
from yeesearch.evaluate import NODE_COST, SHARE, Prepared, errors, objective
from yeesearch.grammar import build_method, distinct_size, size
from yeesearch.primitives import MapStore, config_maps, enumerated, trees

GRID = 8
COUNT = 4  # configurations of the training set
SOURCES = [repr(METHODS[name]) for name in ("fptp", "two_round", "score", "condorcet", "mj", "irv", "star", "dac",
                                            "koth")]
DAC_TOO = "Highest(Tally(Plurality()), among=DescendingCoalitions())"  # DAC everywhere
FPTP, TWO, SCORE, CONDORCET, MJ, IRV, STAR, DAC, KOTH = SOURCES


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store = MapStore(tmp_path_factory.mktemp("maps"), "train", GRID, count=COUNT)
    for k in range(COUNT):
        store.add(k, SOURCES + [DAC_TOO], config_maps("train", k, GRID, SOURCES + [DAC_TOO], measures=True))
    return store


@pytest.fixture(scope="module")
def prepared(store):
    return [Prepared.of(config, GRID) for config in store.configs]


@pytest.fixture(scope="module")
def space(store):
    return Space(store, SOURCES)


TREES = [
    ("if", ("elects", CONDORCET), CONDORCET, SCORE),
    ("if", ("above", 0, 0.4), FPTP, TWO),  # Top(Tally(Plurality())) > 0.4
    ("if", ("below", 9, 0.05), STAR, MJ),  # Lead(Tally(Score(6))) < 0.05
    ("if", ("same", FPTP, TWO), SCORE, CONDORCET),
    ("if", ("differ", MJ, SCORE), KOTH, IRV),
    ("if", ("and", ("same", FPTP, SCORE), ("below", 5, 0.3)), FPTP, STAR),
    ("if", ("same", CONDORCET, TWO), ("if", ("elects", CONDORCET), CONDORCET, MJ), ("if", ("above", 7, 0.2), STAR, DAC)),
]


@pytest.mark.parametrize("tree", TREES, ids=range(len(TREES)))
def test_maps_compose_as_the_blocks_run(space, prepared, tree):
    """The winners of a Conditional from the maps of its primitives, sincere and buried,
    are those of its block run on the voters, buried by Burial itself: burial is pointwise."""
    method = block(tree)
    composed = space.map_of(tree)
    for k, p in enumerate(prepared):
        np.testing.assert_array_equal(composed[0, k], p.winner(method))
        np.testing.assert_array_equal(composed[1, k], p.winner(Burial(method, SHARE)))


@pytest.mark.parametrize("tree", [TREES[0], TREES[5], TREES[6], STAR], ids=range(4))
def test_score_from_the_maps_is_the_objective(space, prepared, tree):
    method = block(tree)
    direct = objective(method, errors(method, prepared))
    fast = space.score(tree)
    assert fast["J"] == pytest.approx(direct["J"], rel=1e-12)
    assert fast["size"] == direct["size"] == distinct_size(method) <= size(method)
    assert fast["sincere"] == pytest.approx(direct["sincere"]) and fast["burial"] == pytest.approx(direct["burial"])


def test_condition_sizes_count_the_blocks(space):
    sizes = dict(zip(space.items, space.sizes))
    for tree in TREES:
        assert size(block(tree)) == 1 + condition_size(tree[1], sizes) + size(block(tree[2])) + size(block(tree[3]))


def paid(tree) -> dict:
    """space.score's J with the cost of every block, distinct or not: the search's."""
    return lambda score: score["J"] + NODE_COST * (size(block(tree)) - score["size"])


def test_the_search_finds_the_best_pair_of_every_condition(space):
    """The matrix product picks for every condition the A != B of the least J, as
    scoring every pair would, by J (paying for every block) and without the cost of the size."""
    search = Search(space)
    np.testing.assert_allclose(search.alone(), [paid(item)(space.score(item)) for item in space.items], rtol=1e-6)
    first = simple(search).unique()
    same = SamePairs(search, 40).unique()
    both = combined(search, [same[k] for k in range(min(6, len(same)))])
    for conditions in (first, same, both):
        pools, best = search.run(conditions)
        for k in range(0, len(conditions), max(1, len(conditions) // 8)):
            scores = [paid(tree)(space.score(tree)) for a, b in itertools.permutations(space.items, 2)
                      for tree in [("if", conditions[k], a, b)]]
            assert best[k] == pytest.approx(min(scores), abs=1e-5)
        for name, cost in COSTS.items():
            pool = pools[name]
            tree = ("if", conditions[int(pool.condition[0])], space.items[pool.a[0]], space.items[pool.b[0]])
            score = space.score(tree)
            assert pool.J[0] == pytest.approx(paid(tree)(score) - NODE_COST * size(block(tree)) + cost * size(block(tree)),
                                              abs=1e-5)


def test_alike_methods_are_one_primitive(store):
    kept, alias = distinct(store, SOURCES + [DAC_TOO])
    assert DAC in kept and DAC_TOO not in kept and alias[DAC_TOO] == DAC
    assert len(kept) == len(SOURCES)


def test_unique_conditions_drop_the_alike_and_the_constant(space):
    search = Search(space)
    conditions = Conditions(search, [("same", FPTP, TWO), ("differ", FPTP, TWO), ("same", FPTP, TWO),
                                     ("above", 0, 0.0)], [11, 11, 11, 4])
    unique = conditions.unique()
    assert unique.trees[:2] == [("same", FPTP, TWO), ("differ", FPTP, TWO)]
    assert ("above", 0, 0.0) not in unique.trees  # some first choices everywhere: always true


def test_enumeration_is_every_tree_by_its_blocks():
    memo = {}
    for blocks in range(1, 7):
        for node in trees("method", blocks, memo):
            assert size(build_method(node)) == blocks
    assert len(enumerated(5)) == 1 + 24 + 30 + 15
    assert len(set(enumerated(6))) == len(enumerated(6))
