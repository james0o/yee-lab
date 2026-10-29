import random

import pytest

from yeelab.build.methods import METHODS

from yeesearch import grammar
from yeesearch.gp import crossover, hoist, point_mutation, seeds, subtree_mutation
from yeesearch.grammar import RULES, TYPES, Node, build_method, check, from_block, positions, random_tree
from yeesearch.pool import parse


def test_every_type_has_a_finite_tree():
    assert set(grammar.MIN_DEPTH) == set(RULES)
    for slot in TYPES:
        assert any(slot in rule.fills for rule in RULES.values())


def test_random_methods_build_and_print_as_their_expression():
    """Nearly every random tree builds a method electing one candidate, and its repr is
    an expression that builds the same method again."""
    rng = random.Random(0)
    built = 0
    for _ in range(500):
        tree = random_tree(rng, "method", rng.randint(2, 7), full=rng.random() < 0.5)
        assert check(tree)
        if grammar.valid(tree):
            method = build_method(tree)
            assert method.seats == 1
            assert parse(repr(method)).same_as(method)
            built += 1
    assert built >= 450


@pytest.mark.parametrize("name", ["fptp", "irv", "schulze", "star", "mj", "koth", "smj_two", "nanson", "three_round"])
def test_known_methods_are_trees(name):
    tree = from_block(METHODS[name])
    assert tree is not None and repr(build_method(tree)) == repr(METHODS[name])


def test_methods_beyond_the_grammar_are_not_trees():
    """Undecided, D21 and score ballots other than SCORE_BALLOTS are left out."""
    for name in ("new", "adhoc", "d21", "system321"):
        assert from_block(METHODS[name]) is None
    assert from_block(METHODS["star"].among, "method") is None  # it chooses two
    assert from_block(METHODS["star"].among, "among") is not None


def test_seeds_include_black():
    sources = {repr(build_method(tree)) for tree in seeds()}
    assert repr(METHODS["schulze"]) in sources
    assert any(source.startswith("Conditional(Unbeaten(Margins(Pairwise()))") for source in sources)


def test_operators_keep_the_types():
    """Crossover and the mutations give trees typed right."""
    rng = random.Random(1)
    trees = grammar.ramped(rng, 40)
    for _ in range(300):
        first, second = rng.sample(trees, 2)
        for tree in (crossover(rng, first, second), subtree_mutation(rng, first), point_mutation(rng, first),
                     hoist(rng, first)):
            assert check(tree), tree


def test_positions_name_the_slot_of_every_subtree():
    tree = from_block(METHODS["koth"])
    slots = [(path, node.rule, slot) for path, node, slot in positions(tree)]
    assert slots[0] == ((), "king", "method")
    assert ((1,), "highest", "method") in slots and ((2,), "tally", "totals") in slots
    assert grammar.replaced(tree, (1,), Node("dac")).children[1] == Node("dac")


def test_every_among_is_paid_for_each_time_it_occurs():
    star, two_round = METHODS["star"], METHODS["two_round"]
    assert grammar.among_count(METHODS["fptp"]) == 0
    assert grammar.among_count(star) == 1
    union = parse(f"Highest(Tally(BordaCount()), among={star!r} | {two_round!r})")
    assert grammar.among_count(union) == 3
    twice = parse(f"Unbeaten(Margins(Pairwise()), among={star!r} | Highest(Tally(Plurality()), among={star!r}))")
    once = parse(f"Unbeaten(Margins(Pairwise()), among={star!r} | Highest(Tally(Plurality())))")
    assert grammar.among_count(twice) == 4  # star's among counted in each copy
    assert grammar.complexity(twice) - grammar.complexity(once) == 2 * grammar.AMONG  # as many blocks, two among= more
