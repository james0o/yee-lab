import random

import pytest

from yeelab.build.methods import METHODS

from yeesearch import grammar
from yeesearch.gp import crossover, hoist, point_mutation, seeds, subtree_mutation
from yeesearch.evaluate import NEEDS, Memo, Prepared, scenarios
from yeesearch.configs import SETS
from yeesearch.grammar import (COARSE_BALLOTS, PLAIN_NESTING, RULES, TRIMS, TYPES, Node, build_method, check, from_block,
                               nesting, plain_leaves, positions, random_tree)
from yeelab.build import (Burial, CoarseTrim, Conditional, Drawn, Highest, ScoreComparisons, Tally, Unbeaten,
                          Winner)
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


@pytest.fixture
def plain():
    rules, depths, nest = dict(RULES), dict(grammar.MIN_DEPTH), grammar.MAX_NESTING
    grammar.restrict_to_plain()
    yield
    RULES.clear()
    RULES.update(rules)
    grammar.MIN_DEPTH.clear()
    grammar.MIN_DEPTH.update(depths)
    grammar.MAX_NESTING = nest


def test_the_plain_vocabulary_has_only_its_leaves_joined_by_conditionals(plain):
    rng = random.Random(0)
    allowed = {"plurality", "score", "tally", "top", "lead", "threshold", "same", "both", "elects", "conditional",
               "fptp", "two_round", "condorcet", "dac", "d20", "mj", "coarse_2", "trimmed"}
    assert set(RULES) == allowed
    trees = grammar.ramped(rng, 200, range(3, 9))
    assert all(check(tree) and nesting(tree) <= PLAIN_NESTING for tree in trees)
    assert any(nesting(tree) == PLAIN_NESTING for tree in trees)
    for tree in trees:
        assert parse(repr(build_method(tree))).same_as(build_method(tree))
        assert from_block(build_method(tree)) == tree


def test_plain_trees_of_known_methods(plain):
    assert from_block(METHODS["score"]) == Node("trimmed", (6, 0))
    assert from_block(METHODS["trimmed"]) == Node("trimmed", (6, 25))
    assert from_block(METHODS["coarse_3"]) == Node("trimmed", (3, 25))
    assert from_block(METHODS["mj"]) == Node("mj")
    for name in ("star", "borda", "koth", "irv", "schulze", "search_1"):
        assert from_block(METHODS[name]) is None
    deep = Node("conditional", (), (Node("same", (True,), (Node("fptp"), Node("dac"))), Node("mj"), Node("dac")))
    assert not grammar.valid(Node("conditional", (), (Node("elects", (), (Node("condorcet"),)), deep,
                                                      Node("conditional", (), (deep.children[0], deep, Node("mj"))))))


def test_every_plain_leaf_reads_the_cached_shares():
    assert len(plain_leaves()) == 7 + 2 * len(TRIMS)
    for source in plain_leaves():
        assert Burial(parse(source), 0.3).needs <= NEEDS


def test_a_trim_changes_nothing_on_a_ballot_of_two_scores():
    prepared = Prepared.of(SETS["validation"]()[0], 8)
    ballot = COARSE_BALLOTS[2]
    plain_map = prepared.winners(Highest(Tally(ballot)))
    for trim in (10, 30, 45):
        trimmed_map = prepared.winners(Highest(Tally(ballot, trim=trim)))
        for scenario, winners in plain_map.items():
            assert (winners == trimmed_map[scenario]).all()


def test_the_training_set_without_rivals_differs_in_nothing_else():
    train, free = SETS["train"](), SETS["free"]()
    assert any(config.rival for config in train) and not any(config.rival for config in free)
    assert [config.name for config in train] == [config.name for config in free]


@pytest.fixture(params=[False, True], ids=["coarse", "coarse-extra"])
def coarse(request):
    rules, depths, nest = dict(RULES), dict(grammar.MIN_DEPTH), grammar.MAX_NESTING
    grammar.restrict_to_coarse(extra=request.param)
    yield request.param
    RULES.clear()
    RULES.update(rules)
    grammar.MIN_DEPTH.clear()
    grammar.MIN_DEPTH.update(depths)
    grammar.MAX_NESTING = nest


def test_the_coarse_vocabulary_joins_its_leaves_by_conditionals_and_runoffs(coarse):
    """Its random methods use only its rules, a runoff is among a draw, the best of a
    trimmed mean or two leaves, a Conditional is never among the candidates of a runoff,
    and every tree prints as an expression that gives the same tree back."""
    allowed = {"same", "both", "elects", "conditional", "fptp", "two_round", "condorcet", "coarse_mean",
               "coarse_trimmed", "coarse_runoff", "drawn", "coarse_finalists", "union", "has_winner"}
    assert set(RULES) == allowed | ({"d20", "dac", "koth"} if coarse else set())
    trees = grammar.ramped(random.Random(0), 200, range(3, 9))
    used = {node.rule for tree in trees for _, node, _ in positions(tree)}
    assert used == set(RULES)
    for tree in trees:
        assert check(tree) and nesting(tree) <= PLAIN_NESTING
        for _, node, _ in positions(tree):
            if node.rule == "coarse_runoff":
                assert node.children[0].rule in ("drawn", "coarse_finalists", "union")
            if node.rule == "union":
                assert not any(child.rule in ("conditional", "coarse_runoff") for child in node.children)
        method = build_method(tree)
        assert parse(repr(method)).same_as(method)
        assert from_block(method) == tree


def test_coarse_trees_of_known_methods(coarse):
    assert from_block(METHODS["two_round"]) == Node("two_round")
    assert from_block(METHODS["coarse_3"]) == Node("coarse_trimmed", (3, 25))
    trim = CoarseTrim(COARSE_BALLOTS[3]).has_winner(METHODS["two_round"])
    assert from_block(Conditional(trim, METHODS["two_round"], METHODS["condorcet"])) == Node(
        "conditional", (), (Node("has_winner", (3,), (Node("two_round"),)), Node("two_round"), Node("condorcet")))
    runoff = Unbeaten(ScoreComparisons(COARSE_BALLOTS[6], tie_weight=1.0), among=Drawn(Tally(COARSE_BALLOTS[2], trim=20)))
    assert from_block(runoff) == Node("coarse_runoff", (6, 1.0), (Node("drawn", (2, 20)),))
    for name in ("star", "mj", "score", "search_1"):  # Score(6) itself is not a coarse ballot
        assert from_block(METHODS[name]) is None
    assert (from_block(METHODS["d20"]) is not None) == coarse


def test_every_coarse_leaf_and_probe_reads_the_cached_shares():
    leaves = grammar.coarse_leaves()
    named = len(grammar.COARSE_NAMED) + len(grammar.EXTRA_NAMED)
    assert len(leaves) == named + 1 + len(TRIMS) * (2 + 3 + 3 * 2)
    probes = grammar.coarse_probes(list(leaves))
    assert len(probes) == 2 * (named + 1 + 2 * len(TRIMS))
    runoff = Unbeaten(ScoreComparisons(COARSE_BALLOTS[2], tie_weight=0.0), among=Drawn(Tally(COARSE_BALLOTS[6], trim=40)))
    for source in [*leaves, *probes, repr(runoff)]:
        block = parse(source)
        assert (Burial(block, 0.3) if isinstance(block, Winner) else block).needs <= NEEDS


def test_remembered_conditions_give_the_same_winners():
    """A method with has_winner gives the same winners through the Memo of subtrees, the
    condition remembered, as run on its own."""
    prepared = Prepared.of(SETS["validation"]()[0], 8)
    two_round, ballot = METHODS["two_round"], COARSE_BALLOTS[3]
    methods = [Conditional(CoarseTrim(ballot).has_winner(two_round), two_round, Highest(Tally(ballot, trim=25))),
               Conditional(CoarseTrim(ballot).has_winner(two_round), METHODS["condorcet"], two_round)]
    memo = Memo()
    for method in methods:
        remembered = prepared.winners(method, memo=memo)
        alone = {name: prepared.winner(each) for name, each in scenarios(method).items()}
        for name in alone:
            assert (remembered[name] == alone[name]).all()
    assert memo.hits > 0
