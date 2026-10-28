"""A typed grammar of voting methods over the blocks of yeelab.build.

A method is a tree of Nodes, each one a Rule (a block's constructor with some of its
arguments fixed) with its constant parameters and its children. Every slot of a rule has a
type, and only the rules that fill that type go there, so every tree builds a block that
the constructors accept, nearly always (build() checks). The types:

    method   a Winner that elects one candidate (seats == 1)
    among    a Winner that chooses candidates for another one to choose among: a method,
             the n highest totals, the Smith set, the winners of two methods
    if       the condition of a Conditional: a Condition, or a method that is true where
             it elects someone (not in a cycle)
    condition  a Condition: two methods compared, a measure against a threshold, and those
             combined
    measure  Top or Lead of a Tally of the mean
    totals   CandidateTotals; tally a Tally (of the mean or the median); mean a Tally of
             the mean
    diffs    PairDiffs; cyclic those that can form a cycle (Smith and StrongestPaths need them)
    ballot   a Ballot; score a score ballot

Undecided() is left out: a method must try to elect someone everywhere. The score ballots
are a small fixed set (SCORE_BALLOTS), so that the shares of the voters cached for all
methods stay few (needs()). D20 (yeelab.build.methods.d20) is a method of its own, a
leaf: weak alone, it may tell where others are right inside a Conditional."""

import random
from dataclasses import dataclass, field, fields

from yeelab.build import (D21, AntiPlurality, BordaCount, Conditional, DescendingCoalitions, Eliminate, Highest, Lead,
                          Margins, Pairwise, Plurality, Score, ScoreComparisons, Smith, StrongestPaths, Tally, Top,
                          Unbeaten, Union, Weakest, Winner)
from yeelab.build.blocks import Block, CombinedCondition, Threshold, WinnerComparison
from yeelab.build.voters import FIRST, PAIRWISE, PROFILE, Ballots, Graded, Scored, ScoredPairwise

SCORE_POWER = 1.5  # as yeelab.build.methods.SCORE_POWER
SCORE_LEVELS = (2, 6)  # approval and the 0 to 5 ballot of the web UI
SCORE_BALLOTS = {levels: Score(levels, power=SCORE_POWER) for levels in SCORE_LEVELS}
TIE_WEIGHTS = (0.0, 1.0)  # of ScoreComparisons: the STAR runoff, and ranked on equal scores
THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7)
FINALISTS = (2, 3)  # of the n highest totals
MAJORITIES = (0.5, 0.55, 0.6)  # of the Smith set
D20 = Highest(Tally(D21(max_positive=2, max_negative=0, ordinal=True)))  # two votes for the first two
MAX_DEPTH = 12  # 9 and 40 before 2026-10-28: room for Conditionals in Conditionals
MAX_NODES = 60


@dataclass(frozen=True)
class Rule:
    """A block's constructor: the types it fills, the types of its children, the values
    each of its parameters can take, and build(params, children) -> block."""

    name: str
    fills: frozenset[str]
    children: tuple[str, ...]
    params: tuple[tuple, ...]
    build: object = field(compare=False, repr=False)


def _rule(name, fills, children, params, build) -> Rule:
    return Rule(name, frozenset(fills.split()), tuple(children.split()), params, build)


WINNERS = "method among"
RULES = {rule.name: rule for rule in [
    # ballots
    _rule("plurality", "ballot", "", (), lambda p, c: Plurality()),
    _rule("antiplurality", "ballot", "", (), lambda p, c: AntiPlurality()),
    _rule("borda_count", "ballot", "", (), lambda p, c: BordaCount()),
    _rule("score", "ballot score", "", (SCORE_LEVELS,), lambda p, c: SCORE_BALLOTS[p[0]]),
    # totals
    _rule("tally", "totals tally mean", "ballot", (), lambda p, c: Tally(c[0])),
    _rule("median", "totals tally", "score", (), lambda p, c: Tally(c[0], statistic="median")),
    _rule("weakest", "totals", "diffs", (), lambda p, c: Weakest(c[0])),
    # diffs
    _rule("margins", "diffs cyclic", "", (), lambda p, c: Margins(Pairwise())),
    _rule("score_comparisons", "diffs cyclic", "score", (TIE_WEIGHTS,),
          lambda p, c: ScoreComparisons(c[0], tie_weight=p[0])),
    _rule("strongest_paths", "diffs", "cyclic", (), lambda p, c: StrongestPaths(c[0])),
    # measures and conditions
    _rule("top", "measure", "mean", (), lambda p, c: Top(c[0])),
    _rule("lead", "measure", "mean", (), lambda p, c: Lead(c[0])),
    _rule("threshold", "condition if", "measure", (THRESHOLDS, (True, False)),
          lambda p, c: Threshold(c[0], p[0], above=p[1])),
    _rule("same", "condition if", "method method", ((True, False),),
          lambda p, c: WinnerComparison(c[0], c[1], equal=p[0])),
    _rule("both", "condition if", "condition condition", (("and", "or"),),
          lambda p, c: CombinedCondition(c[0], c[1], p[0])),
    _rule("elects", "if", "method", (), lambda p, c: c[0]),
    # winners
    _rule("highest", WINNERS, "totals", (), lambda p, c: Highest(c[0])),
    _rule("highest_among", WINNERS, "totals among", (), lambda p, c: Highest(c[0], among=c[1])),
    _rule("eliminate", WINNERS, "tally", (), lambda p, c: Eliminate(c[0], how="min")),
    _rule("eliminate_among", WINNERS, "tally among", (), lambda p, c: Eliminate(c[0], how="min", among=c[1])),
    _rule("eliminate_mean", WINNERS, "mean", (), lambda p, c: Eliminate(c[0], how="mean")),
    _rule("unbeaten", WINNERS, "diffs", (), lambda p, c: Unbeaten(c[0])),
    _rule("runoff", WINNERS, "diffs among", (), lambda p, c: Unbeaten(c[0], among=c[1])),
    _rule("king", WINNERS, "diffs method totals", (), lambda p, c: Unbeaten(c[0], against=c[1], order=c[2])),
    _rule("dac", WINNERS, "", (), lambda p, c: DescendingCoalitions()),
    _rule("d20", WINNERS, "", (), lambda p, c: D20),
    _rule("conditional", WINNERS, "if method method", (), lambda p, c: Conditional(c[0], c[1], c[2])),
    _rule("finalists", "among", "totals", (FINALISTS,), lambda p, c: Highest(c[0], n=p[0])),
    _rule("finalists_among", "among", "totals among", (FINALISTS,), lambda p, c: Highest(c[0], n=p[0], among=c[1])),
    _rule("smith", "among", "cyclic", (MAJORITIES,), lambda p, c: Smith(c[0], majority=p[0])),
    _rule("union", "among", "method method", (), lambda p, c: Union(c[0], c[1])),
]}
TYPES = frozenset().union(*(rule.fills for rule in RULES.values()))


def _min_depths() -> dict[str, int]:
    """The least depth of a tree of each rule, a leaf being 1."""
    depth = {name: 1 for name, rule in RULES.items() if not rule.children}
    by_type = {}
    while len(depth) < len(RULES):
        by_type = {t: min(d for name, d in depth.items() if t in RULES[name].fills)
                   for t in TYPES if any(t in RULES[name].fills for name in depth)}
        for name, rule in RULES.items():
            if name not in depth and all(t in by_type for t in rule.children):
                depth[name] = 1 + max(by_type[t] for t in rule.children)
    return depth


MIN_DEPTH = _min_depths()


@dataclass(frozen=True)
class Node:
    """A rule, its parameters (one value for each of rule.params) and its children."""

    rule: str
    params: tuple = ()
    children: tuple["Node", ...] = ()

    def __repr__(self):
        inner = [repr(value) for value in self.params] + [repr(child) for child in self.children]
        return f"{self.rule}({', '.join(inner)})"


class Invalid(ValueError):
    """A tree whose block the constructors do not accept."""


def build(node: Node):
    """The block of the tree, or Invalid."""
    rule = RULES[node.rule]
    try:
        return rule.build(node.params, [build(child) for child in node.children])
    except (ValueError, TypeError) as error:
        raise Invalid(f"{node!r}: {error}") from error


def build_method(node: Node) -> Winner:
    """build(), and Invalid unless it is a method that elects one candidate."""
    method = build(node)
    if not isinstance(method, Winner) or method.seats != 1:
        raise Invalid(f"{node!r} is not a method electing one candidate")
    return method


def nodes(node: Node) -> int:
    return 1 + sum(nodes(child) for child in node.children)


def depth(node: Node) -> int:
    return 1 + max((depth(child) for child in node.children), default=0)


def size(block) -> int:
    """The number of blocks in a block: the size of a method, whether the grammar can build it or not."""
    return 1 + sum(size(value) for item in fields(block)
                   if isinstance(value := getattr(block, item.name), Block))


def distinct_size(block) -> int:
    """The number of distinct blocks in a block: one that recurs, a method in a condition
    and in a branch or a ballot read twice, counted once, as yeelab.build.methods writes
    it once by name. The size J charges for (evaluate.objective); at most size()."""
    seen = set()

    def walk(part):
        key = repr(part)
        if key not in seen:
            seen.add(key)
            for item in fields(part):
                if isinstance(value := getattr(part, item.name), Block):
                    walk(value)

    walk(block)
    return len(seen)


def check(node: Node, slot: str = "method") -> bool:
    """Whether the tree is typed right in a slot of type `slot`."""
    rule = RULES.get(node.rule)
    return (rule is not None and slot in rule.fills and len(node.params) == len(rule.params)
            and all(value in values for value, values in zip(node.params, rule.params))
            and len(node.children) == len(rule.children)
            and all(check(child, t) for child, t in zip(node.children, rule.children)))


def needs() -> frozenset:
    """Every share of the voters a method of the grammar can read, sincere or buried:
    the voters cached for every configuration (yeesearch.evaluate)."""
    shares = {FIRST, PAIRWISE, PROFILE} | D20.needs | {Ballots(share) for share in D20.needs if isinstance(share, Scored)}
    for ballot in SCORE_BALLOTS.values():
        shares |= {ballot.scored, Graded(ballot.scored), Ballots(ballot.scored)}
        shares |= {ScoredPairwise(ballot.scored, weight) for weight in TIE_WEIGHTS}
    return frozenset(shares)

# ---------------------------------------------------------------- random trees


def random_tree(rng: random.Random, slot: str = "method", max_depth: int = 6, full: bool = False) -> Node:
    """A random tree for a slot of type `slot`, at most `max_depth` deep, or as deep as
    the least tree of the type: "grow" draws any rule that fits, "full" only rules with
    children while they fit."""
    filling = [name for name, rule in RULES.items() if slot in rule.fills]
    max_depth = max(max_depth, min(MIN_DEPTH[name] for name in filling))
    fitting = [name for name in filling if MIN_DEPTH[name] <= max_depth]
    if full:
        inner = [name for name in fitting if RULES[name].children]
        fitting = inner or fitting
    rule = RULES[rng.choice(fitting)]
    params = tuple(rng.choice(values) for values in rule.params)
    return Node(rule.name, params, tuple(random_tree(rng, t, max_depth - 1, full) for t in rule.children))


def ramped(rng: random.Random, count: int, depths=range(3, 7)) -> list[Node]:
    """Ramped half-and-half: `count` random methods of the depths, half grown and half full,
    each of at most MAX_NODES nodes and building a method."""
    trees = []
    depths = list(depths)
    while len(trees) < count:
        tree = random_tree(rng, "method", depths[len(trees) % len(depths)], full=len(trees) % 2 == 1)
        if nodes(tree) <= MAX_NODES and valid(tree):
            trees.append(tree)
    return trees


def valid(node: Node) -> bool:
    try:
        build_method(node)
    except Invalid:
        return False
    return nodes(node) <= MAX_NODES and depth(node) <= MAX_DEPTH

# ---------------------------------------------------------------- positions and edits


def positions(node: Node, slot: str = "method", path: tuple[int, ...] = ()):
    """(path, subtree, type of its slot) of every subtree, the root first."""
    yield path, node, slot
    for k, (child, t) in enumerate(zip(node.children, RULES[node.rule].children)):
        yield from positions(child, t, (*path, k))


def subtree(node: Node, path: tuple[int, ...]) -> Node:
    for k in path:
        node = node.children[k]
    return node


def replaced(node: Node, path: tuple[int, ...], new: Node) -> Node:
    """The tree with the subtree at `path` replaced by `new`."""
    if not path:
        return new
    children = list(node.children)
    children[path[0]] = replaced(children[path[0]], path[1:], new)
    return Node(node.rule, node.params, tuple(children))

# ---------------------------------------------------------------- from blocks


def from_block(block, slot: str = "method") -> Node | None:
    """The tree of a block the grammar can build, typed for `slot`; None for any other."""
    try:
        node = _from_block(block, slot)
    except _Unknown:
        return None
    return node if check(node, slot) else None


class _Unknown(Exception):
    pass


def _from_block(block, slot: str) -> Node:  # noqa: C901 -- one case per block
    match block:
        case Plurality():
            return Node("plurality")
        case AntiPlurality():
            return Node("antiplurality")
        case BordaCount():
            return Node("borda_count")
        case Score() if type(block) is Score and SCORE_BALLOTS.get(block.levels) == block:
            return Node("score", (block.levels,))
        case Tally(statistic="mean"):
            return Node("tally", (), (_from_block(block.ballot, "ballot"),))
        case Tally(statistic="median"):
            return Node("median", (), (_from_block(block.ballot, "score"),))
        case Weakest():
            return Node("weakest", (), (_from_block(block.diffs, "diffs"),))
        case Margins(shares=Pairwise()):
            return Node("margins")
        case ScoreComparisons():
            return Node("score_comparisons", (float(block.tie_weight),), (_from_block(block.ballot, "score"),))
        case StrongestPaths():
            return Node("strongest_paths", (), (_from_block(block.diffs, "cyclic"),))
        case Top():
            return Node("top", (), (_from_block(block.totals, "mean"),))
        case Lead():
            return Node("lead", (), (_from_block(block.totals, "mean"),))
        case Threshold():
            return Node("threshold", (block.threshold, block.above), (_from_block(block.measure, "measure"),))
        case WinnerComparison():
            return Node("same", (block.equal,), (_from_block(block.first, "method"), _from_block(block.second, "method")))
        case CombinedCondition():
            return Node("both", (block.operator,),
                        (_from_block(block.first, "condition"), _from_block(block.second, "condition")))
        case Winner() if slot == "if":
            return Node("elects", (), (_from_block(block, "method"),))
        case Highest() if repr(block) == repr(D20):
            return Node("d20")
        case Highest():
            totals = _from_block(block.totals, "totals")
            among = () if block.among is None else (_from_block(block.among, "among"),)
            if block.n == 1:
                return Node("highest_among" if among else "highest", (), (totals, *among))
            return Node("finalists_among" if among else "finalists", (block.n,), (totals, *among))
        case Eliminate(how="min", until=1):
            tally = _from_block(block.totals, "tally")
            if block.among is None:
                return Node("eliminate", (), (tally,))
            return Node("eliminate_among", (), (tally, _from_block(block.among, "among")))
        case Eliminate(how="mean", until=1, among=None):
            return Node("eliminate_mean", (), (_from_block(block.totals, "mean"),))
        case Unbeaten(against=None, among=None):
            return Node("unbeaten", (), (_from_block(block.diffs, "diffs"),))
        case Unbeaten(against=None):
            return Node("runoff", (), (_from_block(block.diffs, "diffs"), _from_block(block.among, "among")))
        case Unbeaten():
            return Node("king", (), (_from_block(block.diffs, "diffs"), _from_block(block.against, "method"),
                                     _from_block(block.order, "totals")))
        case DescendingCoalitions():
            return Node("dac")
        case Conditional():
            return Node("conditional", (), (_from_block(block.condition, "if"),
                                            _from_block(block.when_true, "method"),
                                            _from_block(block.when_false, "method")))
        case Smith():
            return Node("smith", (block.majority,), (_from_block(block.diffs, "cyclic"),))
        case Union():
            return Node("union", (), (_from_block(block.first, "method"), _from_block(block.second, "method")))
    raise _Unknown(repr(block))
