"""Edge cases of the score rules: where RANGE, AVG, DHONDT, DH_NEAR, CLUSTER and POW differ.

POW is RANGE with its part of the way to the power p, round(part^p * T): with two levels
it approves the candidates beyond 2^(-1 / p) of the way instead of halfway. p is
--power, 1.5 by default here (score.POWER, 1, is RANGE itself), the single power that
fits the real ballots of docs/ballots.py best.

DH_NEAR is HYBRID of yeelab.score, CLUSTER its CLUSTER: the scores s from 0 to
T = levels - 1, never less for a closer candidate, T for the closest and 0 for the
farthest, that minimize

    sum_i (s_i / T - part_i)^2  +  mu * sum_k w_k * max(0, 1 - T g_k) * [s_(k) != s_(k+1)]

part_i is RANGE's part of the way, g_k the gap between neighbouring distinct distances
as a part of the span, w_k how firmly gap k holds a cluster: the strongest run of
neighbours it lies in, 1 - kappa * (largest inner gap) / (smaller gap around), 0 at
least. The gaps next to the closest and the farthest, whose scores are fixed, cost
nothing to split. The methods keep mu and kappa at score.MU and score.KAPPA, the
defaults here.

The summary judges every rule by criteria that do not name one of them; whether a rule
gives RANGE's ballot is shown as a property, not judged.

    uv run python script/score_edge_cases.py              # every case
    uv run python script/score_edge_cases.py -c b -c j    # some of them
    uv run python script/score_edge_cases.py --list
"""

from dataclasses import dataclass

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from yeelab.score import (AVG, CLUSTER, DELTA, DHONDT, HYBRID, KAPPA, MAX_LEVELS, MU, RANGE, _avg_part, _cohesion,
                          from_distances)

POW = "pow"  # RANGE with a power: not a rule of yeelab.score, but RANGE with power=
POWER = 1.5  # default of --power
ORDER = (RANGE, AVG, DHONDT, HYBRID, CLUSTER, POW)
NAMES = {RANGE: "range", AVG: "avg", DHONDT: "dhondt", HYBRID: "dh_near", CLUSTER: "cluster", POW: "pow"}
LABELS = tuple(NAMES[rule] for rule in ORDER)
SEED = 1
console = Console()
app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})


@dataclass(frozen=True)
class Settings:
    delta: float
    mu: float
    kappa: float
    power: float
    levels: tuple[int, ...]  # of every table of ballots; () for each case's own
    voters: int

    def of(self, levels):
        return self.levels or levels


def scores(r, levels, rule, s: Settings):
    """int (..., C): the scores of the rule for the distances r (..., C)."""
    r = np.asarray(r, dtype=np.float64)
    if rule == POW:
        return from_distances(r, levels, RANGE, power=s.power)
    return from_distances(r, levels, rule, s.delta, mu=s.mu, kappa=s.kappa)


def limit(r, rule, s):
    """(..., C): what score / (levels - 1) of the rule tends to with many levels: AVG's part
    of the way bent at the mean, POW's part to its power, the others' part itself."""
    if rule == AVG:
        return _avg_part(r)
    lo, hi = r.min(axis=-1, keepdims=True), r.max(axis=-1, keepdims=True)
    return ((hi - r) / (hi - lo)) ** (s.power if rule == POW else 1.0)


def cohesion(r, kappa=KAPPA):
    """(gaps, w) of the voter at the distances r (C,): how firmly each gap holds a cluster."""
    values = np.unique(np.asarray(r, dtype=np.float64))
    g = np.diff(values) / (values[-1] - values[0])
    w = np.empty_like(g)
    _cohesion(g, float(kappa), w)
    return g, w

# ---------------------------------------------------------------- Output


def styled(values, levels):
    """A ballot, the top score bold green and 0 grey; with two levels the number approved."""
    top, width = levels - 1, len(str(levels - 1))
    text = Text()
    for k, v in enumerate(values):
        style = "bold green" if v == top else "bright_black" if v == 0 else "green" if 2 * v >= top else "yellow"
        text.append((" " if k else "") + f"{v:{width}d}", style)
    if levels == 2:
        text.append(f"  ({values.sum()})", "cyan")
    return text


def ballot(r, levels, rule, s):
    return styled(scores(r, levels, rule, s), levels)


def distances(r):
    return " ".join(f"{d:g}" for d in r)


def note(text):
    console.print(Padding(" ".join(text.split()), (0, 0, 1, 2)), highlight=False)


def table(*columns, title=None):
    """A table whose first column alone wraps where the console is too narrow."""
    out = Table(*columns, title=title, box=box.ROUNDED, header_style="bold", title_justify="left")
    for column in out.columns[1:]:
        column.no_wrap = True
    return out


def ballots(rows, levels, s, title=None):
    out = table("distances", "L", *LABELS, title=title)
    for r in rows:
        for k, L in enumerate(levels):
            out.add_row(distances(r) if k == 0 else "", str(L), *(ballot(r, L, rule, s) for rule in ORDER),
                        end_section=k == len(levels) - 1)
    console.print(out)


def share(value):
    return Text(f"{value:.1%}", "green" if value == 0 else "yellow")

# ---------------------------------------------------------------- Cases

CASES = {}  # letter: (title, function)


def case(letter, title):
    def register(function):
        CASES[letter] = (title, function)
        return function
    return register


@case("a", "Nearly equal gaps")
def knife_edge(s):
    note("""Each interior candidate moves by 0.03 at most: DHONDT jumps from 1 approval to 5.
    No gap stands out, so CLUSTER is RANGE.""")
    ballots([[0, .21, .41, .61, .81, 1], [0, .19, .39, .59, .78, 1]], s.of((2, 6)), s)


@case("b", "A cluster between two nearly equal gaps")
def cluster_between(s):
    note("""0, a, a + 0.1, 1; the gaps tie at a = 0.45. DHONDT moves the cluster as a block.
    CLUSTER splits it with two levels, keeps it with six and grades it like RANGE with 16.
    POW's cut is further from the farthest: it approves the cluster later, one at a time.""")
    for L in s.of((2, 6, 16)):
        out = table("distances", *LABELS, title=f"L = {L}")
        for a in (.19, .29, .39, .41, .44, .46, .49, .51):
            r = [0, a, round(a + .1, 2), 1]
            out.add_row(distances(r), *(ballot(r, L, rule, s) for rule in ORDER))
        console.print(out)


@case("c", "The largest gap among the far candidates")
def far_gap(s):
    note("""The gap 9 to 20 is the largest: DHONDT gives 6, 8 and 9 one score, DH_NEAR approves
    the closest alone, CLUSTER sees 6, 8 and 9 as a cluster.""")
    ballots([[0, 6, 8, 9, 20], [0, 1, 7, 8, 10]], s.of((2, 6)), s)


@case("d", "Clones")
def clones(s):
    note("Exact copies change only AVG, through the mean distance.")
    ballots([[0, .3, 1], [0, 0, 0, .3, 1], [0, .6, 1], [0, .6, 1, 1, 1]], s.of((2, 6)), s)


@case("e", "An extremist")
def extremist(s):
    note("""One far candidate pushes the others to the top; DHONDT and DH_NEAR give the near four one
    score, CLUSTER the three after the closest; with six levels POW keeps the top for the two
    nearest.""")
    ballots([[0, 1, 2, 3], [0, 1, 2, 3, 20]], s.of((2, 6)), s)


@case("f", "Two clusters")
def clusters(s):
    note("""With six levels DHONDT, DH_NEAR and CLUSTER give each cluster one score; with 16 CLUSTER
    grades them like RANGE, DHONDT and DH_NEAR still pull the near one to the top.""")
    ballots([[0, .04, .12, .88, .95, 1]], s.of((2, 6, 16)), s)


@case("g", "Many levels")
def many_levels(s):
    note("""Mean |score / (L - 1) - limit| over random voters: every rule tends to its own limit,
    AVG to its part of the way bent at the mean, POW to the part to its power, the others to
    the part itself.""")
    r = random_distances(s, 5)
    levels = (2, 3, 4, 6, 11, 16)
    out = table("rule", *(f"L={L}" for L in levels))
    for rule in ORDER:
        own = limit(r, rule, s)
        out.add_row(NAMES[rule], *(f"{np.abs(scores(r, L, rule, s) / (L - 1) - own).mean():.3f}" for L in levels))
    console.print(out)


WALK = np.array([[.25, .30], [.40, .70], [.45, .62], [.60, .45], [.80, .25], [.75, .80]])  # candidates


def walk_distances(n=100_001):
    """(n, 6): a voter walking the diagonal from (0, 0) to (1, 1) past the candidates WALK."""
    points = np.linspace(0, 1, n)[:, None] * np.ones(2)
    return np.sqrt(((points[:, None, :] - WALK) ** 2).sum(axis=-1))


@case("h", "A voter walking across the plane")
def walk(s):
    note("100 000 steps past six candidates: how often scores change, and how many at once.")
    r = walk_distances()
    out = table("L", "rule", "changes", "most at once", "several at once")
    for L in s.of((2, 6)):
        for rule in ORDER:
            changed = (np.diff(scores(r, L, rule, s), axis=0) != 0).sum(axis=1)
            events = changed[changed > 0]
            out.add_row(str(L) if rule == ORDER[0] else "", NAMES[rule], str(events.size),
                        Text(str(events.max()), "green" if events.max() == 1 else "yellow"),
                        share(np.mean(events > 1)), end_section=rule == ORDER[-1])
    console.print(out)


@case("i", "Identities over random voters")
def identities(s):
    r = random_distances(s, 6)
    approve = {rule: scores(r, 2, rule, s) for rule in (RANGE, DHONDT, HYBRID)}
    out = table("identity", "voters for whom it fails")
    out.add_row("two levels: DH_NEAR approves no one RANGE or DHONDT leaves out",
                share(np.mean((approve[HYBRID] > np.minimum(approve[RANGE], approve[DHONDT])).any(axis=1))))
    console.print(out)

# ---------------------------------------------------------------- Summary

CRITERIA, PROPERTIES = [], []  # (source case, text, test(rule, s)): pass or fail; a value to show


def criterion(source, text, into=CRITERIA):
    def register(test):
        into.append((source, text, test))
        return test
    return register


def shown(source, text):
    return criterion(source, text, PROPERTIES)


def random_distances(s, count, voters=None):
    return np.random.default_rng(SEED).random((voters or s.voters, count))


def sweep(L, rule, s, low=.39, high=.51, n=1200):
    """(n, 4): the ballots of case B for a from low to high."""
    a = np.linspace(low, high, n)
    return scores(np.stack([np.zeros_like(a), a, a + .1, np.ones_like(a)], axis=-1), L, rule, s)


def is_cluster(r, members, kappa):
    """Whether the candidates `members` are neighbours by distance whose gaps hold a cluster."""
    values = np.unique(r)
    where = np.searchsorted(values, r)
    lo, hi = where[members].min(), where[members].max()
    return set(np.nonzero((where >= lo) & (where <= hi))[0]) == set(members) and (cohesion(r, kappa)[1][lo:hi] > 0).all()


KNIFE_EDGE = ([0, .21, .41, .61, .81, 1], [0, .19, .39, .59, .78, 1], [0, .2, .4, .6, .8, 1])


@criterion(knife_edge, "a move of 0.03 at most: one approval more or less, no score off by more than a level")
def _(rule, s):
    for i, one in enumerate(KNIFE_EDGE):
        for other in KNIFE_EDGE[i + 1:]:
            if abs(scores(one, 2, rule, s).sum() - scores(other, 2, rule, s).sum()) > 1:
                return False
            if np.abs(scores(one, 6, rule, s) - scores(other, 6, rule, s)).max() > 1:
                return False
    return True


@criterion(cluster_between, "L = 2, the cluster across the whole way: one approval changes at a time, "
                            "and it is split somewhere")
def _(rule, s):  # from 0.02 to 0.88 the cluster crosses the cut of every rule
    b = sweep(2, rule, s, .02, .88, 4000)
    return ((b[1:] != b[:-1]).sum(axis=1) <= 1).all() and (b[:, 1] != b[:, 2]).any()


@criterion(cluster_between, "L = 6: the cluster at most a level apart")
def _(rule, s):
    b = sweep(6, rule, s)
    return (np.abs(b[:, 1] - b[:, 2]) <= 1).all()


@criterion(cluster_between, "L = 16, the cluster across the whole way: no score jumps by more than a level")
def _(rule, s):
    return (np.abs(np.diff(sweep(16, rule, s, .02, .88, 4000), axis=0)) <= 1).all()


@criterion(far_gap, "0 6 8 9 21, L = 6: 6, 8 and 9 not all one score")
def _(rule, s):  # 21, not 20: no score exactly half way
    return len(set(scores([0, 6, 8, 9, 21], 6, rule, s)[1:4])) > 1


@criterion(clones, "an exact copy changes no other score, L = 2 and 6")
def _(rule, s):
    r = random_distances(s, 6)
    copy = r[np.arange(len(r)), np.random.default_rng(SEED + 1).integers(0, 6, len(r))]
    after = np.concatenate([r, copy[:, None]], axis=1)
    return all((scores(after, L, rule, s)[:, :6] == scores(r, L, rule, s)).all() for L in (2, 6))


@criterion(many_levels, "L = 16: within a quarter level of its own limit on average")
def _(rule, s):
    r = random_distances(s, 5)
    return np.abs(scores(r, 16, rule, s) - 15 * limit(r, rule, s)).mean() <= 0.25


@criterion(walk, "only a cluster changes several scores in one step, L = 2 and 6")
def _(rule, s):
    r = walk_distances()
    for L in (2, 6):
        b = scores(r, L, rule, s)
        changed = b[1:] != b[:-1]
        for i in np.nonzero(changed.sum(axis=1) > 1)[0]:
            if not is_cluster((r[i] + r[i + 1]) / 2, np.nonzero(changed[i])[0], s.kappa):
                return False
    return True


@criterion(identities, "scaled and shifted distances, same ballot")
def _(rule, s):
    r = random_distances(s, 6)
    return all((scores(100 + 3 * r, L, rule, s) == scores(r, L, rule, s)).all() for L in (2, 6, 16))


@criterion(identities, "closest T, farthest 0, a closer one never less")
def _(rule, s):
    r = random_distances(s, 6, voters=10_000)
    order = np.argsort(r, axis=1)
    for L in range(2, MAX_LEVELS + 1):
        b = np.take_along_axis(scores(r, L, rule, s), order, axis=1)
        if not ((b[:, 0] == L - 1).all() and (b[:, -1] == 0).all() and (np.diff(b, axis=1) <= 0).all()):
            return False
    return True


@shown(extremist, "0 1 2 3 21, near four, L = 6")
def _(rule, s):  # 21, not 20: no score exactly half way
    return styled(scores([0, 1, 2, 3, 21], 6, rule, s)[:4], 6)


@shown(clusters, "two clusters, L = 6")
def _(rule, s):
    return styled(scores([0, .04, .12, .88, .95, 1], 6, rule, s), 6)


@shown(clusters, "two clusters, L = 16")
def _(rule, s):
    return styled(scores([0, .04, .12, .88, .95, 1], 16, rule, s), 16)


LIKE_RANGE = []  # (source case, text, test(rule, s)): whether the rule gives RANGE's ballot


@criterion(knife_edge, "RANGE's ballot, L = 2 and 6", LIKE_RANGE)
def _(rule, s):
    return all((scores(r, L, rule, s) == scores(r, L, RANGE, s)).all() for r in KNIFE_EDGE for L in (2, 6))


@criterion(cluster_between, "L = 2, a from 0.39 to 0.51: one approval at a time, through 1 1 0 0", LIKE_RANGE)
def _(rule, s):
    b = sweep(2, rule, s)
    return ((b[1:] != b[:-1]).sum(axis=1) <= 1).all() and (b[:, 1] != b[:, 2]).any()


@criterion(cluster_between, "L = 16: graded like RANGE", LIKE_RANGE)
def _(rule, s):
    return (sweep(16, rule, s) == sweep(16, RANGE, s)).all()


@criterion(many_levels, "L = 16: within a quarter level of 1 - x on average", LIKE_RANGE)
def _(rule, s):
    r = random_distances(s, 5)
    return np.abs(scores(r, 16, rule, s) - 15 * limit(r, RANGE, s)).mean() <= 0.25


@case("j", "Summary")
def summary(s):
    note("""Criteria pass or fail, and name no rule. Properties are shown, not judged: the ballots
    of some cases, and whether a rule is RANGE where the earlier criteria asked for that.""")
    letters = {function: letter.upper() for letter, (_, function) in CASES.items()}
    out = table("case", "criterion", *LABELS)
    out.columns[1].no_wrap = False
    met = dict.fromkeys(ORDER, 0)
    for k, (source, text, test) in enumerate(CRITERIA):
        passed = {rule: bool(test(rule, s)) for rule in ORDER}
        for rule in ORDER:
            met[rule] += passed[rule]
        out.add_row(letters[source], text, *(Text("pass", "green") if passed[rule] else Text("fail", "bold red")
                                             for rule in ORDER), end_section=k == len(CRITERIA) - 1)
    out.add_row("", "met", *(Text(f"{met[rule]} of {len(CRITERIA)}", "bold") for rule in ORDER))
    console.print(out)
    out = table("case", "like RANGE", *LABELS)
    out.columns[1].no_wrap = False
    for source, text, test in LIKE_RANGE:
        out.add_row(letters[source], text, *(Text("yes") if test(rule, s) else Text("no", "bright_black")
                                             for rule in ORDER))
    console.print(out)
    out = table("property", *(f"{letters[source]}: {text}" for source, text, _ in PROPERTIES))
    for rule in ORDER:
        out.add_row(NAMES[rule], *(test(rule, s) for _, _, test in PROPERTIES))
    console.print(out)

# ---------------------------------------------------------------- Command


@app.command()
def main(
    cases: list[str] = typer.Option([], "--case", "-c", help=f"Case by letter ({', '.join(CASES)}); default all."),
    delta: float = typer.Option(DELTA, "--delta", "-d", help="Divisor of DHONDT and DH_NEAR."),
    mu: float = typer.Option(MU, "--mu", help="CLUSTER: cost of a split; 0 is RANGE."),
    kappa: float = typer.Option(KAPPA, "--kappa", help="CLUSTER: how much farther a cluster's neighbours are."),
    power: float = typer.Option(POWER, "--power", "-p", help="POW: the power of RANGE's part of the way; 1 is RANGE."),
    levels: list[int] = typer.Option([], "--levels", "-l", help="Levels of every ballot table; default each case's."),
    voters: int = typer.Option(100_000, "--voters", "-n", help="Random voters."),
    list_cases: bool = typer.Option(False, "--list", help="List the cases."),
) -> None:
    """Edge cases of the six score rules, side by side."""
    if list_cases:
        out = table("case", "title")
        for letter, (title, _) in CASES.items():
            out.add_row(letter, title)
        console.print(out)
        return
    cases = [c.lower() for c in cases] or list(CASES)
    if unknown := [c for c in cases if c not in CASES]:
        raise typer.BadParameter(f"unknown {', '.join(unknown)}; choose from {', '.join(CASES)}", param_hint="--case")
    if not (0 < delta < np.inf and 0 <= mu < np.inf and 0 < kappa < np.inf and 0 < power < np.inf and voters >= 1):
        raise typer.BadParameter("delta, kappa and power above 0, mu 0 or more, voters 1 or more")
    if not all(2 <= L <= MAX_LEVELS for L in levels):
        raise typer.BadParameter(f"levels must be 2 to {MAX_LEVELS}", param_hint="--levels")

    settings = Settings(delta, mu, kappa, power, tuple(levels), voters)
    console.print(f"delta {delta:g}   mu {mu:g}   kappa {kappa:g}   power {power:g}   voters {voters}",
                  style="bright_black")
    for letter in cases:
        title, function = CASES[letter]
        console.print()
        console.rule(f"[bold]{letter.upper()}. {title}", align="left")
        function(settings)


if __name__ == "__main__":
    app()
