"""Edge cases of the score rules: where RANGE, AVG, DHONDT, DH_NEAR and CLUSTER differ.

DH_NEAR is HYBRID of yeelab.score. CLUSTER lives only here: the scores s from 0 to
T = levels - 1, never less for a closer candidate, T for the closest and 0 for the
farthest, that minimize

    sum_i (s_i / T - part_i)^2  +  mu * sum_k w_k * max(0, 1 - T g_k) * [s_(k) != s_(k+1)]

part_i is RANGE's part of the way, g_k the gap between neighbouring distinct distances
as a part of the span, w_k how firmly gap k holds a cluster: the strongest run of
neighbours it lies in, 1 - kappa * (largest inner gap) / (smaller gap around), 0 at
least. The gaps next to the closest and the farthest, whose scores are fixed, cost
nothing to split.

    uv run python script/score_edge_cases.py              # every case
    uv run python script/score_edge_cases.py -c b -c m    # some of them
    uv run python script/score_edge_cases.py --list
"""

from dataclasses import dataclass, replace

import numpy as np
import typer
from numba import njit
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from yeelab.score import AVG, DELTA, DHONDT, HYBRID, MAX_LEVELS, RANGE, _avg_part, from_distances

CLUSTER = "cluster"
ORDER = (RANGE, AVG, DHONDT, HYBRID, CLUSTER)
NAMES = {RANGE: "range", AVG: "avg", DHONDT: "dhondt", HYBRID: "dh_near", CLUSTER: "cluster"}
LABELS = tuple(NAMES[rule] for rule in ORDER)
MU, KAPPA = 0.1, 2.0  # defaults of --mu and --kappa
SEED = 1
console = Console()
app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})


@dataclass(frozen=True)
class Settings:
    delta: float
    mu: float
    kappa: float
    levels: tuple[int, ...]  # of every table of ballots; () for each case's own
    voters: int

    def of(self, levels):
        return self.levels or levels

# ---------------------------------------------------------------- CLUSTER


@njit
def _cohesion(g, kappa, w):
    """w (m,) for the gaps g (m,) between neighbouring distinct distances in order."""
    m = g.size
    w[:] = 0.0
    for i in range(m + 1):  # the run of distances i..j
        for j in range(i + 1, m + 1):
            if i == 0 and j == m:
                continue
            around = min(g[i - 1] if i > 0 else np.inf, g[j] if j < m else np.inf)
            strength = 1.0 - kappa * g[i:j].max() / around
            for k in range(i, j):
                w[k] = max(w[k], strength)


@njit
def _cluster_ballot(r, top, mu, kappa, out):
    """The CLUSTER ballot of the voter at the distances r (C,), into out (C,)."""
    order = np.argsort(r)
    values = np.empty(r.size)  # the distinct distances, closest first
    group = np.empty(r.size, dtype=np.int64)  # of each candidate: the index of its distance
    n = 0
    for c in order:
        if n == 0 or r[c] > values[n - 1]:
            values[n] = r[c]
            n += 1
        group[c] = n - 1
    if n == 1:
        out[:] = top
        return
    span = values[n - 1] - values[0]
    g = np.diff(values[:n]) / span
    w = np.empty(n - 1)
    _cohesion(g, kappa, w)
    cost = np.full(top + 1, np.inf)  # of the best scores so far, by the score of the last
    cost[top] = 0.0
    new = np.empty(top + 1)
    back = np.zeros((n, top + 1), dtype=np.int64)
    for k in range(1, n):
        split = 0.0 if k == 1 or k == n - 1 else mu * w[k - 1] * max(0.0, 1.0 - top * g[k - 1])
        part = (values[n - 1] - values[k]) / span
        for v in range(top + 1):
            best, arg = cost[v], v
            for u in range(v + 1, top + 1):
                if cost[u] + split < best:
                    best, arg = cost[u] + split, u
            new[v] = best + (v / top - part) ** 2
            back[k, v] = arg
        cost[:] = new
    scores = np.empty(n, dtype=np.int64)
    v = 0
    for k in range(n - 1, -1, -1):
        scores[k] = v
        v = back[k, v]
    out[:] = scores[group]


@njit
def _cluster_ballots(r, top, mu, kappa, out):
    for i in range(r.shape[0]):
        _cluster_ballot(r[i], top, mu, kappa, out[i])


def cluster_scores(r, levels, mu=MU, kappa=KAPPA):
    r = np.asarray(r, dtype=np.float64)
    flat = np.ascontiguousarray(r.reshape(-1, r.shape[-1]))
    out = np.empty(flat.shape, dtype=np.int64)
    _cluster_ballots(flat, levels - 1, float(mu), float(kappa), out)
    return out.reshape(r.shape)


def cohesion(r, kappa=KAPPA):
    """(gaps, w) of the voter at the distances r (C,)."""
    values = np.unique(np.asarray(r, dtype=np.float64))
    g = np.diff(values) / (values[-1] - values[0])
    w = np.empty_like(g)
    _cohesion(g, float(kappa), w)
    return g, w


def scores(r, levels, rule, s: Settings):
    """int (..., C): the scores of the rule for the distances r (..., C)."""
    if rule == CLUSTER:
        return cluster_scores(r, levels, s.mu, s.kappa)
    return from_distances(np.asarray(r, dtype=np.float64), levels, rule, s.delta)

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
    CLUSTER splits it with two levels, keeps it with six and grades it like RANGE with 16.""")
    for L in s.of((2, 6, 16)):
        out = table("distances", *LABELS, title=f"L = {L}")
        for a in (.39, .41, .44, .46, .49, .51):
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
    score, CLUSTER the three after the closest.""")
    ballots([[0, 1, 2, 3], [0, 1, 2, 3, 20]], s.of((2, 6)), s)


@case("f", "Two clusters")
def clusters(s):
    note("""With six levels DHONDT, DH_NEAR and CLUSTER give each cluster one score; with 16 CLUSTER
    grades them like RANGE, DHONDT and DH_NEAR still pull the near one to the top.""")
    ballots([[0, .04, .12, .88, .95, 1]], s.of((2, 6, 16)), s)


@case("g", "Exact ties and halves")
def ties(s):
    note("""Tied gaps give DHONDT's spare steps to the closer ones; an exact half rounds to even
    (CLUSTER: as the floating point falls). Both on sets of no area.""")
    ballots([[0, .25, .5, .75, 1], [0, .5, 1]], s.of((2, 4, 6)), s)


@case("h", "The divisor delta")
def divisor(s):
    note("A lower delta gives the small gaps more steps; with two levels delta does not matter.")
    r, rules = [.1, .5, .65, .8, .9, 1], (DHONDT, HYBRID)
    out = table("L", "delta", *(NAMES[rule] for rule in rules), title=distances(r))
    for L in s.of((2, 6)):
        for k, d in enumerate((0.5, 0.65, 0.8, 1.0)):
            out.add_row(str(L) if k == 0 else "", f"{d:g}", *(ballot(r, L, rule, replace(s, delta=d)) for rule in rules),
                        end_section=k == 3)
    console.print(out)


@case("i", "Many levels")
def many_levels(s):
    note("""Mean |score / (L - 1) - (1 - x)| over random voters: all but AVG tend to 0; AVG tends
    to its own part of the way, bent at the mean (last row).""")
    r = random_distances(s, 5)
    lo, hi = r.min(axis=1, keepdims=True), r.max(axis=1, keepdims=True)
    linear, bent = (hi - r) / (hi - lo), _avg_part(r)
    levels = (2, 3, 4, 6, 11, 16)
    part = {L: {rule: scores(r, L, rule, s) / (L - 1) for rule in ORDER} for L in levels}
    out = table("rule", *(f"L={L}" for L in levels))
    for rule in ORDER:
        out.add_row(NAMES[rule], *(f"{np.abs(part[L][rule] - linear).mean():.3f}" for L in levels),
                    end_section=rule == ORDER[-1])
    out.add_row("avg, to its own", *(f"{np.abs(part[L][AVG] - bent).mean():.3f}" for L in levels))
    console.print(out)


WALK = np.array([[.25, .30], [.40, .70], [.45, .62], [.60, .45], [.80, .25], [.75, .80]])  # candidates


def walk_distances(n=100_001):
    """(n, 6): a voter walking the diagonal from (0, 0) to (1, 1) past the candidates WALK."""
    points = np.linspace(0, 1, n)[:, None] * np.ones(2)
    return np.sqrt(((points[:, None, :] - WALK) ** 2).sum(axis=-1))


@case("j", "A voter walking across the plane")
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


@case("k", "Identities over random voters")
def identities(s):
    r = random_distances(s, 6)
    levels = range(2, MAX_LEVELS + 1)
    approve = {rule: scores(r, 2, rule, s) for rule in (RANGE, DHONDT, HYBRID)}
    out = table("identity", "voters for whom it fails")
    out.add_row("two levels: DH_NEAR approves no one RANGE or DHONDT leaves out",
                share(np.mean((approve[HYBRID] > np.minimum(approve[RANGE], approve[DHONDT])).any(axis=1))))
    out.add_row("three candidates: DHONDT with delta 0.5 is RANGE, any L", share(max(
        np.mean((from_distances(r[:, :3], L, DHONDT, 0.5) != from_distances(r[:, :3], L, RANGE)).any(axis=1))
        for L in levels)))
    out.add_row("CLUSTER with mu = 0 is RANGE, any L", share(max(
        np.mean((cluster_scores(r, L, 0.0, s.kappa) != from_distances(r, L, RANGE)).any(axis=1)) for L in levels)))
    console.print(out)


@case("l", "CLUSTER: cohesion of the gaps, and the cost of a split")
def cluster_rule(s):
    note("""w: how firmly each gap holds a cluster (0 at a border). Every level counts, so the near
    clone 0.95, 0.951 does not hide the two clusters. Then CLUSTER at other mu; mu = 0 is RANGE.""")
    out = table("distances", "gaps", "w", title=f"kappa = {s.kappa:g}")
    for r in ([0, .19, .39, .59, .78, 1], [0, .44, .54, 1], [0, 6, 8, 9, 20], [0, .04, .12, .88, .95, 1],
              [0, .04, .12, .88, .95, .951, 1], [0, .3, .6, .65, 1]):
        g, w = cohesion(r, s.kappa)
        out.add_row(distances(r), " ".join(f"{v:.3g}" for v in g), " ".join(f"{v:.2f}" for v in w))
    console.print(out)
    mus = (0.0, 0.05, MU, 0.2)
    out = table("distances", "L", *(f"mu = {mu:g}" for mu in mus))
    for r in ([0, .44, .54, 1], [0, .04, .12, .88, .95, 1]):
        levels = s.of((2, 6, 11, 16))
        for k, L in enumerate(levels):
            out.add_row(distances(r) if k == 0 else "", str(L), *(ballot(r, L, CLUSTER, replace(s, mu=mu)) for mu in mus),
                        end_section=k == len(levels) - 1)
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


def b_sweep(L, rule, s, n=1200):
    """(n, 4): the ballots of case B for a from 0.39 to 0.51."""
    a = np.linspace(.39, .51, n)
    return scores(np.stack([np.zeros_like(a), a, a + .1, np.ones_like(a)], axis=-1), L, rule, s)


def is_cluster(r, members, kappa):
    """Whether the candidates `members` are neighbours by distance whose gaps hold a cluster."""
    values = np.unique(r)
    where = np.searchsorted(values, r)
    lo, hi = where[members].min(), where[members].max()
    return set(np.nonzero((where >= lo) & (where <= hi))[0]) == set(members) and (cohesion(r, kappa)[1][lo:hi] > 0).all()


@criterion(knife_edge, "no gap stands out: RANGE's ballot, L = 2 and 6")
def _(rule, s):
    rows = ([0, .21, .41, .61, .81, 1], [0, .19, .39, .59, .78, 1], [0, .2, .4, .6, .8, 1])
    return all((scores(r, L, rule, s) == scores(r, L, RANGE, s)).all() for r in rows for L in (2, 6))


@criterion(cluster_between, "L = 2: one approval changes at a time, through 1 1 0 0")
def _(rule, s):
    b = b_sweep(2, rule, s)
    return ((b[1:] != b[:-1]).sum(axis=1) <= 1).all() and (b[:, 1] != b[:, 2]).any()


@criterion(cluster_between, "L = 6: the cluster at most a level apart")
def _(rule, s):
    b = b_sweep(6, rule, s)
    return (np.abs(b[:, 1] - b[:, 2]) <= 1).all()


@criterion(cluster_between, "L = 16: graded like RANGE")
def _(rule, s):
    return (b_sweep(16, rule, s) == b_sweep(16, RANGE, s)).all()


@criterion(far_gap, "0 6 8 9 21, L = 6: 6, 8 and 9 not all one score")
def _(rule, s):  # 21, not 20: no score exactly half way
    return len(set(scores([0, 6, 8, 9, 21], 6, rule, s)[1:4])) > 1


@criterion(clones, "an exact copy changes no other score, L = 2 and 6")
def _(rule, s):
    r = random_distances(s, 6)
    copy = r[np.arange(len(r)), np.random.default_rng(SEED + 1).integers(0, 6, len(r))]
    after = np.concatenate([r, copy[:, None]], axis=1)
    return all((scores(after, L, rule, s)[:, :6] == scores(r, L, rule, s)).all() for L in (2, 6))


@shown(extremist, "0 1 2 3 21, near four, L = 6")
def _(rule, s):  # 21, not 20: no score exactly half way
    return styled(scores([0, 1, 2, 3, 21], 6, rule, s)[:4], 6)


@shown(clusters, "two clusters, L = 6")
def _(rule, s):
    return styled(scores([0, .04, .12, .88, .95, 1], 6, rule, s), 6)


@shown(clusters, "two clusters, L = 16")
def _(rule, s):
    return styled(scores([0, .04, .12, .88, .95, 1], 16, rule, s), 16)


@criterion(many_levels, "L = 16: within a quarter level of 1 - x on average")
def _(rule, s):
    r = random_distances(s, 5)
    lo, hi = r.min(axis=1, keepdims=True), r.max(axis=1, keepdims=True)
    return np.abs(scores(r, 16, rule, s) - 15 * (hi - r) / (hi - lo)).mean() <= 0.25


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


@case("m", "Summary")
def summary(s):
    note("Criteria pass or fail; properties are shown, not judged.")
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
    levels: list[int] = typer.Option([], "--levels", "-l", help="Levels of every ballot table; default each case's."),
    voters: int = typer.Option(100_000, "--voters", "-n", help="Random voters."),
    list_cases: bool = typer.Option(False, "--list", help="List the cases."),
) -> None:
    """Edge cases of the five score rules, side by side."""
    if list_cases:
        out = table("case", "title")
        for letter, (title, _) in CASES.items():
            out.add_row(letter, title)
        console.print(out)
        return
    cases = [c.lower() for c in cases] or list(CASES)
    if unknown := [c for c in cases if c not in CASES]:
        raise typer.BadParameter(f"unknown {', '.join(unknown)}; choose from {', '.join(CASES)}", param_hint="--case")
    if not (0 < delta < np.inf and 0 <= mu < np.inf and 0 < kappa < np.inf and voters >= 1):
        raise typer.BadParameter("delta and kappa above 0, mu 0 or more, voters 1 or more")
    if not all(2 <= L <= MAX_LEVELS for L in levels):
        raise typer.BadParameter(f"levels must be 2 to {MAX_LEVELS}", param_hint="--levels")

    settings = Settings(delta, mu, kappa, tuple(levels), voters)
    console.print(f"delta {delta:g}   mu {mu:g}   kappa {kappa:g}   voters {voters}", style="bright_black")
    for letter in cases:
        title, function = CASES[letter]
        console.print()
        console.rule(f"[bold]{letter.upper()}. {title}", align="left")
        function(settings)


if __name__ == "__main__":
    app()
