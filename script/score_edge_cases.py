"""Edge cases of the score rules of yeelab.score: the voters for whom RANGE, AVG, DHONDT,
DH_NEAR and CLUSTER give the most different ballots, and the properties behind them.

DH_NEAR is the rule HYBRID of yeelab.score: DHONDT among the candidates closer than the
midrange and RANGE for the others; the name tells it from other mixes of the two.

CLUSTER is only here, not yet in yeelab.score: the ballot is the best rounding of the
part of the way of RANGE, part_i, with a cost for splitting a cluster,

    argmin  sum_i (s_i / T - part_i)^2  +  mu * sum_k w_k * max(0, 1 - T g_k) * [s_(k) != s_(k+1)]

over the scores s, T = levels - 1, that never give a closer candidate less, the top
score to the closest and 0 to the farthest. g_k are the gaps between neighbouring
distinct distances as parts of the span; candidates at the same distance count once.
w_k is how firmly gap k holds a cluster together: a run of neighbouring candidates is a
cluster as strong as 1 - kappa * (its largest inner gap) / (the smaller gap around it),
0 at least, and a gap is as cohesive as the strongest run it lies in. A split costs
less as a gap nears a level, 1 / T, and nothing from there, so with many levels CLUSTER
grades like RANGE. Dynamic programming over the candidates in order finds the minimum.

Run from the repository root:

    uv run python script/score_edge_cases.py              # every case
    uv run python script/score_edge_cases.py -c a -c g    # some of them
    uv run python script/score_edge_cases.py --list       # what they are
    uv run python script/score_edge_cases.py -h

A case is a voter's distances to the candidates. Every rule depends on them only up to a
common scale and shift, so they are given on whatever scale reads best; x is a distance
on the scale from 0 at the closest candidate to 1 at the farthest. A ballot is the
scores, 0 to levels - 1, in the order the distances are given, the top score bold green
and 0 grey; with two levels 1 is approved, and the number approved follows in brackets.
The notes describe the defaults of --delta, --mu, --kappa and --levels.
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
NAMES = {RANGE: "range", AVG: "avg", DHONDT: "dhondt", HYBRID: "dh_near", CLUSTER: "cluster"}  # as printed
LABELS = tuple(NAMES[rule] for rule in ORDER)
MU = 0.1  # default of --mu
KAPPA = 2.0  # default of --kappa
SEED = 1
VOTERS = 100_000  # default of --voters
console = Console()
app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})


@dataclass(frozen=True)
class Settings:
    delta: float  # of DHONDT and DH_NEAR
    mu: float  # of CLUSTER: the cost of a split
    kappa: float  # of CLUSTER: how much farther a cluster's neighbours are than its members
    levels: tuple[int, ...]  # of every table of ballots; () for each case's own
    voters: int  # of the cases that draw random voters

    def of(self, levels):
        """The levels of a table whose own are `levels`."""
        return self.levels or levels

# ---------------------------------------------------------------- CLUSTER


@njit
def _cohesion(g, kappa, w):
    """w[k] = how firmly the gap g[k] holds a cluster together, for the gaps g (m,)
    between neighbouring distinct distances in order: the strongest run of candidates
    i..j it lies in, 1 - kappa * (largest of g[i:j]) / (smaller of g[i - 1] and g[j])."""
    m = g.size
    w[:] = 0.0
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            if i == 0 and j == m:
                continue  # every candidate: no gap around them
            around = np.inf
            if i > 0:
                around = g[i - 1]
            if j < m:
                around = min(around, g[j])
            inner = 0.0
            for k in range(i, j):
                inner = max(inner, g[k])
            strength = 1.0 - kappa * inner / around
            for k in range(i, j):
                w[k] = max(w[k], strength)


@njit
def _cluster_ballot(r, top, mu, kappa, out):
    """The CLUSTER ballot of the voter at the distances r (C,) into out (C,)."""
    C = r.size
    order = np.argsort(r)
    values = np.empty(C)  # the distinct distances, closest first
    group = np.empty(C, dtype=np.int64)  # of each candidate: the index of its distance
    n = 0
    for k in range(C):
        c = order[k]
        if n == 0 or r[c] > values[n - 1]:
            values[n] = r[c]
            n += 1
        group[c] = n - 1
    if n == 1:
        out[:] = top
        return
    span = values[n - 1] - values[0]
    g = np.empty(n - 1)
    for k in range(n - 1):
        g[k] = (values[k + 1] - values[k]) / span
    w = np.empty(n - 1)
    _cohesion(g, kappa, w)
    cost = np.full(top + 1, np.inf)  # of the best scores so far, by the score of the last
    cost[top] = 0.0  # the closest gets the top score
    new = np.empty(top + 1)
    back = np.zeros((n, top + 1), dtype=np.int64)
    for k in range(1, n):
        part = (values[n - 1] - values[k]) / span
        split = mu * w[k - 1] * max(0.0, 1.0 - top * g[k - 1])
        for v in range(top + 1):
            best, arg = cost[v], v  # the same score as the one before costs nothing
            for u in range(v + 1, top + 1):
                if cost[u] + split < best:
                    best, arg = cost[u] + split, u
            new[v] = best + (v / top - part) ** 2
            back[k, v] = arg
        cost[:] = new
    scores = np.empty(n, dtype=np.int64)
    v = 0  # the farthest gets 0
    for k in range(n - 1, -1, -1):
        scores[k] = v
        v = back[k, v]
    for c in range(C):
        out[c] = scores[group[c]]


@njit
def _cluster_ballots(r, top, mu, kappa, out):
    for i in range(r.shape[0]):
        _cluster_ballot(r[i], top, mu, kappa, out[i])


def cluster_scores(r, levels, mu=MU, kappa=KAPPA) -> np.ndarray:
    """The CLUSTER scores, int (..., C) from 0 to levels - 1, of voters at the distances
    r (..., C)."""
    r = np.asarray(r, dtype=np.float64)
    flat = np.ascontiguousarray(r.reshape(-1, r.shape[-1]))
    out = np.empty(flat.shape, dtype=np.int64)
    _cluster_ballots(flat, levels - 1, float(mu), float(kappa), out)
    return out.reshape(r.shape)


def cohesion(r, kappa=KAPPA):
    """(gaps, w) of the voter at the distances r (C,): the gaps between neighbouring
    distinct distances as parts of the span, and how firmly each holds a cluster."""
    values = np.unique(np.asarray(r, dtype=np.float64))
    g = np.diff(values) / (values[-1] - values[0])
    w = np.empty_like(g)
    _cohesion(g, float(kappa), w)
    return g, w


def scores(r, levels, rule, s: Settings) -> np.ndarray:
    """The scores, int (..., C), of the rule for the distances r (..., C)."""
    r = np.asarray(r, dtype=np.float64)
    if rule == CLUSTER:
        return cluster_scores(r, levels, s.mu, s.kappa)
    return from_distances(r, levels, rule, s.delta)

# ---------------------------------------------------------------- Output


def score_style(score, top):
    if score == top:
        return "bold green"
    if score == 0:
        return "bright_black"
    return "green" if 2 * score >= top else "yellow"


def styled(values, levels) -> Text:
    """The scores of a ballot with `levels` levels, each in its style; with two levels
    the number approved follows."""
    width = len(str(levels - 1))
    text = Text()
    for k, v in enumerate(values):
        text.append(" " if k else "")
        text.append(f"{v:{width}d}", score_style(v, levels - 1))
    if levels == 2:
        text.append(f"  ({values.sum()})", "cyan")
    return text


def ballot(r, levels, rule, s: Settings) -> Text:
    """The ballot of the rule for the distances r (C,)."""
    return styled(scores(r, levels, rule, s), levels)


def distances(r):
    return " ".join(f"{d:g}" for d in r)


def note(text):
    console.print(Padding(" ".join(text.split()), (0, 0, 1, 2)), highlight=False)


def table(*columns, title=None) -> Table:
    """A table whose first column alone wraps where the console is too narrow."""
    out = Table(*columns, title=title, box=box.ROUNDED, header_style="bold", title_justify="left")
    for column in out.columns[1:]:
        column.no_wrap = True
    return out


def ballots(rows, levels, s: Settings, title=None):
    """Prints the ballot of each rule for each of the distances `rows`, at each number of
    levels."""
    out = table("distances", "L", *LABELS, title=title)
    for r in rows:
        for k, L in enumerate(levels):
            out.add_row(distances(r) if k == 0 else "", str(L), *(ballot(r, L, rule, s) for rule in ORDER),
                        end_section=k == len(levels) - 1)
    console.print(out)


def share(value):
    """A share as a percentage, green where it is 0."""
    return Text(f"{value:.1%}", "green" if value == 0 else "yellow")

# ---------------------------------------------------------------- Cases

CASES = {}  # letter: (title, the function that prints the case)


def case(letter, title):
    def register(function):
        CASES[letter] = (title, function)
        return function
    return register


@case("a", "Nearly equal gaps: DHONDT on a knife edge")
def knife_edge(s: Settings):
    note("""
With two levels the one step goes to the largest gap, so where gaps nearly tie a tiny
move of the voter flips a whole block of candidates at once. Between the first two rows
each interior candidate moves by at most 0.03, and DHONDT goes from approving the
closest alone to all but the farthest; DH_NEAR does the same within the closer half.
RANGE and AVG change one candidate by one point at a time. With six levels the five
steps fall on five nearly equal gaps, one each, and the knife edge of DHONDT is gone
(DH_NEAR, with three steps for three gaps, still moves); it is back wherever two gaps
compete for a step: case B. No gap stands out, so CLUSTER finds no cluster and is RANGE.
""")
    ballots([[0, .21, .41, .61, .81, 1], [0, .19, .39, .59, .78, 1]], s.of((2, 6)), s)


@case("b", "A cluster between two nearly equal gaps")
def cluster_between(s: Settings):
    note("""
The distances 0, a, a + 0.1, 1: a cluster of two between the gaps a and 0.9 - a, which
tie at a = 0.45. DHONDT keeps the cluster together and moves it as a block where the
gaps tie: with two levels it approves both below 0.45 and neither above, with six it
moves both by a point at once. RANGE splits the cluster at the midrange, and each of
the two crosses it on its own, the window as wide as the cluster (a from 0.4 to 0.5);
AVG splits it at the mean, which moves along with the cluster. DH_NEAR approves the
closest alone throughout, as in its closer half the gap 0 to a is the largest. A rule
cannot both treat a cluster alike and change one candidate at a time: somewhere the
block has to move at once, and a rule only chooses where, and how narrow the window.
CLUSTER chooses by the number of levels: with two, keeping the cluster together would
put it at the top or the bottom though it sits in the middle, so there it splits it,
for a from about 0.42 to 0.48, and keeps it whole further out; with three it sits on
the middle level and never moves; with six it moves as a block like DHONDT; with sixteen
the gap of 0.1 is more than a level, a split no longer costs anything, and CLUSTER
grades like RANGE.
""")
    for L in s.of((2, 3, 6, 16)):
        out = table("distances", "gaps", *LABELS, title=f"L = {L}")
        for a in (.39, .41, .43, .44, .46, .47, .49, .51):
            r = [0, a, round(a + .1, 2), 1]
            out.add_row(distances(r), f"{a:g} 0.1 {.9 - a:.2f}", *(ballot(r, L, rule, s) for rule in ORDER))
        console.print(out)


@case("c", "The largest gap among the far candidates")
def far_gap(s: Settings):
    note("""
DHONDT spends its steps on the largest gap even where it separates candidates the voter
does not consider: 0, 6, 8, 9, 20 has the gap 11 from 9 to 20, and DHONDT does not tell
6, 8 and 9 apart. DH_NEAR looks for the gap only below the midrange (10) and approves the
closest alone, though 6 is only 30% of the way. Where the largest gap lies in the closer
half, as for 0, 1, 7, 8, 10, the two approve the same. CLUSTER sees 6, 8 and 9 as a
cluster, between the gaps 6 and 11, and with six levels gives them one score like
DHONDT; it does not cut early like DH_NEAR.
""")
    ballots([[0, 6, 8, 9, 20], [0, 1, 7, 8, 10]], s.of((2, 6)), s)


@case("d", "Clones: only AVG reacts")
def clones(s: Settings):
    note("""
An exact copy of a candidate adds a gap of zero (DHONDT, DH_NEAR), moves neither the
closest nor the farthest (RANGE) and counts once (CLUSTER), so no other score changes.
It does move the mean distance and with it every AVG score: copies of the favourite
make the voter stingier towards the candidate at 0.3, copies of the farthest more
generous towards the one at 0.6. Under AVG a faction gains by nominating more similar
candidates.
""")
    ballots([[0, .3, 1], [0, 0, 0, .3, 1], [0, .6, 1], [0, .6, 1, 1, 1]], s.of((2, 6)), s)


@case("e", "An extremist sets the scale")
def extremist(s: Settings):
    note("""
One candidate far from all the others pushes them to the top of the scale under every
rule. DHONDT and DH_NEAR collapse entirely: one huge gap (for DH_NEAR the gap to the
midrange) takes every step. Closer than the mean AVG is RANGE with twice the mean
distance in place of the farthest; one extremist adds only 1/C to the mean, so AVG
holds out a little (case M), but every copy of it pulls the mean further, the mechanism
of case D. Without the extremist CLUSTER is RANGE, the gaps all alike; with it, 0, 1, 2
and 3 are one cluster, and CLUSTER gives them one score like DHONDT. The second table:
the scores of 0, 1, 2 and 3 with k copies of the extremist at 20; from eleven on, AVG
compresses them more than RANGE.
""")
    ballots([[0, 1, 2, 3], [0, 1, 2, 3, 20]], s.of((2, 6)), s)
    out = table("k", "L", *LABELS, title="the four near candidates, k extremists at 20")
    levels = s.of((6,))
    for k in (1, 2, 6, 11):
        for j, L in enumerate(levels):
            r = np.array([0, 1, 2, 3] + [20] * k, dtype=float)
            out.add_row(str(k) if j == 0 else "", str(L), *(styled(scores(r, L, rule, s)[:4], L) for rule in ORDER),
                        end_section=j == len(levels) - 1)
    console.print(out)


@case("f", "A candidate added between the closest and the farthest")
def interior(s: Settings):
    note("""
RANGE scores depend only on the closest and the farthest candidate, so a candidate added
between them changes no other score; DH_NEAR scores neither, where it lies beyond the
midrange. The new candidate is the last of each row. In the first table one at 0.75
raises the mean, and AVG approves 0.45; it splits the gap 0.45 to 1 at which DHONDT
cut, and DHONDT drops 0.45. In the second one at 0.35 splits the gap 0.2 to the
midrange, the largest of DH_NEAR's closer half, and DH_NEAR drops 0.2; DHONDT's largest
gap is now 0.65 to 1, and it approves 0.65. CLUSTER changes only where the new candidate
makes or breaks a cluster, in neither table here. How often this happens: case M.
""")
    ballots([[0, .15, .45, 1], [0, .15, .45, 1, .75]], s.of((2, 6)), s)
    ballots([[0, .2, .65, 1], [0, .2, .65, 1, .35]], s.of((2,)), s)


@case("g", "Two clusters")
def clusters(s: Settings):
    note("""
DHONDT, DH_NEAR and CLUSTER give every step to the gap between the clusters: the top
score to the near cluster, 0 to the far one. RANGE and AVG still grade within each
cluster. With sixteen levels the inner gaps of 0.07 and 0.08 are more than a level, and
CLUSTER grades across them again; those of 0.04 and 0.05 are still less, and hold.
""")
    ballots([[0, .04, .12, .88, .95, 1]], s.of((2, 6, 16)), s)


@case("h", "Exact ties and halves")
def ties(s: Settings):
    note("""
Equal gaps tie for every step and the first gap, the closest, wins, so DHONDT gives the
closer candidates the spare steps. A score exactly between two is rounded to the even
one, so the middle of 0, .5, 1 goes down with 2 and 6 levels and up with 4; for
CLUSTER, which rounds by the least squared error, the rounding error of the floating
point picks one of the two equally good scores. Both happen only on sets of no area in
the plane.
""")
    ballots([[0, .25, .5, .75, 1]], s.of((2, 6)), s)
    ballots([[0, .5, 1]], s.of((2, 4, 6)), s)


@case("i", "The divisor delta")
def divisor(s: Settings):
    note("""
The distances of math.typ: one large gap of 0.4 and small ones of 0.15 and 0.1. A
lower delta gives the small gaps more steps; DH_NEAR, whose closer half has one gap of
0.4 and one of 0.05, does not move. With two levels delta does not matter. This case
sets delta itself; RANGE and CLUSTER do not use it.
""")
    r = [.1, .5, .65, .8, .9, 1]
    deltas = (0.5, 0.65, 0.8, 1.0)
    out = table("L", "delta", NAMES[RANGE], NAMES[DHONDT], NAMES[HYBRID], NAMES[CLUSTER], title=distances(r))
    for L in s.of((2, 6)):
        for k, d in enumerate(deltas):
            with_delta = replace(s, delta=d)
            out.add_row(str(L) if k == 0 else "", f"{d:g}", *(ballot(r, L, rule, with_delta)
                                                              for rule in (RANGE, DHONDT, HYBRID, CLUSTER)),
                        end_section=k == len(deltas) - 1)
    console.print(out)


@case("j", "A voter far from everyone")
def alienated(s: Settings):
    note("""
Every rule depends on the distances only up to scale and shift: a voter 100 to 101 from
every candidate uses the whole scale like one at 0 to 1. All five share this.
""")
    ballots([[0, .2, .5, 1], [100, 100.2, 100.5, 101]], s.of((2, 6)), s)


@case("k", "Many levels")
def many_levels(s: Settings, count=5):
    note("""
Mean |score / (L - 1) - (1 - x)| over random voters, five candidates at uniform
distances: RANGE, DHONDT, DH_NEAR and CLUSTER tend to the part of the way 1 - x, AVG does
not; it tends to its own part of the way, bent at the mean distance (the last row).
""")
    r = np.random.default_rng(SEED).random((s.voters, count))
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


@case("l", "A voter walking across the plane")
def walk(s: Settings, n=100_001):
    note("""
Six candidates; a voter walks the diagonal from (0, 0) to (1, 1) in 100 000 steps. Per
rule: how often some score changes, the most candidates whose score changes in one
step, and the share of the changes that change more than one. RANGE and AVG change one
candidate at a time; DHONDT and DH_NEAR move whole blocks where a step passes from one
gap to another, CLUSTER where a cluster moves together.
""")
    candidates = np.array([[.25, .30], [.40, .70], [.45, .62], [.60, .45], [.80, .25], [.75, .80]])
    points = np.linspace(0, 1, n)[:, None] * np.ones(2)
    r = np.sqrt(((points[:, None, :] - candidates) ** 2).sum(axis=-1))
    out = table("L", "rule", "changes", "most at once", "several at once")
    for L in s.of((2, 6)):
        for rule in ORDER:
            changed = (np.diff(scores(r, L, rule, s), axis=0) != 0).sum(axis=1)
            events = changed[changed > 0]
            out.add_row(str(L) if rule == ORDER[0] else "", NAMES[rule], str(events.size),
                        Text(str(events.max()), "green" if events.max() == 1 else "yellow"),
                        share(np.mean(events > 1)), end_section=rule == ORDER[-1])
    console.print(out)


@case("m", "Properties over random voters")
def properties(s: Settings, count=6):
    note("""
Six candidates at uniform distances. With two levels DH_NEAR approves no one that RANGE
or DHONDT leaves out; for three candidates DHONDT with delta = 0.5 is RANGE at any
number of levels, and so is CLUSTER with mu = 0 for any number of candidates. Then the
share of the voters for whom a candidate added to the six changes the score of one of
the six, and the mean part of the top score they give the six before and after an
extremist joins, ten times their span beyond the farthest.
""")
    rng = np.random.default_rng(SEED)
    r = rng.random((s.voters, count))
    approve = {rule: scores(r, 2, rule, s) for rule in ORDER}
    outside = (approve[HYBRID] > np.minimum(approve[RANGE], approve[DHONDT])).any(axis=1)
    levels = range(2, MAX_LEVELS + 1)
    sainte_lague = max(np.mean((from_distances(r[:, :3], L, DHONDT, 0.5) != from_distances(r[:, :3], L, RANGE))
                               .any(axis=1)) for L in levels)
    free = max(np.mean((cluster_scores(r, L, 0.0, s.kappa) != from_distances(r, L, RANGE)).any(axis=1))
               for L in levels)
    checks = table("check", "voters for whom it fails")
    checks.add_row("two levels: DH_NEAR approves no one RANGE or DHONDT leaves out", share(outside.mean()))
    checks.add_row(f"three candidates: DHONDT with delta 0.5 is RANGE, L = 2 to {MAX_LEVELS}", share(sainte_lague))
    checks.add_row(f"CLUSTER with mu = 0 is RANGE, L = 2 to {MAX_LEVELS}", share(free))
    console.print(checks)

    lo, hi = r.min(axis=1), r.max(axis=1)
    added = {
        "an exact copy of one of them": r[np.arange(s.voters), rng.integers(0, count, s.voters)],
        "one below the midrange": lo + rng.uniform(0, .5, s.voters) * (hi - lo),
        "one beyond the midrange": lo + rng.uniform(.5, 1, s.voters) * (hi - lo),
    }
    extreme = np.concatenate([r, (hi + 10 * (hi - lo))[:, None]], axis=1)
    out = table("L", "added to the six", *LABELS, title="voters whose ballot of the six changes")
    for L in s.of((2, 6)):
        before = {rule: scores(r, L, rule, s) for rule in ORDER}
        for k, (label, d) in enumerate(added.items()):
            after = np.concatenate([r, d[:, None]], axis=1)
            out.add_row(str(L) if k == 0 else "", label,
                        *(share(np.mean((scores(after, L, rule, s)[:, :count] != before[rule]).any(axis=1)))
                          for rule in ORDER))
        out.add_row("", "an extremist: mean part of the top",
                    *(f"{before[rule].mean() / (L - 1):.2f} to "
                      f"{scores(extreme, L, rule, s)[:, :count].mean() / (L - 1):.2f}" for rule in ORDER),
                    end_section=True)
    console.print(out)


@case("n", "CLUSTER: which gaps hold a cluster, and what a split costs")
def cluster_rule(s: Settings):
    note("""
The first table: the gaps between neighbouring distinct distances as parts of the span,
and how firmly each holds a cluster together (w, 0 at a border or where no gap stands
out). Every level of clusters counts at once, so a near clone, 0.95 and 0.951, is a
cluster of its own inside the far cluster and does not hide the two clusters. The second
table: CLUSTER at other costs of a split; mu = 0 is RANGE, and the larger mu the more
levels a cluster keeps together over.
""")
    rows = [[0, .19, .39, .59, .78, 1], [0, .44, .54, 1], [0, 6, 8, 9, 20], [0, .04, .12, .88, .95, 1],
            [0, .04, .12, .88, .95, .951, 1], [0, .3, .6, .65, 1], [0, .1, .35, .45, .55, 1]]
    out = table("distances", "gaps", "w", title=f"kappa = {s.kappa:g}")
    for r in rows:
        g, w = cohesion(r, s.kappa)
        out.add_row(distances(r), " ".join(f"{v:.3g}" for v in g),
                    Text(" ".join(f"{v:.2f}" for v in w), "" if w.any() else "bright_black"))
    console.print(out)
    mus = (0.0, 0.05, MU, 0.2)
    out = table("distances", "L", *(f"mu = {mu:g}" for mu in mus))
    for r in ([0, .44, .54, 1], [0, .46, .56, 1], [0, .04, .12, .88, .95, 1], [0, .04, .12, .88, .95, .951, 1]):
        levels = s.of((2, 3, 6, 11, 16))
        for k, L in enumerate(levels):
            out.add_row(distances(r) if k == 0 else "", str(L),
                        *(ballot(r, L, CLUSTER, replace(s, mu=mu)) for mu in mus), end_section=k == len(levels) - 1)
    console.print(out)

# ---------------------------------------------------------------- Command


@app.command()
def main(
    cases: list[str] = typer.Option(
        [], "--case", "-c", help=f"Case to show, by letter ({', '.join(CASES)}), repeatable; default every one."),
    delta: float = typer.Option(DELTA, "--delta", "-d", help="Divisor of DHONDT and DH_NEAR."),
    mu: float = typer.Option(MU, "--mu", help="Cost of splitting a cluster, of CLUSTER; 0 is RANGE."),
    kappa: float = typer.Option(
        KAPPA, "--kappa", help="How much farther a cluster's neighbours must be than its members, of CLUSTER."),
    levels: list[int] = typer.Option(
        [], "--levels", "-l", help="Levels of every table of ballots, repeatable; default each case's own."),
    voters: int = typer.Option(VOTERS, "--voters", "-n", help="Random voters of cases K and M."),
    list_cases: bool = typer.Option(False, "--list", help="List the cases and stop."),
) -> None:
    """Edge cases of the five score rules, side by side."""
    if list_cases:
        out = table("case", "title")
        for letter, (title, _) in CASES.items():
            out.add_row(letter, title)
        console.print(out)
        return
    cases = [c.lower() for c in cases] or list(CASES)
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        raise typer.BadParameter(f"unknown case(s) {', '.join(unknown)}; choose from {', '.join(CASES)}",
                                 param_hint="--case")
    if not 0 < delta < np.inf:
        raise typer.BadParameter(f"delta must be above 0, got {delta}", param_hint="--delta")
    if not 0 <= mu < np.inf:
        raise typer.BadParameter(f"mu must be 0 or more, got {mu}", param_hint="--mu")
    if not 0 < kappa < np.inf:
        raise typer.BadParameter(f"kappa must be above 0, got {kappa}", param_hint="--kappa")
    if not all(2 <= L <= MAX_LEVELS for L in levels):
        raise typer.BadParameter(f"levels must be 2 to {MAX_LEVELS}", param_hint="--levels")
    if voters < 1:
        raise typer.BadParameter("at least one voter", param_hint="--voters")

    settings = Settings(delta, mu, kappa, tuple(levels), voters)
    console.print(f"[bold]Score rules[/bold] {', '.join(LABELS)}   delta {delta:g}   mu {mu:g}   kappa {kappa:g}   "
                  f"levels {', '.join(map(str, levels)) or 'of each case'}   random voters {voters}", highlight=False)
    for letter in cases:
        title, function = CASES[letter]
        console.print()
        console.rule(f"[bold]{letter.upper()}. {title}", align="left")
        function(settings)


if __name__ == "__main__":
    app()
