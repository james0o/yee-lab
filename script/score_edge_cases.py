"""Edge cases of the score rules of yeelab.score: the voters for whom RANGE, AVG, DHONDT
and HYBRID give the most different ballots, and the properties behind them.

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
The notes describe the defaults of --delta and --levels.
"""

from dataclasses import dataclass

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from yeelab.score import AVG, DELTA, DHONDT, HYBRID, MAX_LEVELS, RANGE, _avg_part, from_distances, scored

ORDER = (RANGE, AVG, DHONDT, HYBRID)
SEED = 1
VOTERS = 100_000  # default of --voters
console = Console()
app = typer.Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})


@dataclass(frozen=True)
class Settings:
    delta: float  # of DHONDT and HYBRID
    levels: tuple[int, ...]  # of every table of ballots; () for each case's own
    voters: int  # of the cases that draw random voters

    def of(self, levels):
        """The levels of a table whose own are `levels`."""
        return self.levels or levels

# ---------------------------------------------------------------- Output


def score_style(score, top):
    if score == top:
        return "bold green"
    if score == 0:
        return "bright_black"
    return "green" if 2 * score >= top else "yellow"


def styled(scores, levels) -> Text:
    """The scores of a ballot with `levels` levels, each in its style; with two levels
    the number approved follows."""
    width = len(str(levels - 1))
    text = Text()
    for k, s in enumerate(scores):
        text.append(" " if k else "")
        text.append(f"{s:{width}d}", score_style(s, levels - 1))
    if levels == 2:
        text.append(f"  ({scores.sum()})", "cyan")
    return text


def ballot(r, levels, rule, delta) -> Text:
    """The ballot of the rule for the distances r (C,)."""
    return styled(from_distances(np.asarray(r, dtype=float), levels, rule, delta), levels)


def distances(r):
    return " ".join(f"{d:g}" for d in r)


def short(x):
    """x in [0, 1] with three digits and no leading zero."""
    return "0" if x == 0 else "1" if x == 1 else f"{x:.3f}".removeprefix("0")


def note(text):
    console.print(Padding(" ".join(text.split()), (0, 0, 1, 2)), highlight=False)


def table(*columns, title=None) -> Table:
    return Table(*columns, title=title, box=box.ROUNDED, header_style="bold", title_justify="left")


def ballots(rows, levels, delta, title=None):
    """Prints the ballot of each rule for each of the distances `rows`, at each number of
    levels."""
    out = table("distances", "L", *ORDER, title=title)
    for r in rows:
        for k, L in enumerate(levels):
            out.add_row(distances(r) if k == 0 else "", str(L), *(ballot(r, L, rule, delta) for rule in ORDER),
                        end_section=k == len(levels) - 1)
    console.print(out)


def share(value):
    """A share as a percentage, green where it is 0."""
    return Text(f"{value:.1%}", "green" if value == 0 else "yellow")

# ---------------------------------------------------------------- Helpers


def middle_of_three(levels, rule, delta, n=200_001):
    """x (n,) and the score (n,) of the middle of three candidates at 0, x and 1."""
    x = np.linspace(0, 1, n)
    r = np.stack([np.zeros_like(x), x, np.ones_like(x)], axis=-1)
    return x, from_distances(r, levels, rule, delta)[:, 1]


def bands(levels, rule, delta):
    """{score: (from x, to x)} of the middle of three candidates at 0, x and 1."""
    x, s = middle_of_three(levels, rule, delta)
    at = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    return {int(s[i]): (x[i], end) for i, end in zip(at, np.r_[x[at[1:]], 1.0])}

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
closest alone to all but the farthest; HYBRID does the same within the closer half.
RANGE and AVG change one candidate by one point at a time. With six levels the five
steps fall on five nearly equal gaps, one each, and the knife edge of DHONDT is gone
(HYBRID, with three steps for three gaps, still moves); it is back wherever two gaps
compete for a step, as for the two middle candidates of the second table.
""")
    ballots([[0, .21, .41, .61, .81, 1], [0, .19, .39, .59, .78, 1]], s.of((2, 6)), s.delta)
    ballots([[0, .44, .54, 1], [0, .46, .56, 1]], s.of((6,)), s.delta)


@case("b", "The largest gap among the far candidates")
def far_gap(s: Settings):
    note("""
DHONDT spends its steps on the largest gap even where it separates candidates the voter
does not consider: 0, 6, 8, 9, 20 has the gap 11 from 9 to 20, and DHONDT does not tell
6, 8 and 9 apart. HYBRID looks for the gap only below the midrange (10) and approves the
closest alone, though 6 is only 30% of the way. Where the largest gap lies in the closer
half, as for 0, 1, 7, 8, 10, the two approve the same.
""")
    ballots([[0, 6, 8, 9, 20], [0, 1, 7, 8, 10]], s.of((2, 6)), s.delta)


@case("c", "Clones: only AVG reacts")
def clones(s: Settings):
    note("""
An exact copy of a candidate adds a gap of zero (DHONDT, HYBRID) and moves neither the
closest nor the farthest (RANGE), so no other score changes. It does move the mean
distance and with it every AVG score: copies of the favourite make the voter stingier
towards the candidate at 0.3, copies of the farthest more generous towards the one at
0.6. Under AVG a faction gains by nominating more similar candidates.
""")
    ballots([[0, .3, 1], [0, 0, 0, .3, 1], [0, .6, 1], [0, .6, 1, 1, 1]], s.of((2, 6)), s.delta)


@case("d", "An extremist sets the scale")
def extremist(s: Settings):
    note("""
One candidate far from all the others pushes them to the top of the scale under every
rule. DHONDT and HYBRID collapse entirely: one huge gap (for HYBRID the gap to the
midrange) takes every step. Closer than the mean AVG is RANGE with twice the mean
distance in place of the farthest; one extremist adds only 1/C to the mean, so AVG
holds out a little (case N), but every copy of it pulls the mean further, the mechanism
of case C. The second table: the scores of 0, 1, 2 and 3 with k copies of the extremist
at 20; from eleven on, AVG compresses them more than RANGE.
""")
    ballots([[0, 1, 2, 3], [0, 1, 2, 3, 20]], s.of((2, 6)), s.delta)
    out = table("k", "L", *ORDER, title="the four near candidates, k extremists at 20")
    levels = s.of((6,))
    for k in (1, 2, 6, 11):
        for j, L in enumerate(levels):
            r = np.array([0, 1, 2, 3] + [20] * k, dtype=float)
            out.add_row(str(k) if j == 0 else "", str(L),
                        *(styled(from_distances(r, L, rule, s.delta)[:4], L) for rule in ORDER),
                        end_section=j == len(levels) - 1)
    console.print(out)


@case("e", "Three candidates, the middle at x")
def three(s: Settings):
    note("""
One interior candidate and no structure of gaps: the rules differ only in how they
round. Each cell is the range of x in which the middle candidate gets that score. With
two levels HYBRID approves it only below x = 1/4, the others below 1/2. DHONDT with
delta = 0.5 (Sainte-Lague) is RANGE for three candidates.
""")
    rows = [(rule, rule, s.delta) for rule in ORDER] + [(f"dhondt, delta {d:g}", DHONDT, d) for d in (0.5, 1.0)]
    for L in s.of((2, 6, 7)):
        out = table("rule", *(f"score {v}" for v in range(L - 1, -1, -1)), title=f"L = {L}")
        out.columns[0].no_wrap = True
        for label, rule, delta in rows:
            band = bands(L, rule, delta)
            out.add_row(label, *(f"{short(band[v][0])}-{short(band[v][1])}" if v in band else "-"
                                 for v in range(L - 1, -1, -1)),
                        end_section=label == ORDER[-1])
        console.print(out)


@case("f", "HYBRID and the parity of the levels")
def parity(s: Settings):
    note("""
HYBRID gives the closer half ceil(T / 2) of the T = L - 1 steps and the farther half
floor(T / 2): with an even number of levels the farther half has one step less, and the
scores are lower than those of the other rules. The table: the mean part of the top
score of the middle of three candidates, x uniform in [0, 1]; the other rules are
symmetric about x = 1/2 and give 0.5.
""")
    levels = range(2, 11)
    out = table("rule", *(f"L={L}" for L in levels))
    for rule in ORDER:
        means = [middle_of_three(L, rule, s.delta)[1].mean() / (L - 1) for L in levels]
        out.add_row(rule, *(Text(f"{m:.3f}", "" if abs(m - 0.5) < 5e-4 else "yellow") for m in means))
    console.print(out)


@case("g", "A candidate added between the closest and the farthest")
def interior(s: Settings):
    note("""
RANGE scores depend only on the closest and the farthest candidate, so a candidate added
between them changes no other score; HYBRID scores neither, where it lies beyond the
midrange. The new candidate is the last of each row. In the first table one at 0.75
raises the mean, and AVG approves 0.45; it splits the gap 0.45 to 1 at which DHONDT
cut, and DHONDT drops 0.45. In the second one at 0.35 splits the gap 0.2 to the
midrange, the largest of HYBRID's closer half, and HYBRID drops 0.2; DHONDT's largest
gap is now 0.65 to 1, and it approves 0.65. How often this happens: case N.
""")
    ballots([[0, .15, .45, 1], [0, .15, .45, 1, .75]], s.of((2, 6)), s.delta)
    ballots([[0, .2, .65, 1], [0, .2, .65, 1, .35]], s.of((2,)), s.delta)


@case("h", "Two clusters")
def clusters(s: Settings):
    note("""
DHONDT and HYBRID give every step to the gap between the clusters: the top score to the
near cluster, 0 to the far one. RANGE and AVG still grade within each cluster.
""")
    ballots([[0, .04, .12, .88, .95, 1]], s.of((2, 6)), s.delta)


@case("i", "Exact ties and halves")
def ties(s: Settings):
    note("""
Equal gaps tie for every step and the first gap, the closest, wins, so DHONDT gives the
closer candidates the spare steps. A score exactly between two is rounded to the even
one, so the middle of 0, .5, 1 goes down with 2 and 6 levels and up with 4. Both happen
only on sets of no area in the plane.
""")
    ballots([[0, .25, .5, .75, 1]], s.of((2, 6)), s.delta)
    ballots([[0, .5, 1]], s.of((2, 4, 6)), s.delta)


@case("j", "The divisor delta")
def divisor(s: Settings):
    note("""
The distances of math.typ: one large gap of 0.4 and small ones of 0.15 and 0.1. A
lower delta gives the small gaps more steps; HYBRID, whose closer half has one gap of
0.4 and one of 0.05, does not move. With two levels delta does not matter. This case
sets delta itself.
""")
    r = [.1, .5, .65, .8, .9, 1]
    deltas = (0.5, 0.65, 0.8, 1.0)
    out = table("L", "delta", RANGE, DHONDT, HYBRID, title=distances(r))
    for L in s.of((2, 6)):
        for k, d in enumerate(deltas):
            out.add_row(str(L) if k == 0 else "", f"{d:g}", ballot(r, L, RANGE, d), ballot(r, L, DHONDT, d),
                        ballot(r, L, HYBRID, d), end_section=k == len(deltas) - 1)
    console.print(out)


@case("k", "A voter far from everyone")
def alienated(s: Settings):
    note("""
Every rule depends on the distances only up to scale and shift: a voter 100 to 101 from
every candidate uses the whole scale like one at 0 to 1. All four share this.
""")
    ballots([[0, .2, .5, 1], [100, 100.2, 100.5, 101]], s.of((2, 6)), s.delta)


@case("l", "Many levels")
def many_levels(s: Settings, count=5):
    note("""
Mean |score / (L - 1) - (1 - x)| over random voters, five candidates at uniform
distances: RANGE, DHONDT and HYBRID tend to the part of the way 1 - x, AVG does not; it
tends to its own part of the way, bent at the mean distance (the last row).
""")
    r = np.random.default_rng(SEED).random((s.voters, count))
    lo, hi = r.min(axis=1, keepdims=True), r.max(axis=1, keepdims=True)
    linear, bent = (hi - r) / (hi - lo), _avg_part(r)
    levels = (2, 3, 4, 6, 11, 16)
    part = {L: {rule: from_distances(r, L, rule, s.delta) / (L - 1) for rule in ORDER} for L in levels}
    out = table("rule", *(f"L={L}" for L in levels))
    for rule in ORDER:
        out.add_row(rule, *(f"{np.abs(part[L][rule] - linear).mean():.3f}" for L in levels),
                    end_section=rule == ORDER[-1])
    out.add_row("avg, to its own", *(f"{np.abs(part[L][AVG] - bent).mean():.3f}" for L in levels))
    console.print(out)


@case("m", "A voter walking across the plane")
def walk(s: Settings, n=100_001):
    note("""
Six candidates; a voter walks the diagonal from (0, 0) to (1, 1) in 100 000 steps. Per
rule: how often some score changes, the most candidates whose score changes in one
step, and the share of the changes that change more than one. RANGE and AVG change one
candidate at a time; DHONDT and HYBRID move whole blocks where a step passes from one
gap to another.
""")
    candidates = np.array([[.25, .30], [.40, .70], [.45, .62], [.60, .45], [.80, .25], [.75, .80]])
    points = np.linspace(0, 1, n)[:, None] * np.ones(2)
    out = table("L", "rule", "changes", "most at once", "several at once")
    levels = s.of((2, 6))
    for L in levels:
        for rule in ORDER:
            changed = (np.diff(scored(points, candidates, L, rule, s.delta), axis=0) != 0).sum(axis=1)
            events = changed[changed > 0]
            out.add_row(str(L) if rule == ORDER[0] else "", rule, str(events.size),
                        Text(str(events.max()), "green" if events.max() == 1 else "yellow"),
                        share(np.mean(events > 1)), end_section=rule == ORDER[-1])
    console.print(out)


@case("n", "Properties over random voters")
def properties(s: Settings, count=6):
    note("""
Six candidates at uniform distances. With two levels HYBRID approves no one that RANGE
or DHONDT leaves out; for three candidates DHONDT with delta = 0.5 is RANGE at any
number of levels. Then the share of the voters for whom a candidate added to the six
changes the score of one of the six, and the mean part of the top score they give the
six before and after an extremist joins, ten times their span beyond the farthest.
""")
    rng = np.random.default_rng(SEED)
    r = rng.random((s.voters, count))
    approve = {rule: from_distances(r, 2, rule, s.delta) for rule in ORDER}
    outside = (approve[HYBRID] > np.minimum(approve[RANGE], approve[DHONDT])).any(axis=1)
    differ = max(np.mean((from_distances(r[:, :3], L, DHONDT, 0.5) != from_distances(r[:, :3], L, RANGE)).any(axis=1))
                 for L in range(2, MAX_LEVELS + 1))
    checks = table("check", "voters for whom it fails")
    checks.add_row("two levels: HYBRID approves no one RANGE or DHONDT leaves out", share(outside.mean()))
    checks.add_row(f"three candidates: DHONDT with delta 0.5 is RANGE, L = 2 to {MAX_LEVELS}", share(differ))
    console.print(checks)

    lo, hi = r.min(axis=1), r.max(axis=1)
    added = {
        "an exact copy of one of them": r[np.arange(s.voters), rng.integers(0, count, s.voters)],
        "one below the midrange": lo + rng.uniform(0, .5, s.voters) * (hi - lo),
        "one beyond the midrange": lo + rng.uniform(.5, 1, s.voters) * (hi - lo),
    }
    extreme = np.concatenate([r, (hi + 10 * (hi - lo))[:, None]], axis=1)
    out = table("L", "added to the six", *ORDER, title="voters whose ballot of the six changes")
    out.columns[1].no_wrap = True
    for L in s.of((2, 6)):
        before = {rule: from_distances(r, L, rule, s.delta) for rule in ORDER}
        for k, (label, d) in enumerate(added.items()):
            after = np.concatenate([r, d[:, None]], axis=1)
            out.add_row(str(L) if k == 0 else "", label,
                        *(share(np.mean((from_distances(after, L, rule, s.delta)[:, :count] != before[rule]).any(axis=1)))
                          for rule in ORDER))
        out.add_row("", "an extremist: mean part of the top",
                    *(f"{before[rule].mean() / (L - 1):.2f} to "
                      f"{from_distances(extreme, L, rule, s.delta)[:, :count].mean() / (L - 1):.2f}" for rule in ORDER),
                    end_section=True)
    console.print(out)

# ---------------------------------------------------------------- Command


@app.command()
def main(
    cases: list[str] = typer.Option(
        [], "--case", "-c", help=f"Case to show, by letter ({', '.join(CASES)}), repeatable; default every one."),
    delta: float = typer.Option(DELTA, "--delta", "-d", help="Divisor of DHONDT and HYBRID."),
    levels: list[int] = typer.Option(
        [], "--levels", "-l", help="Levels of every table of ballots, repeatable; default each case's own."),
    voters: int = typer.Option(VOTERS, "--voters", "-n", help="Random voters of cases L and N."),
    list_cases: bool = typer.Option(False, "--list", help="List the cases and stop."),
) -> None:
    """Edge cases of the four score rules of yeelab.score, side by side."""
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
    if not all(2 <= L <= MAX_LEVELS for L in levels):
        raise typer.BadParameter(f"levels must be 2 to {MAX_LEVELS}", param_hint="--levels")
    if voters < 1:
        raise typer.BadParameter("at least one voter", param_hint="--voters")

    settings = Settings(delta, tuple(levels), voters)
    console.print(f"[bold]Score rules[/bold] {', '.join(ORDER)}   delta {delta:g}   "
                  f"levels {', '.join(map(str, levels)) or 'of each case'}   random voters {voters}", highlight=False)
    for letter in cases:
        title, function = CASES[letter]
        console.print()
        console.rule(f"[bold]{letter.upper()}. {title}", align="left")
        function(settings)


if __name__ == "__main__":
    app()
