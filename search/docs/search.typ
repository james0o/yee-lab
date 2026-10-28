#set document(title: "yee-search")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 15pt)
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show table: it => block(breakable: false, it)
#show raw.where(block: false): set text(size: 9pt)
#show raw.where(block: true): set text(size: 7.5pt)

#let data = json("data.json")
#let num(x, digits: 4) = {
  let s = str(calc.round(x, digits: digits))
  if not s.contains(".") { s += "." }
  s + "0" * (digits - s.split(".").at(1).len())
}
#let pct(x) = str(calc.round(100 * x)) + " %"
#let colours = (random: rgb("#3b6fb6"), hand: rgb("#d9822b"), ood: rgb("#3a9a5b"))

#align(center)[
  #text(size: 17pt, weight: "bold")[yee-search]
  #v(0.2em)
  #text(size: 11pt)[Searching for voting methods that resist burial: what is scored, on which
    configurations, and how the searches find methods]
]

#v(1em)

`search/` is a project of its own (its own `uv` lock and environment) that builds voting
methods out of the blocks of `yeelab.build` and looks for those that stay close to the
nearest candidate both when every voter is sincere and when 30 % of the voters bury.
@ch-methods describes the methods it can build, @ch-score how one method is scored,
@ch-sets the configurations of candidates it is scored on, @ch-fast how the scoring is made
fast, @ch-gp and @ch-conditional the two searches, @ch-select how the methods found are
chosen and checked, @ch-run how to run it all, and @ch-results the methods of the web UI's
row "Found by search".

= Methods as trees of blocks <ch-methods>

A method is a tree of blocks of `yeelab.build` (docs/math.pdf, chapter on the voting
methods): ballots (`Plurality`, `BordaCount`, `AntiPlurality`, `Score`), totals
(`Tally` of the mean or the median, `Weakest`), pairwise comparisons (`Margins`,
`ScoreComparisons`, `StrongestPaths`), methods (`Highest`, `Eliminate`, `Unbeaten`,
`DescendingCoalitions`), sets of candidates (`Highest(..., n=2)`, `Smith`, the union `|`
of two winners) and conditions (`A == B`, a measure `Top` or `Lead` against a threshold,
`&`, `|`) for `Conditional(condition, when_true, when_false)`.

The grammar (`grammar.py`) types every slot of a tree, so that only blocks that fit go
there: a *method* elects one candidate, *among* chooses candidates for another method to
choose from, *if* is the condition of a Conditional (a condition, or a method that is true
where it elects someone), a *measure* is `Top` or `Lead` of a tally of the mean, and so
on. The constants are few on purpose: the score ballots `Score(2)` and `Score(6)` with
power 1.5, the tie weights 0 and 1 of the score runoff, the thresholds 0.02, 0.05, 0.1,
0.2, 0.3, 0.4, 0.5, 0.55, 0.6 and 0.7, two or three finalists, and Smith sets of a
majority of 0.5, 0.55 or 0.6. D20 (two votes for the first two, `Highest(Tally(D21(...)))`)
is a method of its own, a leaf: weak alone, it may tell where others are right inside a
Conditional. `Undecided()` is left out: a method must try to elect someone everywhere. A
tree is at most 12 deep and has at most 60 nodes.

= Scoring a method <ch-score>

== The configuration and its diagram

A configuration is a set of candidates in the unit square and the deviation $d$ of the
Beta voters around every point (the voter model of YeeLab with the spread rule `rms`).
The method is run on the shares of the voters at every point of a grid of $G$ cells per
side, $(G+2)^2$ points with both walls: $G = 16$ for training, 24 for validation and 32 for
the test (@ch-select). Every point of the grid is drawn at the *geometric median* of its
voters, as the web UI starts ("Point in diagram is $p = arg min_q E||V - q||$"): the
winners are those of the voters of the grid, only placed elsewhere, and near the walls the
diagram leaves a strip empty (docs/math.pdf, the geometric diagram).

== Burial

The method is scored twice: with every voter sincere, and with 30 % of the voters burying
at every point, as the web UI's "Burying voters" does. The frontrunners at a point are the
two with the most first choices (`Burial`'s default poll), and a burying voter ranks the one
of them they like less last and gives it 0 on a score ballot. FPTP and the two-round
system cannot be moved so: burial changes neither the first choices nor the head-to-head of
the two frontrunners.

== The error

At a point $x$ that elects the candidate $w$ the loss is
$ ell(x, w) = r / (r + s), quad r = ||x - w|| - min_c ||x - c||, $
$s$ the median distance between two candidates: 0 where the nearest candidate wins, close
to 1 far from it. A point without a winner (a cycle, a draw) counts as won by the farthest
candidate, so that a method cannot leave out the points it decides badly. The error of the
method over an area $A$ is relative to a candidate drawn at random at every point,
$ e_A = (sum_(x in A) a_x ell(x, w_x)) / (sum_(x in A) a_x 1/C sum_c ell(x, c)), $
$a_x$ the area of the point. It is taken over the region around the candidates (the
dashed outline of the web UI's "metrics region") and over the whole diagram, and
$ e = 0.8 e_"region" + 0.2 e_"whole", $
mostly where the candidates are, but so that a method that elects the wrong candidate far
from all of them pays too. The web UI shows $sqrt(e_A)$ of each area; the search does not take the
root. For configuration $k$, $e_k$ is this error with sincere voters and $b_k$ with 30 %
burying.

== The objective

$ J = op("trimmed mean", limits: #true)_k (0.25 e_k + 0.75 b_k) + 0.0001 dot "complexity", $
lower being better. Burial weighs three times as much as the sincere error: a method that
only resists burial is FPTP, whose sincere error is large. The trimmed mean leaves out, for
each method, the 5 % of the configurations it does best on and the 5 % it does worst on
($floor(0.05 K)$ of each end of $K$ configurations), so that a few configurations do not
decide the order; the sincere and the burial error reported with $J$ are the means over the
same kept configurations. A method that fails or takes too long has $J = infinity$.

== Complexity <sec-complexity>

The complexity counts the blocks of a method, so that of two methods that err alike the
simpler one wins. Let $T(m)$ be the tree of blocks of $m$ and $D(m)$ the set of its
_distinct_ subtrees: two blocks are the same if they have the same type, parameters and
children (the same `repr`), so that a block that recurs is one element of $D(m)$. Let
$F$ be the free types `Tally`, `Weakest` and `Union`, and $C(m) subset D(m)$ the distinct
Conditionals. The depth $delta(c)$ of a Conditional $c$ is the number of Conditionals
that enclose it in $T(m)$, in a branch or in a condition, found where the tree is first
entered. Then
$ "complexity"(m) = underbrace(|{b in D(m) : "type"(b) in.not F}|, "blocks paid for") + 2 sum_(c in C(m)) delta(c). $ <eq-complexity>
The cost in $J$ is $0.0001$ times this.

- *Distinct blocks.* A method in a condition and in a branch, or the score ballot that four
  blocks read, is paid for once, as `methods.py` writes it once by name.
- *Free types.* Every ballot needs a `Tally` to give a winner, every pairwise comparison a
  `Weakest` or an `Unbeaten`, and a `Union` (`A | B`) only lists two finalists, so none is
  paid for: only what is inside them. A union costs the blocks of $A$ and $B$ and nothing
  for the `|`.
- *Nesting.* A Conditional pays 2 more for each Conditional around it. A chain
  $"Conditional"(c_1, A, "Conditional"(c_2, B, "Conditional"(c_3, C, D)))$ has depths
  $0, 1, 2$ and pays $2(0 + 1 + 2) = 6$ beyond its blocks; a balanced one, a root with a
  Conditional in each branch, has depths $0, 1, 1$ and pays $2(0 + 1 + 1) = 4$. One
  Conditional of a combined condition, $"Conditional"((x == y) | (y != z), A, B)$, has
  depth 0 and pays nothing beyond its blocks: a `CombinedCondition` and two
  `WinnerComparison`s are blocks, but no Conditional is nested.
- Parameters (`n=2`, `how="mean"`, a threshold, a tie weight) are free.

*`among=`.* It is a field of `Highest`, `Eliminate` and `Unbeaten` like any other, so its
value is a subtree counted by the same rule. `Highest(Tally(Plurality()), n=2)` costs two
blocks (`Highest` and `Plurality`, the `Tally` being free); `among=A | B` costs the blocks
of $A$ and $B$ together, those they share counted once.

*Examples.* STAR, `Unbeaten(ScoreComparisons(Score(6)), among=Highest(Tally(Score(6)), n=2))`,
has the distinct blocks `Unbeaten`, `ScoreComparisons`, `Score(6)` (read twice, paid once),
`Highest` and `Tally`; without the free `Tally`, $"complexity" = 4$. `search_6` has 20
distinct blocks: 5 different `Highest`, 5 different `Tally`, and one each of `Conditional`,
`WinnerComparison`, `Unbeaten`, `ScoreComparisons`, `Union`, `Eliminate`, `BordaCount`,
`Plurality`, `Score(6)` and `D21`. Without the 5 `Tally` and the `Union` it is
$20 - 6 = 14$; its only Conditional is the root, $delta = 0$, so
$"complexity" = 14$ and the cost $0.0014$. `search_5` has a Conditional in the else-branch
of its root, $delta = 1$, which adds $2$ beyond its blocks, giving 20.

#figure(table(columns: 4, align: (left, right, right, right),
  [method], [blocks], [distinct blocks], [complexity],
  ..for (name, c) in data.complexity { (raw(name), [#c.blocks], [#c.distinct], [#c.complexity]) }),
  caption: [Blocks, distinct blocks and complexity of some methods of `methods.py`: the chains
    `adhoc` and `con_fptp` pay for their nesting.])

= The configurations <ch-sets>

== Random configurations and hand-made families

A random configuration has 3, 4, 5 or 6 candidates drawn uniformly in $[0.1, 0.9]^2$, each
two at least 0.08 apart, and a deviation of 0.1, 0.2 or 0.3. The hand-made families are the
hard cases: _dark horse_ (two frontrunners close to the centre and a weak third far away,
who can win where the voters of each frontrunner bury the other), _dark horse four_ (one
more candidate on the other side), _center squeeze_ (a Condorcet winner at the centre with
few first choices, three around it), _clones_ (two nearly equal candidates and two others),
_collinear_ (candidates on a line), _near walls_ (candidates at the walls and corners),
_triangle_ (an equilateral triangle, where burial easily makes a cycle) and _outlier_ (three
close together, one far away). A family is a function of its parameters, so that the sets
hold other instances of it: turned about the centre, spread wider or narrower, with other
voters.

== The sets

#figure(table(columns: 4, align: (left, left, center, right),
  [set], [made of], [grid], [configurations],
  [train], [12 random (seed 1); the families at 0°, scale 1, deviation 0.2], [16], [#data.sets.train.len()],
  [wide], [train; 36 random (seed 6); the families at 200°, scale 1.05, deviation 0.12], [16], [#data.sets.wide.len()],
  [validation], [20 random (seed 2); the families at 90°, scale 0.9, deviation 0.15], [24], [#data.sets.validation.len()],
  [test], [200 random (seed 3); the families at 30°, 135° and 250° (scales 1.1, 0.8, 1, deviations 0.1, 0.25, 0.3);
    out of the training distribution: 20 of 7 or 8 candidates (seed 4), 20 of deviation 0.05 or 0.4 (seed 5)],
    [32], [#data.sets.test.len()]),
  caption: [The sets of configurations, which share none. GP trains on `train`, the search of
    Conditionals on `wide`.])

A configuration is left out of every set where a candidate is nearer than 0.1 to a wall, or
nearer than 0.05 to the strip the geometric diagram leaves empty along the walls: such a
candidate is drawn at the very edge of the web UI's diagram or beyond it. The strip is
wider for wider voters:

#figure(table(columns: data.strip.len() + 1,
  [deviation], ..data.strip.keys().map(k => [#k]),
  [empty strip], ..data.strip.values().map(v => [#v])),
  caption: [How far from a wall, at most, the geometric diagram starts.])

The rule leaves out #data.dropped.train.len() configurations of the training set,
#data.dropped.wide.len() of `wide`, #data.dropped.validation.len() of the validation set and
#data.dropped.test.len() of the test set: every instance of _near walls_, two of _outlier_ in
the test, and random ones with a candidate near a wall and wide voters.

#let config-box(c, side: 1.08cm) = box(width: side, height: side, stroke: 0.4pt + luma(170), clip: true, {
  for p in c.candidates {
    place(dx: p.at(0) * side - 1.3pt, dy: (1 - p.at(1)) * side - 1.3pt,
      circle(radius: 1.3pt, fill: colours.at(c.kind)))
  }
  place(bottom + right, dx: -1pt, dy: -0.5pt, text(4.5pt, fill: luma(110))[#c.deviation])
})
#let set-figure(name) = figure(
  grid(columns: 14, gutter: 2pt, ..data.sets.at(name).map(c => config-box(c))),
  caption: [The #name set: the candidates of every configuration in the unit square and its
    deviation; random #box(circle(radius: 2pt, fill: colours.random)), hand-made
    #box(circle(radius: 2pt, fill: colours.hand)), out of the training distribution
    #box(circle(radius: 2pt, fill: colours.ood)).])

#set-figure("train")
#set-figure("validation")
#set-figure("wide")
#set-figure("test")

= Making the scoring fast <ch-fast>

Scoring a method runs its blocks on the voters of every configuration, sincerely and buried
at every pair of frontrunners (about five a configuration): a few hundred small numpy
calls on a grid of 324 points, where the calls cost more than the arithmetic. What the
search does about it:

- The shares of the voters every method of the grammar can read are computed once for every
  configuration and kept (`Prepared`), each method getting only those it reads; what burial
  takes from a score ballot is remembered too (`MemoGrid`).
- What every subtree that elects or chooses candidates gives is remembered for each
  configuration and pair of frontrunners (`Memo`, 256 MB a process, the least recently used
  going first): a child of the genetic programming keeps most of its parent, an enumeration
  most of the smaller methods. A block runs its children on the voters it got, so a subtree
  gives the same wherever it is. About half of the work of a genetic programming is so
  saved: 214 ms a method became 105 ms on 18 configurations, sincerely and buried, with the
  same errors to the last digit, and the maps of the library of @ch-conditional are 2--4 ms a
  method and configuration.
- The compiled kernels run in the worker's own thread, not handed to a thread pool chunk by
  chunk.
- The workers (4 by default, a process each) score methods in parallel, each with a time
  limit (60 s).

= Genetic programming <ch-gp>

`yee-search gp` evolves a population of 1000 trees of the grammar. The first population is
the methods of `yeelab.build.methods` the grammar can build, the best of a search of
Conditionals where given (`--seed-from`, 20 of each of its rankings) and random trees of
depth 3 to 8. Each generation keeps the 5 best, adds 25 % of new random trees, and fills the
rest with children of parents chosen by tournaments of 5: a crossover swaps subtrees of the
same type (half of the children), a mutation regrows a subtree, redraws one node or hoists a
subtree into the place of its ancestor. Of the methods that err alike on every
configuration only the best stays, and where the best has not improved for 30 generations
the whole population starts again from random trees. The 50 best methods of the training
(the hall of fame) are scored on the validation set after every generation. A run writes
every method scored (`evals.jsonl`), so that a run stopped (by a power cut: `resume.sh`)
goes on without scoring any method twice.

= The search of Conditionals <ch-conditional>

`yee-search conditional` scores every Conditional of two methods of a library, and of two
conditions and three methods, exactly, without running a block for any of them.

== The library and its maps

The library is every method of the grammar of at most 8 blocks (enumerated, not drawn), the
methods of `yeelab.build.methods` the grammar can build and the best of the GP runs. For each
one and each configuration its *map* is kept: the winner at every point, sincerely and
buried (`maps/`, shared by every run). Methods with the same maps everywhere are one: of
8712 methods 818 differ on `wide`.

== Why the maps are enough

Burial is pointwise: the poll names the frontrunners at a point from the sincere first
choices, whatever the method, and the method then runs on the buried voters of the point.
So a Conditional buried elects where its condition is true buried what its first branch
elects buried, else what its second does,
$ "Burial"("Conditional"(c, A, B)) = "where"(c_"buried", A_"buried", B_"buried"), $
and a condition $X == Y$ is where the maps of $X$ and $Y$ agree. The maps of the library give
the maps of every Conditional of it; the tests check them against the blocks run.

== Every Conditional at once

The loss of a point depends only on its winner, so with $l_X$ the losses of method $X$ at
every point (of every configuration, sincere and buried, weighted as in $J$),
$ J("Conditional"(c, A, B)) = c dot l_A + (1 - c) dot l_B + "cost", $
a part of $A$ alone and a part of $B$ alone. For a condition, the best two methods are the
best $A$ of the first part and the best $B$ of the second, and the products $c dot l_X$ of
all conditions with all methods are one matrix product: about $1.5 dot 10^12$ Conditionals
in 20 minutes, under a nanosecond each. The conditions are where a method elects someone,
`Top` or `Lead` of a tally against every threshold, $X == Y$ of every two methods of the
library (about 300 000 that differ), and `&` of two of the best of those, each maybe
negated: with $A$ and $B$ swapped, every `&` and `|`.

Two levels separate as well:
$ J("Conditional"(c_1, A_1, "Conditional"(c_2, A_2, B))) = & c_1 dot l_(A_1) + (not c_1 and c_2) dot l_(A_2) \
  & + (not c_1 and not c_2) dot l_B + "cost", $
three parts of one method each, so one matrix product a pair of conditions gives the best
three methods. Every two of about 1900 conditions are tried: the best of all, and the best
agreements $X == Y$ of FPTP, the two-round system and D20 with any other method.

== Exact scores

The matrix products sum over the points, which the trimmed mean and the complexity do not:
the products leave out the 5 % easiest and hardest configurations for all methods alike
(by the median error of the library, as a method's best and worst configurations are mostly
those) and pay for every block of each part. The best of them, by the products, are then
scored exactly. They are ranked five ways: by $J$, without the cost of the complexity
("raw", to see whether a Conditional helps at all), the methods of the library alone, and
two levels by $J$ and raw.

= Choosing the methods <ch-select>

Of millions of methods, the best on a few configurations is partly chance. So the best 200
of each ranking on the training set are scored on the validation set, the best 20 of those
on the test set, and only the test set, of other configurations and a finer grid, tells how
good a method is. The test set has been used to choose the methods of the web UI's rows,
which makes their test scores somewhat too good; a new set of the same kind, used once, tells
how much. Two methods are best compared configuration by configuration: the table of
@ch-results gives on how many test configurations a method buries better than Star Two Round
and on how many worse.

= Running it <ch-run>

```sh
cd search
uv run yee-search conditional --out runs/cond --maps runs/2026-10-27-conditional/maps
uv run yee-search gp --out runs/night --until 06:00 --seed-from runs/cond
uv run yee-search report runs/night        # its best and the known methods on the test set
./night.sh 06:00 4                         # all three, until 06:00, with 4 workers
```

A run of the search of Conditionals writes `library.json`, `train.json`,
`validation.json`, `test.json` and `summary.md` (the table of @ch-results); a GP run
`evals.jsonl`, `validation.jsonl`, `generations.jsonl`, `hall.json` and the report.
`night.sh` keeps the computer from sleeping while it runs, but not with the lid closed in
KDE. Each GP worker holds about 2.5 GB of voters and remembered subtrees.

= The methods found <ch-results>

#let shown = data.scores.pairs().sorted(key: p => p.at(1).J)
#figure(table(columns: 7, align: (left, right, right, right, right, right, right),
  [method], [$J$], [$J$ without cost], [sincere], [burial], [burial better / worse than STR], [complexity],
  ..for (name, s) in shown {
    (raw(name), [#num(s.J)], [#num(s.raw)], [#num(s.sincere)], [#num(s.burial)],
     [#pct(s.better) / #pct(s.worse)], [#s.size])
  }),
  caption: [The methods of the web UI's rows on the test set (230 configurations, grid 32):
    sincere and burial are the means over the configurations $J$ keeps.])

The three found last, as `yeelab.build.methods` writes them (`nanson`, `d20`, `mj`, `M` and
`score_ballot` are its names):

```python
search_5 = Conditional(
    nanson == Highest(Tally(score_ballot), among=Highest(Tally(Plurality()), n=2)),
    when_true=nanson,
    when_false=Conditional(Unbeaten(ScoreComparisons(score_ballot)) == d20,
        when_true=d20,
        when_false=Highest(Tally(score_ballot, statistic="median"), among=Highest(Weakest(M), n=2))))
search_6 = Conditional(
    Highest(Tally(score_ballot), among=Highest(Tally(Plurality()), n=2)) == nanson,
    when_true=nanson,
    when_false=Unbeaten(ScoreComparisons(score_ballot),
        among=Highest(Tally(score_ballot, statistic="median"),
                      among=Highest(Tally(score_ballot), n=2)) | d20))
search_7 = Unbeaten(ScoreComparisons(score_ballot, tie_weight=1.0),
                    among=mj | Unbeaten(ScoreComparisons(score_ballot)))
```

`search_6`: Nanson's winner where the higher mean score of the two with the most first
choices agrees with it, else a runoff on the scores of each ballot between the higher
median score of the two best by mean score and D20's winner. `search_7`: the runoff of
`search_1` between the majority judgment and the Condorcet winner of the score comparisons,
counting each ballot's ranking where its scores tie.
