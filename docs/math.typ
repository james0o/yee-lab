#set document(title: "The mathematics of YeeLab")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 15pt)
#show heading.where(level: 1): set heading(supplement: [Chapter])
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: underline
// Keep every table on one page.
#show table: it => block(breakable: false, it)

#let P = math.upright("P")
#let E = math.upright("E")
#let Beta = math.op("Beta")
#let logit = math.op("logit")
#let expit = math.op("expit")

#align(center)[
  #text(size: 17pt, weight: "bold")[The mathematics of YeeLab]
  #v(0.2em)
  #text(size: 11pt)[Voter models, exact ranking probabilities, voting methods,
    and how Beta and normal voters differ]
]

#v(1em)

This document explains what YeeLab computes and why. @ch-model defines the model: the
Yee diagram, the two voter distributions (normal and Beta) and the spread rules that
fix the Beta distributions. @ch-compute shows how the share of every ranking of the
candidates is computed exactly for every pixel, and @ch-methods how the voting methods
turn these shares into winners. @ch-compare explains why Beta and normal voters give
different diagrams. @ch-spread compares the spread rules and explains why `rms` is the
default. @ch-shapes asks when a candidate's region is convex or in one piece, and how
that relates to the monotonicity of the method. @ch-realtime shows how the web UI draws
the regions as curves fast enough to follow a dragged candidate. @ch-geometric describes
the diagram in which every pixel is drawn at the geometric median of its voters.

The settings that appear in the mathematics, with their defaults (file names are in the
package `src/yeelab/`):

#align(center, table(
  columns: 4,
  align: (left, center, left, left),
  stroke: none,
  table.hline(),
  [quantity], [symbol], [set by], [default],
  table.hline(stroke: 0.5pt),
  [pixels per axis], [$n$], [`--pixels` (`pixels/plot.py`), `PIXELS` (`margin/shares.py`)], [400 / 300],
  [deviation], [$D$], [`--deviation`, _Deviation_ slider], [0.2],
  [voter distribution], [], [`--distribution`, _Voters_], [Beta],
  [Beta spread rule], [], [`--spread`; the web UI always uses `rms`], [`rms`],
  [what a pixel is (Beta, UI)], [], [_Pixel is_, `PIXEL_MEDIAN` (`margin/geometric.py`)], [median along each axis],
  [interpolation nodes per axis], [$N$], [`--nodes` (`NODES`)], [49],
  [Gauss–Legendre points per edge], [$Q$], [`QUAD_NODES`], [24],
  [score levels (UI)], [$L$], [_Score categories_, `SCORE_LEVELS` (`web/app.py`)], [6],
  [power of the score ballot (UI)], [$p$], [_Score power p_, `SCORE_POWER` (`build/methods.py`)], [1.5],
  [voter grid of the score shares], [$K$], [`GRID_CELLS`, `GRID_SUB` (`margin/shares.py`)], [256, 8],
  [contour grid per axis (UI)], [$G$], [`DRAG_GRID` / `FINAL_GRID` (`web/app.py`)], [160 / 320],
  table.hline(),
))

#v(1em)

#outline(indent: auto, depth: 2)

#v(1.5em)

= The model <ch-model>

== Yee diagrams

The political space is the unit square $[0,1]^2$. There are $C$ candidates at fixed
points $c_1, dots, c_C in [0,1]^2$. The square is divided into $n times n$ pixels, and
every pixel is its own election:

- the voters of the pixel are spread around the pixel centre
  $m = ((k + 1/2) slash n, (l + 1/2) slash n)$, $k, l = 0, dots, n - 1$;
- every voter ranks the candidates by Euclidean distance, closest first;
- the pixel is coloured by the winner of a voting method.

A pixel does not have a finite number of voters. Its electorate is a probability
distribution, and all that a voting method needs from it is, for every strict ranking
$r$ of the candidates,

$ P_(k l)(r) = "share of the voters of pixel" (k, l) "whose ranking is" r . $ <eq-share>

@ch-compute computes these shares exactly, without sampling voters, so the diagrams
have no Monte Carlo noise. A voter can only be tied between two candidates on a line
(@sec-cells), and lines have probability zero, so ties are ignored.

If every voter sat exactly at the pixel centre, every method would elect the candidate
nearest to $m$, and the diagram would be the *Voronoi diagram* of the candidates
(`pixels.methods.voronoi`, _Voronoi_ in the web UI). It is the reference against which
the other diagrams are compared.

== Normal voters <sec-normal-model>

In the original Yee diagrams the voters of a pixel are isotropic normal around the
pixel centre,

$ (X, Y) tilde cal(N)(m, sigma^2 I), $

so some of them lie outside the square. The spread is set by the *deviation* $D$, the
mean absolute deviation of each coordinate from the pixel centre. For
$X tilde cal(N)(m_x, sigma^2)$, $E|X - m_x| = sigma sqrt(2 slash pi)$, so

$ sigma = D sqrt(pi / 2) quad (D = 0.2 => sigma = 0.251, quad D = 0.3 => sigma = 0.376) $ <eq-sigma>

(`normal.sigma_from_deviation`).

== Beta voters <sec-beta-model>

YeeLab's own model keeps every voter inside the square. The two coordinates are
independent and Beta distributed,

$ X tilde Beta(a_x, b_x), quad Y tilde Beta(a_y, b_y), $

where $(a_x, b_x)$ depend only on the pixel column and $(a_y, b_y)$ only on the pixel
row. Both come from the same function of the median, so it is enough to study one
coordinate $X tilde Beta(a, b)$ with median $m in (0, 1)$. We write $f, F$ for the pdf
and CDF of $X$ and $g, G$ for those of $Y$. The CDF of $Beta(a, b)$ is the regularised
incomplete beta function $I_x (a, b)$ (`scipy.special.betainc`), its inverse is
`betaincinv`.

A Beta distribution has two parameters, so two conditions fix it:

+ the *median* is the pixel centre,
  $ I_m (a, b) = 1/2 ; $ <eq-median>
+ a *spread rule* fixes how far the voters spread (@sec-centre, @sec-spreads).

The median plays the role that the mean plays for normal voters: a line through $m$
parallel to an axis has exactly half of the pixel's voters on each side. For oblique
lines this is no longer true, and @ch-compare shows that this is the root of most
differences between the two models.

If $X tilde Beta(a, b)$ then $1 - X tilde Beta(b, a)$. So the pixel with median
$1 - m$ has the parameters of median $m$ swapped, and the rules below only need to be
defined for $m >= 1/2$.

== The deviation and the centre pixel <sec-centre>

At the centre pixel, $m = 1/2$, symmetry forces $a = b = a_0$, and every spread rule
fixes $a_0$ by the same condition as the normal model, $E|X - 1/2| = D$. For a
symmetric Beta this has a closed form,

$ E|X - 1/2| = 2^(-2 a_0) / (a_0 B(a_0, a_0)) = D, $ <eq-centre>

which decreases from $1/2$ ($a_0 -> 0$) to $0$ ($a_0 -> oo$), so $0 < D < 1/2$ is
required. It is solved for $a_0$ by bracketing (`centre_shape`). The uniform
distribution, $a_0 = 1$, has $D = 1/4$ and separates two shapes:

- $D < 1/4$ gives $a_0 > 1$: the voters of the centre pixel are bell-shaped, dense in
  the middle and thin at the walls. The default $D = 0.2$ gives $a_0 = 1.724$.
- $D > 1/4$ gives $a_0 < 1$: the density is U-shaped and infinite at both walls, with
  more voters near the walls than in the middle. $D = 0.3$ gives $a_0 = 0.602$
  ($Beta(1/2, 1/2)$ has $D = 1 slash pi approx 0.318$).

Away from the centre, as the median approaches a wall, one of the parameters drops
below $1$ for every $D$ (at $D = 0.2$ with `rms`, $b = 0.42$ at $m = 0.9$), and the
density becomes infinite at that wall. Much of the numerics of @ch-compute is shaped
by this.

== Spread rules <sec-spreads>

=== One free parameter

With the median fixed by @eq-median, one parameter is left. The convenient one is the
*concentration* $kappa = a + b$:

$ "Var" X = (mu (1 - mu)) / (kappa + 1), quad mu = E X = a / kappa . $

For given $kappa$ and $m$, @eq-median determines $a$ and $b$ (@sec-solve). A spread
rule is therefore a curve $kappa(m)$ for $m in [1/2, 1)$ that starts at
$kappa(1/2) = 2 a_0$; a larger $kappa$ means voters closer together.

=== Why the spread must change near a wall

Near a wall, "the same spread for every pixel" cannot hold in every sense. When the
median $m$ is close to $1$, the half of the voters above the median lies within
$1 - m$ of it and contributes almost nothing to any measure of spread; the whole spread
has to come from the other half, the one towards the centre of the square. For the
mean absolute deviation,

$ E|X - m| = 1/2 E[m - X | X < m] + 1/2 E[X - m | X > m] <= 1/2 E[m - X | X < m] + (1 - m) / 2, $

so keeping $E|X - m| = D$ forces

$ E[m - X | X < m] >= 2 D - (1 - m) . $ <eq-push>

At the centre this far half lies at mean distance $D$ from the median; next to a wall
it must lie twice as far. For the power mean $(E|X - m|^p)^(1/p)$ the same argument
gives a factor $2^(1/p)$: $2$ for $p = 1$, $sqrt(2) approx 1.41$ for the root mean
square, $1.26$ for $p = 3$. This holds for every distribution on $[0, 1]$, not only for
Beta. Every rule therefore decides, in effect, how far the far half of the voters is
pushed out near a wall, and this is what the two rules differ in.

=== The two rules

*`rms` (default).* The root mean square distance of the voters from the median is the
same as at the centre pixel,

$ sqrt(E(X - m)^2) = s = 1 / (2 sqrt(2 a_0 + 1)), quad E(X - m)^2 = "Var" X + (mu - m)^2, $ <eq-rms>

where $s$ is the standard deviation of $Beta(a_0, a_0)$ ($s = 0.237$ for $D = 0.2$,
$0.337$ for $D = 0.3$). The square weights distant voters more, so near a wall a few
voters far out make up the spread and the rest of the far half stays where it is. It is
the rule with the clearest meaning, and of the rules compared in @sec-bench its borders
are among the straightest. The web UI uses it for all its Beta voters.

*`mean_abs` (legacy).* The mean absolute deviation is $D$ for every median,
$E|X - m| = D$. This was the first rule, and it is kept to reproduce older results.
Because it keeps the mean absolute deviation fixed, @eq-push applies in full: near a
wall the far half of the voters is pushed twice as far out, towards the opposite wall.
At median $0.98$ the share of voters below $0.1$ is back at $0.192$, more than at the
centre, where `rms` has $0.049$ (@fig-densities). This push bends Condorcet borders,
adds cycles and gives IRV a round edge that `rms` does not have (@ch-spread). It is not
offered in the web UI; `--spread mean_abs` selects it for the plots.

#figure(
  image("figures/spread_densities.png", width: 100%),
  caption: [Density of one coordinate of a pixel's voters, $D = 0.3$, for medians
    moving towards the wall at $1$ (grey: the median). Both rules agree at the centre.
    Near the wall `mean_abs` moves many voters to the opposite wall; `rms` moves far
    fewer.],
) <fig-densities>

#block(breakable: false)[
At $D = 0.3$ (`docs/figures.py`):

#align(center, table(
  columns: 7,
  align: (right, left, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [median], [rule], [$kappa = a + b$], [mean], [$E|X - m|$], [$sqrt(E(X - m)^2)$], [$P(X < 0.1)$],
  table.hline(stroke: 0.5pt),
  [0.50], [all], [1.204], [0.500], [0.300], [0.337], [0.175],
  table.hline(stroke: 0.3pt),
  [0.70], [`rms`], [1.197], [0.621], [0.283], [0.337], [0.095],
  [], [`mean_abs`], [1.012], [0.612], [0.300], [0.355], [0.121],
  table.hline(stroke: 0.3pt),
  [0.90], [`rms`], [1.062], [0.750], [0.229], [0.337], [0.051],
  [], [`mean_abs`], [0.554], [0.685], [0.300], [0.430], [0.146],
  table.hline(stroke: 0.3pt),
  [0.98], [`rms`], [0.799], [0.812], [0.186], [0.337], [0.049],
  [], [`mean_abs`], [0.307], [0.698], [0.300], [0.490], [0.192],
  table.hline(),
))
]

== Solving for the parameters (`beta_params_at`) <sec-solve>

Only the distinct values of $max(m, 1 - m) >= 1/2$ are solved, in increasing order;
medians below $1/2$ get the mirrored parameters.

*The median for a given concentration* (`_b_for_median`). For a fixed $kappa = a + b$
the median of $Beta(kappa - b, b)$ decreases from $1$ to $1/2$ as $b$ grows from $0$ to
$kappa slash 2$, so @eq-median has exactly one root, found by bracketing $b$ (Brent's
method).

*`rms`* (`_solve_rms`). Along the Beta distributions with median $m$, the RMS distance
@eq-rms, which is explicit in $(a, b)$, decreases as $kappa$ grows. An outer bracketing
search over $log kappa$, starting from the solution at the previous median, finds the
$kappa$ whose RMS distance is $s$.

*`mean_abs`* (`_solve_beta`). Split the absolute deviation at the median and use
$F(m) = 1/2$:

$
E|X - m| &= E[X - m] - 2 E[(X - m) bb(1){X < m}] \
         &= mu - m - 2 E[X bb(1){X < m}] + 2 m F(m) \
         &= mu - 2 E[X bb(1){X < m}] .
$

Since $x dot x^(a-1)(1-x)^(b-1) slash B(a,b) = mu dot$ (density of $Beta(a+1, b)$),
$E[X bb(1){X < m}] = mu I_m (a+1, b)$, and therefore

$ E|X - m| = a/(a+b) (1 - 2 I_m (a+1, b)) = D . $ <eq-mad>

@eq-median and @eq-mad are solved together for $(a, b)$ with Levenberg–Marquardt
(`scipy.optimize.root(method="lm")`). A two-dimensional solver needs a good initial
guess, so the medians are solved from the centre solution $(a_0, a_0)$ outwards, each
solution being the initial guess for the next median, and intermediate medians are
inserted wherever two consecutive medians are more than $0.25$ apart in $logit(m)$
(`CONTINUATION_STEP`). This keeps the solver on the correct branch, also for a single
median close to $0$ or $1$.

The tests check that every rule keeps its quantity fixed (to $10^(-9)$, with the
moments computed by independent quadrature), the median (to $10^(-12)$), the mirror
symmetry, and that a lone median gets the same parameters as a full sweep.

#pagebreak()

= Exact ranking probabilities <ch-compute>

This chapter computes the shares (@eq-share) for Beta voters (`ranking_cells.py` and
`pixels/beta.py`, @sec-cells to @sec-interpolation) and for normal voters (`normal.py`
and `pixels/normal.py`, @sec-normal).
The only approximations are Gauss–Legendre quadrature and polynomial interpolation,
both of which converge exponentially fast; the normal model needs no quadrature at all.
For Beta voters the steps are:

+ Solve the Beta parameters at the interpolation nodes (@sec-solve, @sec-interpolation).
+ Cut the square by the $binom(C, 2)$ bisectors into cells, each with one ranking
  (@sec-cells).
+ Write the probability of each cell as a sum of integrals over its edges; each edge is
  shared by two cells (@sec-green).
+ Compute every edge integral for all pairs of node parameters (@sec-edges).
+ Interpolate from the $N times N$ nodes to the $n times n$ pixels (@sec-interpolation).

== Ranking cells (`ranking_cells`) <sec-cells>

=== Bisectors

A voter at $p$ prefers $c_i$ to $c_j$ exactly when

$
|p - c_i|^2 < |p - c_j|^2
quad <==> quad
2 (c_j - c_i) dot p < |c_j|^2 - |c_i|^2 .
$ <eq-bisector>

The boundary of @eq-bisector is the perpendicular bisector of $c_i c_j$, a straight
line. A voter's full ranking is determined by the signs of all $binom(C, 2)$
comparisons, so the ranking can only change when the voter crosses one of these
bisectors.

The $binom(C, 2)$ lines cut the unit square into convex polygons ("cells"). Inside a cell
every comparison has a fixed sign, hence *every point of a cell has the same ranking*.
Points on a bisector have a tie, but lines have zero area and so zero probability.

The crucial observation is that *the cells do not depend on the pixel*: they depend only
on the candidates. Only the probability mass that each pixel's distribution puts on
each cell changes. So

$ P_(k l)(r) = P_(k l)((X, Y) in K_r) = integral.double_(K_r) f_k (x) g_l (y) dif x dif y, $ <eq-cell-prob>

where $K_r$ is the cell with ranking $r$.

=== Construction

The construction starts with the square as a single polygon and, for every pair
$(i, j)$, splits every current polygon into its parts on either side of the bisector
(half-plane clipping, `_clip`, a Sutherland–Hodgman step for one line). Slivers with
area below $10^(-14)$ are discarded. All polygons stay convex and counter-clockwise
(CCW). The ranking of a cell is found by sorting the candidates by distance from the
cell's vertex average, which lies strictly inside the convex cell.

@fig-cells shows the arrangement for the candidates A–E used throughout this document.

// Generated from ranking_cells(CANDIDATES) with the candidates of docs/figures.py.
#let candidates = (
  ("A", (0.6, 0.35)),
  ("B", (0.25, 0.4)),
  ("C", (0.35, 0.3)),
  ("D", (0.5, 0.5)),
  ("E", (0.3, 0.7)),
)
#let cells = (
  (pts: ((0.5400, 0.0000), (0.5550, 0.0000), (0.5250, 0.0750),), rank: "ACBDE"),
  (pts: ((0.5550, 0.0000), (0.9583, 0.0000), (0.4662, 0.3691), (0.5250, 0.0750),), rank: "ACDBE"),
  (pts: ((0.9583, 0.0000), (1.0000, 0.0000), (1.0000, 0.4292), (0.6450, 0.4883), (0.4662, 0.3691),), rank: "ADCBE"),
  (pts: ((1.0000, 0.4292), (1.0000, 0.5844), (0.7404, 0.5519), (0.6450, 0.4883),), rank: "ADCEB"),
  (pts: ((1.0000, 0.5844), (1.0000, 0.7250), (0.7404, 0.5519),), rank: "ADECB"),
  (pts: ((0.4450, 0.5150), (0.4417, 0.4917), (0.4679, 0.5179),), rank: "DABCE"),
  (pts: ((0.4459, 0.5215), (0.4450, 0.5150), (0.4679, 0.5179),), rank: "DABEC"),
  (pts: ((0.6250, 0.6750), (0.4459, 0.5215), (0.4679, 0.5179),), rank: "DAEBC"),
  (pts: ((0.4679, 0.5179), (0.4417, 0.4917), (0.4662, 0.3691), (0.6450, 0.4883),), rank: "DACBE"),
  (pts: ((0.4679, 0.5179), (0.6450, 0.4883), (0.7404, 0.5519),), rank: "DACEB"),
  (pts: ((1.0000, 0.7250), (1.0000, 0.9964), (0.6250, 0.6750), (0.4679, 0.5179), (0.7404, 0.5519),), rank: "DAECB"),
  (pts: ((0.9500, 1.0000), (0.8000, 1.0000), (0.4667, 0.6667), (0.4459, 0.5215), (0.6250, 0.6750),), rank: "DEABC"),
  (pts: ((0.8000, 1.0000), (0.5143, 1.0000), (0.4667, 0.6667),), rank: "EDABC"),
  (pts: ((1.0000, 0.9964), (1.0000, 1.0000), (0.9500, 1.0000), (0.6250, 0.6750),), rank: "DEACB"),
  (pts: ((0.3714, 0.0000), (0.5400, 0.0000), (0.5250, 0.0750), (0.4197, 0.3382),), rank: "CABDE"),
  (pts: ((0.5250, 0.0750), (0.4662, 0.3691), (0.4197, 0.3382),), rank: "CADBE"),
  (pts: ((0.4662, 0.3691), (0.4282, 0.3976), (0.4197, 0.3382),), rank: "CDABE"),
  (pts: ((0.4662, 0.3691), (0.4417, 0.4917), (0.4282, 0.3976),), rank: "DCABE"),
  (pts: ((0.4417, 0.4917), (0.4450, 0.5150), (0.4372, 0.5140),), rank: "DBACE"),
  (pts: ((0.4450, 0.5150), (0.4459, 0.5215), (0.4372, 0.5140),), rank: "DBAEC"),
  (pts: ((0.4459, 0.5215), (0.4353, 0.5233), (0.4372, 0.5140),), rank: "DBEAC"),
  (pts: ((0.4459, 0.5215), (0.4667, 0.6667), (0.4167, 0.6167), (0.4353, 0.5233),), rank: "DEBAC"),
  (pts: ((0.4667, 0.6667), (0.5143, 1.0000), (0.3400, 1.0000), (0.4167, 0.6167),), rank: "EDBAC"),
  (pts: ((0.0250, 0.0750), (0.0000, 0.0583), (0.0000, 0.0500),), rank: "BCADE"),
  (pts: ((0.0000, 0.0000), (0.3714, 0.0000), (0.4197, 0.3382), (0.0250, 0.0750), (0.0000, 0.0500),), rank: "CBADE"),
  (pts: ((0.3606, 0.4483), (0.0000, 0.1393), (0.0000, 0.0583), (0.0250, 0.0750), (0.3821, 0.4321),), rank: "BCDAE"),
  (pts: ((0.3718, 0.4580), (0.3606, 0.4483), (0.3821, 0.4321),), rank: "BDCAE"),
  (pts: ((0.4417, 0.4917), (0.4372, 0.5140), (0.3718, 0.4580), (0.3821, 0.4321),), rank: "DBCAE"),
  (pts: ((0.4197, 0.3382), (0.3821, 0.4321), (0.0250, 0.0750),), rank: "CBDAE"),
  (pts: ((0.4197, 0.3382), (0.4282, 0.3976), (0.3821, 0.4321),), rank: "CDBAE"),
  (pts: ((0.4282, 0.3976), (0.4417, 0.4917), (0.3821, 0.4321),), rank: "DCBAE"),
  (pts: ((0.0000, 0.2000), (0.0000, 0.1393), (0.3606, 0.4483), (0.2964, 0.4964),), rank: "BCDEA"),
  (pts: ((0.0000, 0.4594), (0.0000, 0.2000), (0.2964, 0.4964),), rank: "BCEDA"),
  (pts: ((0.0000, 0.5958), (0.0000, 0.4594), (0.2964, 0.4964), (0.2107, 0.5607),), rank: "BECDA"),
  (pts: ((0.3606, 0.4483), (0.3718, 0.4580), (0.3536, 0.5036), (0.2964, 0.4964),), rank: "BDCEA"),
  (pts: ((0.3536, 0.5036), (0.3393, 0.5393), (0.2964, 0.4964),), rank: "BDECA"),
  (pts: ((0.3393, 0.5393), (0.2107, 0.5607), (0.2964, 0.4964),), rank: "BEDCA"),
  (pts: ((0.0000, 0.7188), (0.0000, 0.5958), (0.2107, 0.5607),), rank: "EBCDA"),
  (pts: ((0.1550, 1.0000), (0.0000, 1.0000), (0.0000, 0.7188), (0.2107, 0.5607), (0.3393, 0.5393),), rank: "EBDCA"),
  (pts: ((0.4372, 0.5140), (0.3536, 0.5036), (0.3718, 0.4580),), rank: "DBCEA"),
  (pts: ((0.4372, 0.5140), (0.4353, 0.5233), (0.3393, 0.5393), (0.3536, 0.5036),), rank: "DBECA"),
  (pts: ((0.4353, 0.5233), (0.4167, 0.6167), (0.3393, 0.5393),), rank: "DEBCA"),
  (pts: ((0.4167, 0.6167), (0.3400, 1.0000), (0.1550, 1.0000), (0.3393, 0.5393),), rank: "EDBCA"),
)

#let palette = (rgb("#4e79a7"), rgb("#f28e2b"), rgb("#59a14f"), rgb("#e15759"), rgb("#b07aa1"))
#let S = 9cm
#let pt(q) = (q.at(0) * S, (1 - q.at(1)) * S)

#figure(
  box(width: S, height: S, stroke: 0.6pt, {
    for cell in cells {
      let first = cell.rank.codepoints().at(0)
      let idx = "ABCDE".position(first)
      place(top + left, polygon(
        fill: palette.at(idx).lighten(55%),
        stroke: 0.4pt + black,
        ..cell.pts.map(pt),
      ))
    }
    for (name, c) in candidates {
      let (x, y) = pt(c)
      place(top + left, dx: x - 3pt, dy: y - 3pt, circle(radius: 3pt, fill: black))
      place(top + left, dx: x + 3pt, dy: y - 12pt, text(size: 9pt, weight: "bold", name))
    }
  }),
  caption: [The #cells.len() ranking cells of the default candidates A–E (rows of
    `CANDIDATES`). Colour marks the first choice; every cell has its own full ranking.],
) <fig-cells>

== Cell probability as a boundary integral <sec-green>

Computing the double integral @eq-cell-prob directly for every cell and every pixel
would be expensive. Green's theorem turns it into a sum of integrals along the cell's
edges, and each edge is shared by two cells.

=== Green's theorem

For a region $K$ with CCW boundary $partial K$ and a 1-form $omega = L dif x + M dif y$,

$ integral.cont_(partial K) L dif x + M dif y = integral.double_K (partial_x M - partial_y L) dif x dif y. $

Choose

$ omega = -f(x) G(y) dif x . $ <eq-omega>

Then $partial_x M - partial_y L = f(x) g(y)$, which is the joint density, so

$ P(K) = integral.cont_(partial K) omega = sum_("edges" s -> e) integral_s^e omega. $

An equally valid choice is

$ omega' = F(x) g(y) dif y, quad omega' - omega = F g dif y + f G dif x = dif(F G). $ <eq-omega-prime>

Because $omega'$ and $omega$ differ by an exact form, on any segment

$ integral_s^e omega = integral_s^e omega' - [F(x) G(y)]_s^e. $ <eq-switch>

Which form is integrated numerically is chosen edge by edge (@sec-edges).

*Sanity check.* The probabilities of all cells sum to one: interior edges are traversed
once in each direction and cancel, leaving the boundary of the square. On it
$dif x = 0$ on the vertical sides, $G(0) = 0$ on the bottom, and the top ($y = 1$,
traversed from $x = 1$ to $x = 0$) gives $-integral_1^0 f(x) dif x = 1$.

=== Sharing edges (`compute_ranking_probabilities`)

Every edge is stored once under a canonical orientation (lexicographically smaller
endpoint first). A cell that traverses the edge in the canonical direction adds its
integral with sign $+1$; the neighbouring cell traverses it the other way and adds it with
sign $-1$. The edges are integrated in parallel threads (`betainc` / `betaincinv`
release the GIL).

The result of each edge integral is a matrix: entry $[k, l]$ uses the $X$-parameters
of pixel column $k$ and the $Y$-parameters of pixel row $l$. Because $X$ and $Y$ are
independent, one set of $n$ parameter pairs serves both axes, and the whole
$n times n$ grid of pixels comes out of $O(n)$ one-dimensional quantities combined in
an outer-product fashion.

== Edge integrals (`_edge_integral`) <sec-edges>

Let the edge go from $s = (x_s, y_s)$ to $e = (x_e, y_e)$.

=== Axis-parallel edges (closed form)

- *Vertical* ($x_s = x_e$): $dif x = 0$, so $integral omega = 0$.
- *Horizontal* ($y_s = y_e$): $G(y)$ is constant, so
  $ integral_s^e omega = -G(y_s) (F(x_e) - F(x_s)). $

These include all edges on the boundary of the square.

=== General edges: substitution $u = F(x)$

On the line through $s$ and $e$, $y = ell(x) = y_s + (x - x_s)(y_e - y_s) slash (x_e - x_s)$.
Integrating @eq-omega directly is a bad idea: $f(x)$ is infinite at $0$ or $1$ when
$a < 1$ or $b < 1$ (@sec-centre), and quadrature would converge slowly. Substituting
$u = F(x)$, $dif u = f(x) dif x$ absorbs the density:

$ integral_s^e omega = -integral_(F(x_s))^(F(x_e)) G(ell(F^(-1)(u))) dif u . $ <eq-u>

The integrand is now bounded by $1$ and smooth unless the edge reaches $y = 0$ or
$y = 1$, where $G$ itself behaves like $y^a$ or $1 - (1-y)^b$ and is not smooth.

=== Edges touching $y = 0$ or $y = 1$: substitution $v = G(y)$

For those edges the roles of the axes are swapped. Integrate $omega'$ from
@eq-omega-prime with $v = G(y)$, $dif v = g(y) dif y$, and $x = ell^(-1)(y)$:

$ integral_s^e omega' = integral_(G(y_s))^(G(y_e)) F(ell^(-1)(G^(-1)(v))) dif v , $

and convert back to $omega$ with @eq-switch. Now the endpoint on $y in {0, 1}$ is harmless
because it is mapped to $v in {0, 1}$ and the singular behaviour of $G$ is absorbed by
the substitution, while $x$ stays in the interior of $(0, 1)$ so $F$ is smooth there.

=== Edges touching both kinds of boundary

An edge can run from a vertical side of the square ($x in {0,1}$) to a horizontal one
($y in {0,1}$). Neither substitution handles both endpoints, so the edge is split at its
midpoint; each half touches only one kind of boundary and is handled as above.
(An edge ending exactly at a corner of the square is integrated with the $v$ form.)

=== Quadrature

The remaining one-dimensional integrals over $[u_s, u_e]$ (or $[v_s, v_e]$) are computed
with $Q$-point Gauss–Legendre quadrature (`QUAD_NODES`, default $Q = 24$):

$
integral_(u_s)^(u_e) h(u) dif u approx (u_e - u_s)/2 sum_(q=1)^Q w_q h(u_s + (u_e - u_s) (t_q + 1)/2),
$

with Legendre nodes $t_q$ and weights $w_q$ on $[-1, 1]$. Note that the interval
$[F_k (x_s), F_k (x_e)]$ depends on the pixel column $k$, so the nodes are placed per
pixel.

*Cost.* For the $u$ form, $F_k^(-1)$ is evaluated at $Q$ nodes for each of the $n$
column-parameters ($Q n$ calls of `betaincinv`), then $G_l$ is evaluated at each resulting
$y$ for each of the $n$ row-parameters ($Q n^2$ calls of `betainc`). An edge therefore
costs $O(Q n^2)$ Beta CDF evaluations, and the total cost is
$O(E dot Q dot n^2)$ for $E$ distinct edges.

== Interpolation in the pixel median <sec-interpolation>

With $n$ large, $O(Q n^2)$ per edge is still costly. But the cell probabilities are
*smooth* functions of the medians $(m_x, m_y)$: $(a, b)$ depends smoothly on $m$, and
the integrals @eq-cell-prob depend smoothly on $(a, b)$. (The *winner* of a voting
method jumps between pixels, but the probabilities do not.) So they are computed
exactly only on an $N times N$ grid of medians (`NODES`, default $N = 49$) and
interpolated to all $n times n$ pixels. If $N >= n$ no interpolation is used.

=== Nodes (`node_medians`)

The interpolation variable is $z = logit(m) = log(m slash (1 - m))$. It stretches the
regions near $0$ and $1$, where $(a, b)$ change fastest. The nodes are the
Chebyshev–Lobatto points in $z$ spanning the first and last pixel median:

$
t_j = -cos(pi j / (N - 1)), quad
z_j = Z t_j, quad
m_j = expit(z_j), quad
j = 0, dots, N - 1,
$

where $Z = logit(1 - 1/(2n))$. (The code writes $t_j$ as
$sin(pi (2j - (N-1)) / (2(N-1)))$, which is the same thing.) Chebyshev–Lobatto points
cluster at the ends of the interval, which further concentrates nodes near the walls
and avoids the Runge phenomenon.

=== Barycentric interpolation (`_interpolation_matrix`)

For Chebyshev–Lobatto nodes the barycentric formula of the second kind is

$
p(z) = (sum_j (lambda_j)/(z - z_j) h_j) / (sum_j (lambda_j)/(z - z_j)),
quad lambda_j = (-1)^j, quad lambda_0, lambda_(N-1) "halved",
$

where $h_j$ are the values at the nodes. (The weights are invariant under the affine map
$t -> Z t$, up to a common factor that cancels.) When a target coincides with a node,
the node value is used directly. Evaluated at all $n$ pixel medians this is a fixed
$n times N$ matrix $L$, and on the 2-D tensor grid

$ P_("pixels")[dot, dot, r] = L thin P_("nodes")[dot, dot, r] thin L^T , $

applied as two matrix products (`interpolate_to_pixels`). The final product is carried
out in `float32` because the result is by far the largest array; the rounding
($approx 10^(-7)$) can only affect pixels whose winning margin is equally small.
Values are finally clipped to $[0, 1]$.

Because the probabilities are analytic in $z$, the interpolation error decreases
exponentially in $N$. The interpolation reproduces constants exactly, so the
probabilities of each pixel still sum to one up to rounding. Narrow voters change
faster with the median: for Beta voters with $D < 0.1$ the web UI uses $2N - 1 = 97$
nodes, because with $49$ the shares of a pixel at $D = 0.05$ would sum to up to $1.025$.

Only the node probabilities are cached on disk (`pixels/cache/beta/<spread>/`,
`pixels/cache/normal/`), together with the medians, the parameters and the settings they were
made with; interpolation happens on every load.

== Normal voters (`normal.py`) <sec-normal>

`normal.py` and `pixels/normal.py` compute the same shares for normal voters
(@sec-normal-model), with the same interface as for Beta voters.

=== Cells in a larger box (`normal_cells`)

Voters can lie outside the square, so the bisector arrangement is built in the box
$[-L, 1 + L]^2$ with $L = 10 sigma$ (`BOX`). For every pixel centre in $[0,1]^2$ the
mass outside the box is at most $4 Phi(-10) approx 3 dot 10^(-23)$. The similarity
map $p |-> (p + L) slash (1 + 2L)$ maps perpendicular bisectors to perpendicular
bisectors and preserves the order of distances, so `ranking_cells` is applied to the
mapped candidates and the polygons are mapped back.

=== Exact cell probabilities with Owen's T function

No quadrature is needed. For a convex CCW polygon $K$ with vertices $p_1, dots, p_r$
and any point $m$, the indicator of $K$ is, almost everywhere, the signed sum of the
indicators of the triangles $(m, p_k, p_(k+1))$ (a fan from $m$; triangles oriented
clockwise count negatively). Hence

$ P(K) = sum_("edges" s -> e) P_plus.minus (m, s, e). $

Each triangle $(m, s, e)$ is split at the foot $q$ of the perpendicular from $m$ onto
the edge line into two right triangles with legs $rho$ (from $m$ to $q$) and $t$ (from
$q$ along the line). The isotropic normal is invariant under rotations, so in units of
$sigma$ a right triangle is ${0 <= z_1 <= h, 0 <= z_2 <= a z_1}$ with
$h = rho slash sigma$, $a = t slash rho$, for standard normal $(Z_1, Z_2)$. The wedge
${z_1 > 0, 0 < z_2 < a z_1}$ holds $arctan(a) slash (2 pi)$ of the mass, and the part of
it beyond $z_1 = h$ is by definition Owen's T function,

$
T(h, a) = 1/(2 pi) integral_0^a exp(-h^2 (1 + x^2) slash 2) / (1 + x^2) dif x =
P(Z_1 > h, 0 < Z_2 < a Z_1) quad (h, a >= 0).
$

Therefore the right triangle has probability

$ R(rho, t) = arctan(t slash rho) / (2 pi) - T(rho slash sigma, t slash rho). $ <eq-owen>

Both terms are odd in $t$; $T$ is even in $h$ and odd in $a$, so @eq-owen with a
*signed* $rho$ is the signed probability. For the edge $s -> e$ with unit direction $u$
and left normal $u^perp = (-u_y, u_x)$ let

$
rho = (m - s) dot u^perp, quad t_s = (s - m) dot u, quad t_e = t_s + |e - s|,
$

where $rho > 0$ when $m$ lies left of the edge, i.e. inside a CCW cell. Then

$ P_plus.minus (m, s, e) = R(rho, t_e) - R(rho, t_s), $

and $0$ when $rho = 0$ (the triangle degenerates). For the $N times N$ grid of pixel
centres $rho$, $t_s$, $t_e$ are outer sums, so each edge costs two evaluations of Owen's
T per pixel. They are compiled (@sec-compiled): for $0 <= a <= 1$ the integral above with
12-point Gauss–Legendre, which agrees with scipy's `owens_t` to $10^(-16)$ for every $h$
(the integrand is analytic in $x$), and for $a > 1$ the identity

$ T(h, a) + T(a h, 1 slash a) = 1/2 (Phi(h) + Phi(a h)) - Phi(h) Phi(a h) quad (h, a >= 0), $

which leaves $T(a h, 1 slash a)$ with $1 slash a < 1$. Five candidates take about
$0.02$ s on $49 times 49$ nodes.

=== Nodes

Unlike the Beta parameters, nothing changes faster near $0$ or $1$, so the nodes are
Chebyshev–Lobatto points in the median itself,
$m_j = 1/2 + (1/2 - 1/(2n)) t_j$, and the barycentric interpolation of
@sec-interpolation is applied in $m$ instead of $logit(m)$
(`interpolate_to_pixels(..., transform=...)`). The logit nodes would be far too
sparse in the middle for small $sigma$ (maximum error at $120$ pixels, $N = 49$):

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: none,
  table.hline(),
  [nodes in], [$sigma = 0.376$], [$sigma = 0.125$], [$sigma = 0.063$],
  table.hline(stroke: 0.5pt),
  [$m$], [$1.7 dot 10^(-7)$], [$3.6 dot 10^(-7)$], [$2.9 dot 10^(-7)$],
  [$logit(m)$], [$1.6 dot 10^(-7)$], [$7.5 dot 10^(-5)$], [$1.4 dot 10^(-2)$],
  table.hline(),
))

The floor of about $10^(-7)$ is the `float32` rounding of the result.

== Accuracy and tests <sec-validation>

The tests (`tests/`) check the probabilities against quantities that are known
independently:

- the cells partition the square (or the box) and every point inside a cell has the
  cell's ranking;
- the shares of every pixel sum to one (to $10^(-12)$ before interpolation);
- for a bisector parallel to an axis the share preferring one candidate is a Beta CDF,
  and at median $1/2$ it is exactly one half (to $10^(-8)$);
- for normal voters the pairwise shares match their closed form (@eq-normal-pairwise)
  (to $10^(-12)$), and every Condorcet method draws the Voronoi diagram;
- the shares match $10^6$ sampled voters within five standard errors
  (`monte_carlo_check` does the same for any pixels, with $2 dot 10^6$ voters, whose
  standard error is at most $1 slash (2 sqrt(M)) approx 3.5 dot 10^(-4)$);
- interpolated shares match the exact ones at every pixel (to $2 dot 10^(-5)$ for Beta
  voters with 33 nodes, $2 dot 10^(-6)$ for normal voters with 49 nodes).

#pagebreak()

= Voting methods (`pixels/methods.py`) <ch-methods>

Every method receives the rankings (an $R times C$ array, best first, one row per
cell) and the shares $P(r)$ of every pixel (an $n times n times R$ array), and returns
the winner of every pixel. Because an electorate is a distribution, a method works with
shares instead of vote counts; every step is a matrix product over the $R$ rankings,
done for all pixels at once.

== First past the post (`fptp`)

The first-choice share of candidate $i$ is

$ s_i = sum_(r: r_1 = i) P(r), $ <eq-first-choice>

and the candidate with the largest share wins.

== Instant runoff (`irv`)

With a set $S$ of eliminated candidates, every ballot counts for its highest ranked
candidate outside $S$,

$ s_i^S = sum_(r: "first of" r "outside" S "is" i) P(r) . $

The candidate with the smallest share is added to $S$, and after $C - 1$ rounds the
remaining candidate wins. The rankings are complete, so no ballot is ever exhausted and
a candidate with a majority is never eliminated: running to the end gives the same
winner as stopping at a majority. The rounds run in a compiled loop over the pixels
(@sec-compiled): every ranking points to its highest ranked candidate outside $S$, and
eliminating a candidate moves only the rankings that point to it. All $C - 1$ rounds
together take at most $R C$ steps per pixel.

== Borda count (`borda`)

With $"pos"_r (i) in {0, dots, C - 1}$ the position of $i$ in ranking $r$,

$ "score"_i = sum_r P(r) (C - 1 - "pos"_r (i)), $

and the highest score wins.

== Pairwise majorities

Condorcet methods only use the pairwise shares

$ pi_(i j) = sum_(r: i "above" j "in" r) P(r), quad pi_(i j) + pi_(j i) = 1, $ <eq-pairwise>

the share of voters preferring $c_i$ to $c_j$; $c_i$ beats $c_j$ when
$pi_(i j) > 1/2$.

*Condorcet winner* (`condorcet`). The candidate who beats every other candidate.
If there is none the pixel is marked with `CYCLE` (black in the figures). As a
function of the pixel, $pi_(i j) = 1/2$ holds only on a curve, so apart from pixel
centres that happen to lie exactly on such a curve (@sec-ties) the majority relation
is a tournament. A tournament in which nobody beats everyone always contains a cycle
(a tournament without cycles is a strict order and has a top), so a black pixel is a
Condorcet cycle unless it is such a tie.

*Schulze* (`schulze`). The defeat strength of $c_i$ over $c_j$ is $pi_(i j)$ if
$pi_(i j) > pi_(j i)$ and $0$ otherwise. The strength of the strongest path is found
with the Floyd–Warshall recursion
$p_(i j) <- max(p_(i j), min(p_(i k), p_(k j)))$ for $k = 1, dots, C$, and the winner is
a candidate with $p_(i j) >= p_(j i)$ for all $j$. Schulze elects the Condorcet winner
whenever there is one, so its diagram differs from the Condorcet winner diagram only on
the black pixels, where it resolves the cycle.

== Baldwin and Nanson (`baldwin`, `nanson`) <sec-baldwin-nanson>

These two are only in the web UI (@ch-realtime), where they are built from blocks
(`yeelab/build/`) as `Eliminate(Tally(BordaCount()), how="min")` and `how="mean"`.
Both repeat the Borda count among the remaining candidates $S$. There a ballot gives
$c_i$ one point per remaining candidate below it, and the expected number of those is
the sum of the chances of being above each of them:

$ "score"_i^S = sum_r P(r) (|S| - 1 - "pos"_r^S (i)) = sum_(j in S, j != i) pi_(i j), $ <eq-borda-pairwise>

with $"pos"_r^S (i)$ the position of $c_i$ among $S$ in ranking $r$. For $S$ = all
candidates this is the Borda score, so Borda, Baldwin and Nanson need only the pairwise
shares. As
$pi_(i j) + pi_(j i) = 1$, the scores in $S$ add up to the number of pairs, and their
mean is $(|S| - 1) slash 2$.

*Baldwin* drops the candidate with the lowest score (ties: the first) and counts again,
until one is left. *Nanson* drops, in every round, each candidate whose score is at
most the mean; if that is all of them, all scores are equal and the first candidate
wins (a set of zero area). A Condorcet winner beats every other remaining candidate,
so its score is above the mean, and both methods elect it: like Schulze, they differ
from the Condorcet winner only where there is a cycle.

Their margins (@sec-zero-sets) are the smallest gap of any round: for Baldwin the gap
between the two lowest scores, as for IRV, and for Nanson the distance of the score
closest to the mean,

$ mu_"Baldwin" = min_S ("score"_((2))^S - "score"_((1))^S), quad
  mu_"Nanson" = min_S min_(i in S) abs("score"_i^S - (|S| - 1) / 2), $

over the sets $S$ of the rounds, with $"score"_((1))^S <= "score"_((2))^S$ the two
lowest. The winner changes only where the decision of some round flips, a tie of the
two lowest or a score at the mean, and there the gap of that round is $0$.

== Minimax and Black (`minimax`, `black`) <sec-minimax-black>

These two are only in the web UI as well, built from blocks as
`Highest(Weakest(Margins(Pairwise())))` and
`Fallback(Unbeaten(Margins(Pairwise())), Highest(Tally(BordaCount())))`. Both compare
candidates by the margins of their head-to-head results,

$ ell_(i j) = pi_(i j) - pi_(j i) = 2 pi_(i j) - 1 = -ell_(j i), $ <eq-link>

which are positive where $c_i$ beats $c_j$.

*Minimax.* The score of $c_i$ is its weakest link, its narrowest win or, if it loses
somewhere, minus its worst defeat,

$ "weak"_i = min_(j != i) ell_(i j), $

and the highest score wins (ties: the first): the winner is the candidate whose worst
defeat is the smallest. A Condorcet winner has a positive score and every other
candidate a negative one, as it loses to the Condorcet winner, so minimax elects the
Condorcet winner whenever there is one. In a cycle it elects the candidate with the
smallest worst defeat, which is the candidate that the Condorcet winner method finds
closest to being one (@sec-zero-sets), so minimax extends it to cycles. Unlike Schulze it
looks at a single defeat and not at paths of defeats.

*Black.* The Condorcet winner if there is one, otherwise the Borda winner. Every pixel
has a winner, never a cycle, and like Schulze, Baldwin and Nanson, Black differs from
the Condorcet winner diagram only where there is a cycle.

== Score voting (`score`) <sec-score>

Score voting is only in the web UI as well, built from blocks as
`Highest(Tally(Score(6, power=1.5)))`. A score ballot is not a ranking: the voter gives
every candidate a score from $0$ to $L - 1$, and the candidate with the highest mean score
wins. The scores depend on how far the candidates are and not only on their order, so the
model needs one more assumption than the ranked methods (`score.py`). The UI has one
score method and two sliders for it: _Score categories_ sets the number of scores $L$,
from $2$ to $11$, by default $6$, and _Score power p_ sets the power $p$ of @eq-range,
from $1$ to $2$ in steps of $0.05$, by default $1.5$ (`SCORE_POWER` in
`build/methods.py`). With $L = 2$ a score ballot is an approval ballot, and approval
voting is score voting with two levels.

*The ballot.* A voter at $p$ puts the candidates in order of distance,

$ r_((1)) <= r_((2)) <= dots <= r_((C)), quad r_i = |p - c_i|, $ <eq-distances>

and gives the closest the top score $L - 1$ and the farthest $0$. Every other candidate
gets a score by its _part of the way_ from the farthest to the closest,
$t_i = (r_((C)) - r_i) slash (r_((C)) - r_((1)))$, raised to the power $p$:

$ "score"_i = "round"(t_i^p (L - 1)), quad t_i = (r_((C)) - r_i) / (r_((C)) - r_((1))), $ <eq-range>

with a half rounded to the even score. A voter as far from every candidate gives them all
the top score. $p = 1$ (`POWER` in `score.py`, the default of the block) is in
proportion to the distance; above $1$ the top scores stay with the candidates near the
closest. Only the closest and the farthest candidate set the scale, whatever $p$.

The ballot has the properties one expects of a sincere score ballot, at every $p$: the
closest candidate gets the top score and the farthest $0$, a closer candidate never gets
a lower score, the ballot does not change when all distances are scaled or shifted, so no
unit of distance has to be chosen, and a copy of a candidate changes no other score. As
the voter moves, the scores change one at a time, each where its own part of the way
crosses a border, and a small move changes a score by one level at most. With more levels
$"score"_i slash (L - 1)$ tends to $t_i^p$, not to $t_i$: the power is not a rounding
effect, it bends the scale itself. `script/score_edge_cases.py` checks these properties
on edge cases and random voters.

*What $p$ does.* A score changes where its part of the way is half a score,

$ t_i = ((k + 1/2) / (L - 1))^(1 slash p), quad k = 0, dots, L - 2 . $ <eq-range-borders>

#figure(
  align(center, table(
    columns: 4,
    align: (left, right, left, left),
    stroke: none,
    table.hline(),
    [$p$], [$L = 2$: approves beyond], [$L = 3$], [$L = 6$],
    table.hline(stroke: 0.5pt),
    [$1$], [$0.50$], [$0.25$, $0.75$], [$0.10$, $0.30$, $0.50$, $0.70$, $0.90$],
    [$1.5$], [$0.63$], [$0.40$, $0.83$], [$0.22$, $0.45$, $0.63$, $0.79$, $0.93$],
    [$2$], [$0.71$], [$0.50$, $0.87$], [$0.32$, $0.55$, $0.71$, $0.84$, $0.95$],
    table.hline(),
  )),
  caption: [The borders of @eq-range-borders: the parts of the way where a score changes,
    for the ends and the default of the slider.],
) <tab-range-borders>

With two levels the voter approves the candidates beyond $2^(-1 slash p)$ of the way:
halfway for $p = 1$, $0.63$ for $p = 1.5$ and $0.71$ for $p = 2$ (@tab-range-borders).
The farthest candidate sets the scale, and it is usually far from every candidate the
voter considers, so with $p = 1$ halfway is a generous cut: real voters give the top
scores much more sparingly, and $p = 1.5$ is the one power that fits their ballots best
(@sec-score-data). In the plane the generous cut favours a candidate in the middle of the
others: the voters on every side have it closer than halfway, so it is approved by many
whose nearest candidate is someone else (@fig-approval).

The slider stops at $1$ and at $2$. No ballot of @sec-score-data is fitted better by a
power below $1$, and above $2$ the ballots with a neutral middle score fit much worse
(@sec-score-data): with three levels $p = 2$ puts the border of the middle score exactly
at halfway, where a voter who gives a candidate the middle opinion of a scale puts it, and
the half is rounded down.

*Curved borders.* A score changes where @eq-range-borders holds,
$r_((C)) - r_i = ((k + 1/2) slash (L - 1))^(1 slash p) dot (r_((C)) - r_((1)))$: a
curve in the plane, as the distances are not squared. (The bisector of @eq-bisector is
straight because there the squares $|p|^2$ cancel.) The voters who give a candidate a
score are therefore not a union of polygons (@fig-approval, top), and the mean score is
not a sum of the edge terms of @sec-green.

*Distances, not their squares.* Take a voter who stands on a candidate, with the others
at the squared distances $0.35$, $0.45$, $0.55$ and $1$. The distances are $0$, $0.59$,
$0.67$, $0.74$ and $1$: the candidate at the voter is far ahead of the rest. On the
squares the three in the middle move a part of the way of $0.1$ to $0.3$ closer to the top
score. Real ratings agree with the distances: their squares fit them worst
(@sec-score-data).

*Shares from a grid.* Let $q_i$ be the mean score of $c_i$ as a part of the top score,
and $u_i = 1 - q_i$ the part not given; with two levels $u_i$ is the share of the voters
who do not approve $c_i$. Like a ranking, a ballot depends only on where the voter is,
so the points $b_i (x, y)$ that a voter at $(x, y)$ gives below the top score are the
same for every pixel, and $u_i = E[b_i (X, Y)] slash (L - 1)$. The plane is cut into
rectangles by the lines $x_0 < x_1 < dots < x_K$, the same along both axes. $X$ and $Y$
are independent, so the rectangle $[x_k, x_(k + 1)] times [x_l, x_(l + 1)]$ holds the
share $mu_k nu_l$ of the voters, with $mu_k = F(x_(k + 1)) - F(x_k)$ from the exact CDF
$F$ of $X$, and $nu_l$ likewise from that of $Y$. With $a_(i k l)$ the mean of
$b_i slash (L - 1)$ over that rectangle,

$ u_i approx sum_(k, l) mu_k thin a_(i k l) thin nu_l . $ <eq-grid-share>

For all medians at once these are two matrix products per candidate (`unscored_shares`
in `margin/shares.py`). `unscored` in `score.py` finds $a$: a rectangle whose four
corners have the same ballot counts as all of that ballot, and in the others, about 2% of
them, the ballots at $8 times 8$ points are averaged.

The grid has $256$ equal cells across the square. For Beta voters the cell at each wall
is halved again and again down to $10^(-6)$: the density of a Beta with $a < 1$ is
infinite at the wall, and the voters of a pixel next to a wall sit within a small part of
an equal cell. For normal voters the cells grow by a factor of $1.1$ each beyond the
square, out to the box of @sec-normal.

Only $a$ is approximate: within a rectangle the borders are replaced by a mean over the
rectangle. Where exact shares are known the grid is within $2 dot 10^(-4)$ of them for
$D >= 0.2$ and within $5 dot 10^(-4)$ for $D = 0.05$
(`tests/test_score.py`): with two candidates every ballot gives the first choice the top
score and the other $0$, so $u_i$ is a share of @eq-first-choice. With two levels the
share who approve a candidate is also between the exact shares of the voters with it
first and with it among the $C - 1$ closest.

*Why the part not given.* Narrow voters far from the candidates all give the same
candidates the top score. With four candidates and $D = 0.05$ the voters of a pixel in a
far corner approve the same two, and both $q_i$ are $1$ up to $10^(-20)$ or less. Double
precision cannot tell $1 - 10^(-20)$ from $1$, so on a large part of the square the two
$q_i$ would be equal, or differ only by rounding. The winner there is the candidate with
the smaller $u_i$, and @eq-grid-share has no negative terms, so a $u_i$ of $10^(-40)$ is
as exact, in relative terms, as one of $0.4$. Two things keep it so.

- $mu_k$ is exact in both tails. Below the median it is $F(x_(k + 1)) - F(x_k)$. Above
  the median $F$ is close to $1$ and that difference would be $0$; there
  $mu_k = S(x_k) - S(x_(k + 1))$ with $S = 1 - F$ computed directly, as $I_(1 - x) (b, a)$
  for Beta voters and $Phi(-z)$ for normal ones.
- The $u_i$ are not interpolated. The interpolant of @sec-interpolation is exact to about
  $10^(-5)$, far more than these shares. They are computed at the points where the
  borders are traced (@sec-zero-sets), with $mu$ and $nu$ of the voters of those points;
  for Beta voters their parameters are solved at each of these medians (@sec-solve). One
  more point is one more row of each matrix product.

With two candidates and normal voters $u_i$ is known: $Phi$ of the distance to the
bisector over $sigma$. The grid follows it within 6% down to $10^(-36)$
(`tests/test_score.py`). At $D = 0.05$ and $G = 320$ such a tail falls by a factor of
about $1.5$ from one traced point to the next, so 6% moves a border by a small part of
that step.

*Winner.* The candidate with the largest $q_i$ wins, that is the smallest $u_i$, and the
margin (@sec-zero-sets) is its lead over the second: the second smallest $u_i$ minus the
smallest. The tally of a score ballot in `yeelab.build` is $-u_i = q_i - 1$, the mean
score counted down from $1$. With two levels the $q_i$ do not sum to $1$: their sum is
the mean number of approved candidates.

#figure(
  image("figures/approval.png", width: 100%),
  caption: [Two levels at the ends and the default of the slider, $p = 1$, $1.5$ and
    $2$. Top: the voters who approve D, those beyond $2^(-1 slash p)$ of the way from the
    farthest candidate to the closest; the borders are curves. Bottom: the diagrams
    (candidates A–E, Beta voters, `rms`, $D = 0.2$).],
) <fig-approval>

*Two levels.* D is in the middle of A–E, and its Voronoi cell is 21% of the square
(@fig-approval). With $p = 1$ D is approved by 64% of the voters of the square and wins
42% of it, more than FPTP (14%), Schulze (25%) or Borda (31%), and A only 8%: a voter
near A has B, C and E far away, which puts D closer than halfway, so D is approved as
well. With $p = 1.5$ the cut moves to $0.63$ of the way, D is approved by 45% and wins
23%, and A 25%; with $p = 2$, 36% and 16%, and A 31%. A voter approves $2.3$ candidates on
average with $p = 1$, $1.9$ with $1.5$ and $1.6$ with $2$. The part of the square not won
by the nearest candidate falls from 24% to 9% and 8%.

#figure(
  image("figures/score.png", width: 100%),
  caption: [Six levels, the default, at $p = 1$, $1.5$ and $2$, next to the Voronoi
    diagram (candidates A–E, Beta voters, `rms`, $D = 0.2$).],
) <fig-score>

*Six levels.* With six levels the power matters less, as more levels give every
candidate more of its part of the way (@fig-score): D wins 29%, 24% and 22% of the square
with $p = 1$, $1.5$ and $2$, and A 24%, 26% and 27%; the part not won by the nearest
candidate is 10%, 5% and 3%. The mean scores of the voters of the square fall with $p$,
D from $3.15$ to $2.62$ and $2.37$ of $5$, as every candidate but the closest gets less.

*Against the Voronoi diagram.* The same holds for other candidates. Over 60 random
layouts of 3 to 7 candidates, every other one with two candidates $0.08$ to $0.2$ apart,
with the voters of the UI's default ($D = 0.2$), the part of the square not won by the
nearest candidate is (`docs/figures.py`):

#figure(
  align(center, table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    [], [2 levels], [lost], [6 levels], [lost],
    table.hline(stroke: 0.5pt),
    [`Score`, $p = 1$], [34.4%], [1.43], [14.2%], [0.12],
    [`Score`, $p = 1.5$], [23.6%], [0.92], [9.0%], [0.05],
    [`Score`, $p = 2$], [18.4%], [0.65], [7.6%], [0.07],
    table.hline(stroke: 0.5pt),
    [`ScoreAvg`], [37.7%], [1.38], [14.1%], [0.10],
    [`ScoreDH`], [29.5%], [1.45], [15.6%], [0.28],
    [`ScoreHybrid`], [15.1%], [0.68], [12.2%], [0.17],
    [`ScoreCluster`], [34.5%], [1.48], [13.7%], [0.15],
    table.hline(),
  )),
  caption: [The part of the square not won by the nearest candidate (at the median of
    the voters), and _lost_: the candidates per diagram who win less than a quarter of
    their Voronoi cell, over 60 random layouts (Beta voters, `rms`, $D = 0.2$).],
) <tab-score-voronoi>

A higher $p$ is closer to the Voronoi diagram, but only up to a point: the Voronoi
diagram is what the voters' own nearest candidate would draw, and a ballot that gives the
top score to the closest candidate alone would draw it best at small $D$. That is not how
voters score (@sec-score-data), so the Voronoi diagram only says in which direction to
bend; the data say how far.

*Other ballots.* `score.py` has four other ways from distances to scores. The web UI
does not offer them, but `script/score_edge_cases.py`, `docs/ballots.py` and the
comparisons above use them, as `ScoreAvg`, `ScoreDH`, `ScoreHybrid` and `ScoreCluster`
in `yeelab.build`.

- `ScoreAvg` puts the mean distance $overline(r) = 1/C sum_j r_j$ in the middle of the
  scale, so with two levels it approves the candidates closer than $overline(r)$: the
  best approval ballot of a voter with the utility $-r_i$ who takes every pair of
  candidates to be as likely to tie (Weber). Every candidate moves the scale through
  $overline(r)$, so a copy of a candidate changes the other scores, and the far
  candidates of an election raise the mean and with it the cut: of all these ballots it
  fits real ones worst.
- `ScoreDH` shares the $L - 1$ steps from the top score to $0$ out among the gaps between
  neighbours in the order of distance, each in turn to the gap with the largest
  gap$slash$(its steps $+ delta$), $delta = 0.8$ (D'Hondt at $1$); with two levels it
  approves the candidates above the largest gap. Candidates close together share a score,
  so two near rivals are approved alike and one of them can lose its region altogether,
  and a small move of the voter can change several scores at once.
- `ScoreHybrid` does what `ScoreDH` does for the candidates closer than halfway, on the
  upper half of the scale, and what $p = 1$ does for the others, on the lower half. Its
  cut is far from the farthest candidate, as on real approval ballots (it is the best
  rule with two levels in @tab-score-voronoi), but it inherits the jumps of `ScoreDH`,
  and of the 15 kinds of real ballot it fits only the Italian approval ballots better
  than $p = 1.5$.
- `ScoreCluster` gives the scores of $p = 1$ unless they split a cluster of candidates
  much closer to each other than to those around them, at a cost $mu$ per split and with
  $kappa$ setting how far apart a cluster must stand. It has two parameters more and
  fits real ballots no better than $p = 1$.

`script/score_edge_cases.py` checks ten properties of a ballot on edge cases (small
moves, clusters swept across the scale, clones, many levels, scaling, the order of the
scores). `Score` meets all ten at every power of the slider, `ScoreAvg` and
`ScoreCluster` nine, `ScoreHybrid` eight and `ScoreDH` six. On real ballots `Score` with $p = 1.5$ misses 19.6% fewer
points than with $p = 1$, `ScoreHybrid` 10.8% fewer, `ScoreCluster` 0.2% more,
`ScoreDH` 7.5% more and `ScoreAvg` 22.9% more (@sec-score-data).

== Score ballots on real ballots <sec-score-data>

@eq-range is an assumption about how a voter turns distances into scores. It can be
checked where the same people gave both something a distance follows from and a coarser
ballot or rating of the same candidates. Seven public data sets do (@tab-ballot-data);
`docs/ballots.py` downloads them and prints every number of this section
(`uv run --with pandas --with rdata --with pyreadstat python docs/ballots.py`). The
ballot depends on the distances only up to a common scale and shift, so any of these
distances gives the same ballot as the distance in a plane would.

#figure(
  align(center, table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    [data], [people and candidates], [distance], [ballots],
    table.hline(stroke: 0.5pt),
    [#link("https://zenodo.org/records/1199545")[Voter Autrement 2017], online], [37 726
      participants, 11 candidates of the French presidential election],
      [100 minus their opinion, 0–100], [approval; evaluation on 0/1/2, −1/0/1, 0/1/2/3
      and −1/0/1/2],
    [#link("https://zenodo.org/records/10998451")[Voter Autrement 2022]], [2 284
      participants, 12 candidates], [100 minus their opinion, 0–100], [approval; scores on
      0/1/2, −1/0/1, 0/1/2/3, −1/0/1/2; majority judgment on 5 and 7 grades],
    [#link("https://data.mendeley.com/datasets/dgsd5yb7zp/1")[Votare Altrimenti 2022]],
      [1 021 respondents, 14 Italian parties], [100 minus their opinion, 0–100],
      [approval; scores 0–4; evaluative 0–4],
    table.hline(stroke: 0.5pt),
    [`perfume_ideal`, `cream_id`, #link("https://cran.r-project.org/package=SensoMineR")[SensoMineR]],
      [103 and 86 consumers, 14 perfumes and 9 creams], [perceived to ideal profile],
      [liking 1–9 and 0–10],
    [#link("https://cses.org")[CSES] Integrated Module], [235 030 respondents, 3 to 9
      parties each], [left–right, as the respondent placed self and party],
      [like–dislike 0–10],
    [#link("https://osf.io/gvqjs")[Kuhlmann et al. 2017]], [879 people, 26 personality
      items], [top of a slider 1–101 minus the answer], [Likert 1–5],
    [#link("https://osf.io/5uwcp/?view_only=ad027d57b2ea45ab976b9882dea98f93")[Zhang et al.], CES-D 8],
      [394 people, 8 items], [top of either of two sliders 0–100 minus the answer],
      [Likert 0–4; scale 0–14],
    table.hline(),
  )),
  caption: [Data sets that pair a distance with a ballot or rating of the same candidates
    by the same people: three voting experiments (15 kinds of ballot), and ratings given
    without a stake (8 kinds).],
) <tab-ballot-data>

In the voting experiments every participant cast a few ballots of kinds drawn at random
and gave each candidate an opinion on a slider from 0 to 100. A ballot of @eq-range
uses its lowest and its highest score, so only the voters whose real ballot does too
count: 10 108 of the 10 479 Voter Autrement 2017 participants with an approval ballot
and every opinion, for instance. A rating on more
than four levels is first stretched from the person's lowest to their highest rating, as
few people use the whole of a long scale when nothing is at stake. The measure is the
mean distance between the ballot of @eq-range and the real one, in points of the scale,
per voter and then over the voters.

*The power.* @tab-power gives, for every kind of ballot, the points missed at the ends
and the default of the slider and the best power.

#figure(
  align(center, table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    stroke: none,
    table.hline(),
    [ballots], [$L$], [voters], [$p = 1$], [$1.5$], [$2$], [best $p$],
    table.hline(stroke: 0.5pt),
    [VA 2017 approval], [2], [10 108], [0.150], [0.092], [0.083], [1.90],
    [VA 2017 0/1/2], [3], [2 636], [0.223], [0.144], [0.148], [1.75],
    [VA 2017 −1/0/1], [3], [6 074], [0.196], [0.168], [0.211], [1.35],
    [VA 2017 0/1/2/3], [4], [4 399], [0.312], [0.215], [0.221], [1.65],
    [VA 2017 −1/0/1/2], [4], [5 389], [0.284], [0.221], [0.256], [1.40],
    [VA 2022 approval], [2], [1 329], [0.139], [0.099], [0.091], [2.00],
    [VA 2022 0/1/2], [3], [365], [0.186], [0.152], [0.205], [1.50],
    [VA 2022 −1/0/1], [3], [729], [0.180], [0.182], [0.268], [1.20],
    [VA 2022 0/1/2/3], [4], [617], [0.351], [0.246], [0.257], [1.55],
    [VA 2022 −1/0/1/2], [4], [712], [0.309], [0.274], [0.312], [1.25],
    [VA 2022 MJ, 5 grades], [5], [1 140], [0.424], [0.376], [0.405], [1.45],
    [VA 2022 MJ, 7 grades], [7], [1 131], [0.597], [0.543], [0.611], [1.40],
    [Italy 2022 approval], [2], [753], [0.151], [0.103], [0.088], [2.90],
    [Italy 2022 scores 0–4], [5], [708], [0.487], [0.442], [0.462], [1.50],
    [Italy 2022 evaluative 0–4], [5], [635], [0.537], [0.513], [0.543], [1.35],
    table.hline(stroke: 0.5pt),
    [perfumes, liking], [9], [103], [1.773], [1.860], [2.019], [0.95],
    [creams, liking], [11], [86], [2.185], [2.344], [2.536], [1.00],
    [CSES, like–dislike], [11], [235 030], [2.902], [2.874], [2.942], [1.35],
    [Kuhlmann, Likert 1–5], [5], [570], [0.416], [0.544], [0.688], [1.00],
    [Zhang, slider c, Likert 0–4], [5], [383], [0.873], [0.904], [0.948], [0.95],
    [Zhang, slider c, scale 0–14], [15], [390], [2.251], [2.419], [2.664], [0.95],
    [Zhang, slider d, Likert 0–4], [5], [383], [0.870], [0.895], [0.942], [1.05],
    [Zhang, slider d, scale 0–14], [15], [390], [2.182], [2.336], [2.580], [1.00],
    table.hline(),
  )),
  caption: [Points missed per candidate by the ballot of @eq-range at $p = 1$, $1.5$ and
    $2$, and the power that misses fewest (searched from $0.8$ to $3$ in steps of
    $0.05$). VA: Voter Autrement; MJ: majority judgment.],
) <tab-power>

Every one of the 15 kinds of election ballot is fitted better by a power above $1$, from
$1.20$ to $2.90$. One power for all of them (@tab-power-summary) misses 19.6% fewer
points than $p = 1$ at $p = 1.5$, and is worse on no kind by more than 0.8%. The optimum is
flat from $1.4$ to $1.7$; beyond $2$ it falls apart, as the scales with a neutral middle
fit much worse.

#figure(
  align(center, table(
    columns: 9,
    align: (left, right, right, right, right, right, right, right, right),
    stroke: none,
    table.hline(),
    [$p$], [$1.2$], [$1.3$], [$1.4$], [$1.5$], [$1.6$], [$1.8$], [$2.0$], [$2.5$],
    table.hline(stroke: 0.5pt),
    [elections: mean], [−14.3%], [−17.1%], [−18.7%], [−19.6%], [−19.5%], [−18.2%], [−10.6%], [+0.4%],
    [elections: worst kind], [−4.0%], [−1.8%], [−0.1%], [+0.8%], [+3.5%], [+8.8%], [+48.8%], [+65.6%],
    [elections: kinds better], [15], [15], [15], [14], [14], [14], [9], [7],
    table.hline(stroke: 0.5pt),
    [ratings: mean], [+1.5%], [+3.1%], [+5.4%], [+7.8%], [+10.0%], [+14.6%], [+18.8%], [+28.1%],
    table.hline(),
  )),
  caption: [The change of the points missed against $p = 1$, the same power for every
    kind of ballot: the mean over the 15 kinds of election ballot and the worst of them,
    the number of kinds it fits better than $p = 1$, and the mean over the 8 kinds of
    rating.],
) <tab-power-summary>

*Where voters cut.* On the scale of @eq-range, from $1$ at the closest candidate to $0$
at the farthest, the farthest candidate a Voter Autrement 2017 voter approves is at
$0.76$ (median) and the closest one the voter does not approve at $0.60$; for 80% of the
voters every approved candidate is closer than every other one. They approve 2.85
candidates. Approving above $0.7$ of the way gets 91.7% of the decisions right, above
halfway 85.0%. With $p = 1$ the ballot approves 4.22 candidates and gets 23.5% of the
ballots wholly right; with $p = 1.5$ it approves 3.22 and gets 39.0%, with $p = 2$ 2.69 and
42.9%.

*By kind of ballot.* Approval wants the most bend: the best single power of the three
approval ballots is $2.0$ (40.2% fewer points missed; 32.9% at $1.5$). The scales from $0$
(0/1/2, 0/1/2/3, majority judgment, the Italian 0–4) want $1.55$, and the scales with a
neutral middle (−1/0/1, −1/0/1/2) the least, $1.35$: their middle score is the voter's
neutral, not a score halfway to the top. On −1/0/1 with $p = 2$ a candidate exactly
halfway gets $0$ (@sec-score): 9.4% of the candidates of Voter Autrement 2022 are there
(an opinion of 50, mostly), and 71% of them get the middle score, so they are missed by
$0.92$ points instead of $0.29$. That is why the slider stops at $2$.

*A power per voter.* Voters differ much more than kinds of ballot do: each voter's own
best power would miss 52.4% fewer points than $p = 1$. That difference is not in their
distances, though. A power chosen per voter from where the other candidates are on the
way (by quintile of their mean part of the way, fitted on half the voters of each kind and
tested on the other half) misses 22.6% fewer points, one power per kind of ballot 22.2%
and $p = 1.5$ for all 20.0%. A voter's power is a matter of temperament, not of where
the candidates are, so the slider has one power, the same for every voter.

*Other utilities.* @eq-range is a utility $t_i^p$ of the part of the way, scored in
proportion. Other utilities with one parameter, scored the same way from $0$ at the
farthest to $1$ at the closest, fit no better (one parameter for all 15 kinds):

#figure(
  align(center, table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    table.hline(),
    [utility], [parameter], [mean], [worst kind],
    table.hline(stroke: 0.5pt),
    [$t_i^p$], [$p = 1.55$], [−19.6%], [+2.3%],
    [$exp(-k (1 - t_i))$], [$k = 1.2$], [−19.9%], [+1.8%],
    [$exp(-(r_i slash (s r_((C))))^2)$], [$s = 0.5$], [−18.0%], [−0.6%],
    [$-log(r_i + c thin r_((C)))$], [$c = 0.5$], [−17.6%], [+0.6%],
    [$-r_i^q$], [$q = 0.6$], [−15.2%], [+4.8%],
    table.hline(),
  )),
  caption: [The change of the points missed against $p = 1$ for other utilities of the
    distance, each with the one parameter that fits the 15 kinds of election ballot best.],
) <tab-utilities>

The two convex utilities of the part of the way, the power and the exponential, are
the best and nearly the same: both put the approval cut at about $0.63$ of the way. Those
of the distance itself, which also depend on how far the voter is from even the closest
candidate, are worse. The power is the simpler, gives the borders of
@eq-range-borders in closed form, and is the ballot of the definition at $p = 1$.

*Ratings.* The ratings given without a stake want no bend (@tab-power): the best power
of the products and of the psychological items is $0.95$ to $1.05$, and $p = 1.5$ fits
them 3% to 31% worse. Only the parties of CSES, rated without a ballot, want $1.35$, and
gain 1%. The bend is a property of voting, of how sparingly people give the top scores of
a ballot, and not of turning a fine scale into a coarse one.

*Distances, not their squares.* @eq-range on the distances to an exponent misses by:

#align(center, table(
  columns: 5,
  align: (left, right, right, right, right),
  stroke: none,
  table.hline(),
  [], [$r^(0.5)$], [$r^(0.75)$], [$r$], [$r^2$],
  table.hline(stroke: 0.5pt),
  [perfumes], [1.820], [1.777], [1.773], [1.994],
  [creams], [2.213], [2.213], [2.185], [2.304],
  [CSES], [2.904], [2.868], [2.902], [3.171],
  table.hline(),
))

The squares are the worst in all three; the distances themselves are best for the
products and within $0.04$ of the best for the parties.

*Limits.* The voting experiments are online and their participants chose to take part:
39% of the Voter Autrement 2017 voters above voted Mélenchon in the first round, 21%
Macron and 16% Hamon. They gave their opinions after voting, so the opinions may have
been fitted to the ballots, and no ballot of the experiments counted. The opinions are
utilities, not distances in a plane: where utility is not linear in distance, the cut
falls elsewhere in the plane. The power was chosen on these data; the Voter Autrement
rounds of 2017 and 2022 and the Italian survey agree on it, but they are not independent
of the choice.

== Ties <sec-ties>

Exact ties are broken by the order of the candidates: `argmax` and `argmin` take the
first candidate. With exact shares a tie only happens on sets of zero area, e.g. for
pixel centres exactly on a bisector under normal voters, where $pi_(i j) = 1/2$ and
neither candidate beats the other (the dotted black lines in @fig-compare).

#pagebreak()

= Beta versus normal voters <ch-compare>

With the same candidates the two voter models give visibly different diagrams, and the
difference is largest where one would expect it least: for Condorcet methods. On
#link("https://votingmethods.net/yee/")[votingmethods.net/yee] (normal voters) the
River method splits the square into one clean region per candidate, while Schulze on
the Beta model has distorted regions and even pixels without a Condorcet winner
(@fig-compare). This chapter explains why.

All numbers below use the candidates A–E (@fig-cells), the default spread rule `rms`,
$300 times 300$ pixels, $49 times 49$ nodes and $D = 0.3$. This is more than the default
$0.2$ so that the effects are easy to see; at $0.2$ they are the same in kind but smaller
(@sec-whole). The numbers and figures are produced by `docs/figures.py`. How the
other spread rule changes them is the subject of @ch-spread.

#figure(
  image("figures/compare.png", width: 100%),
  caption: [Beta voters (`rms`, top) and normal voters (bottom), $D = 0.3$. Normal
    Schulze and the normal Condorcet winner are exactly the Voronoi diagram of the
    candidates. Black: no Condorcet winner. The dotted black lines in the bottom right
    plot are pixels whose centre lies exactly on a bisector (a tie, not a cycle).],
) <fig-compare>

== The two models

#align(center, table(
  columns: 3,
  align: (left, left, left),
  stroke: none,
  table.hline(),
  [], [*Beta* (`ranking_cells.py`)], [*normal* (`normal.py`)],
  table.hline(stroke: 0.5pt),
  [voters of pixel $m$], [$X tilde Beta(a_x, b_x)$, $Y tilde Beta(a_y, b_y)$ independent],
    [$(X, Y) tilde cal(N)(m, sigma^2 I)$],
  [support], [unit square], [whole plane],
  [$m$ is], [the median of each coordinate], [mean and median, centre of symmetry],
  [spread], [$E|X - 1/2| = D$ at the centre pixel, \ RMS from the median fixed (`rms`)],
    [$E|X - m_x| = D$ everywhere, @eq-sigma],
  [symmetric about $m$], [only for $m = (1/2, 1/2)$], [always],
  table.hline(),
))

votingmethods.net uses a standard deviation of $0.4$ by default ($D approx 0.319$ here)
and samples only 140 voters per pixel, so its diagrams also carry Monte Carlo noise
(ragged IRV borders, small islands). The shares here are exact.

== Pairwise majorities

By @eq-bisector a voter at $p$ prefers $c_i$ to $c_j$ exactly when
$p dot nu < omega$ with $nu = c_j - c_i$ and $omega = (|c_j|^2 - |c_i|^2) slash 2$.
The pairwise share (@eq-pairwise) of the voters of pixel $m$ who prefer $c_i$ to $c_j$
is therefore

$ pi_(i j)(m) = P_m (X dot nu < omega), $

and it depends only on the one-dimensional projection $X dot nu$ of the voters onto the
direction from $c_i$ to $c_j$. Every Condorcet method only looks at these numbers.

=== Normal voters: the majority is decided by distance <sec-normal-majority>

The projection of an isotropic normal is normal,
$X dot nu tilde cal(N)(m dot nu, sigma^2 |nu|^2)$, so

$
pi_(i j)(m) = Phi((delta_(i j)(m)) / sigma), quad
delta_(i j)(m) = (omega - m dot nu) / (|nu|),
$ <eq-normal-pairwise>

where $delta_(i j)(m)$ is the signed distance of $m$ from the bisector, positive on the
side of $c_i$. A majority prefers $c_i$ to $c_j$ if and only if $delta_(i j)(m) > 0$, that
is, if and only if $|m - c_i| < |m - c_j|$. Consequently:

+ The majority relation of every pixel is the order of the candidates by distance from
  $m$. It is transitive: *there are no Condorcet cycles.*
+ The Condorcet winner always exists and is the candidate nearest to $m$, so *every*
  Condorcet method (Schulze, River, Ranked Pairs, Minimax, …) draws exactly the Voronoi
  diagram of the candidates.
+ This holds *for every $sigma$*. The only property used is that every line through $m$
  has half of the voters on each side, which is true for every distribution that is
  centrally symmetric about $m$.

This is why the River diagram of votingmethods.net looks like a Voronoi diagram; its
borders agree with the bisectors (e.g. the D|A bisector meets $x = 1$ at $y = 0.725$ and
the D|E bisector meets $y = 1$ at $x = 0.8$). The spread $sigma$ still matters for
methods that use more than pairwise majorities (FPTP, IRV, Borda).

=== Beta voters: the median is a median only along the axes

For the Beta model $m$ is the median of $X$ and of $Y$ separately. If $nu$ is parallel
to an axis, $X dot nu$ is a multiple of one coordinate, its median is $m dot nu$, and the
line through $m$ perpendicular to $nu$ splits the voters exactly in half. For an oblique
$nu$ this fails: the median of $nu_x X + nu_y Y$ is in general *not*
$nu_x m_x + nu_y m_y$, because medians are not additive for skewed variables. For every
direction $nu$ there is a different "effective centre" of the pixel, shifted from $m$
towards the mean of the voters. Consequently:

+ The Condorcet winner is no longer the candidate nearest to $m$, so Condorcet regions
  are distorted.
+ Different pairs of candidates are compared from different effective centres, so the
  majority relation need not be transitive: *Condorcet cycles appear* (black in
  @fig-compare). If some point were a median in every direction (a "total median",
  Davis, DeGroot and Hinich, 1972), the argument of the normal case would apply with
  that point in place of $m$; a product of skewed Beta marginals generally has none.

@ch-geometric draws every pixel at the geometric median of its voters instead of at $m$,
which takes back part of this shift for every direction at once.

=== The shape of Beta voters <sec-skew>

At $D = 0.3 > 1/4$ the Beta is U-shaped even at the centre (@sec-centre). Towards a
wall:

#align(center, table(
  columns: 6,
  align: right,
  stroke: none,
  table.hline(),
  [median], [$a$], [$b$], [mean], [$P(X < 0.1)$], [$P(X > 0.9)$],
  table.hline(stroke: 0.5pt),
  [0.50], [0.602], [0.602], [0.500], [0.175], [0.175],
  [0.70], [0.743], [0.454], [0.621], [0.095], [0.298],
  [0.90], [0.796], [0.265], [0.750], [0.051], [0.500],
  [0.98], [0.648], [0.151], [0.812], [0.049], [0.640],
  table.hline(),
))

As the median approaches the wall, half of the voters crowd into the strip between the
median and the wall, while the other half forms a long tail towards the centre of the
square. The mean lags behind the median by up to $0.17$: the voters of every pixel
off the centre are skewed towards the middle of the square, and so are their effective
centres.

=== Example: the pull towards the centre

The pixels below lie on A's side of the bisector of A $= (0.6, 0.35)$ and the centre
candidate D $= (0.5, 0.5)$; $delta$ is their distance from it. Share of voters preferring
A (Beta: $4 dot 10^6$ sampled voters):

#align(center, table(
  columns: 5,
  align: right,
  stroke: none,
  table.hline(),
  [pixel $m$], [$delta$], [normal], [Beta], [Beta mean],
  table.hline(stroke: 0.5pt),
  [(0.95, 0.55)], [0.118], [0.623], [0.525], [(0.786, 0.530)],
  [(0.90, 0.60)], [0.049], [0.551], [*0.473*], [(0.750, 0.560)],
  [(0.80, 0.45)], [0.118], [0.623], [0.541], [(0.684, 0.470)],
  table.hline(),
))

In all three pixels A gets far less support than under normal voters, and in the
second the median voter is closer to A, yet D wins the pairwise contest: the long tails
of the Beta point towards the middle of the square, and so does the effective centre.

=== The whole diagram <sec-whole>

Leaving out the 357 pixels whose centre lies exactly on a bisector (where both
candidates are correct winners):

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: none,
  table.hline(),
  [], [Schulze $!=$ Voronoi], [no Condorcet winner], [area of D (Voronoi: 0.208)],
  table.hline(stroke: 0.5pt),
  [Beta, $D = 0.3$], [12 705 pixels (14.2 %)], [1 101 pixels (1.23 %)], [0.312],
  [Beta, $D = 0.2$], [4 980 pixels (5.6 %)], [48 pixels (0.05 %)], [0.247],
  [normal, any $D$], [0], [0], [0.208],
  table.hline(),
))

The centre candidate D gains most: the Beta electorates of the whole upper right part
of the square are pulled towards the centre, where D is.

== Curved borders <sec-round-edges>

Under normal voters the borders of Condorcet methods are bisectors, i.e. straight
lines. The borders of FPTP and IRV are curved in both models, and the Beta model adds
bends near the walls of the square. The key is to look at a first-choice share as a
function of the pixel: it is the Voronoi cell of the candidate, *blurred*.

=== First-choice shares are blurred Voronoi cells

The FPTP winner changes where two first-choice shares (@eq-first-choice) are equal,

$ s_i (m) = s_j (m), quad s_i (m) = P_m (X in V_i) = integral_(V_i) f_m (p) dif p, $

with $V_i$ the Voronoi cell of $c_i$ and $f_m$ the voter density of pixel $m$. For normal
voters $f_m (p) = phi_sigma (p - m)$ is one Gaussian bump moved to $m$, and since it is
symmetric,

$ s_i = bb(1)_(V_i) * phi_sigma , $ <eq-blur>

the indicator of the cell (1 inside, 0 outside) convolved with a Gaussian of radius
$sigma$: the cell *blurred* like an out-of-focus photograph. The FPTP diagram is
obtained by blurring every Voronoi cell and colouring each pixel by the brightest one;
for $sigma -> 0$ this is the Voronoi diagram itself. IRV does the same once per round,
with the cells $V_i^S$ of the candidates $S$ still in the race. A border is therefore a
*contour line* where two blurred cells are equally bright, and the question is when
such a contour line is straight.

For Beta voters there is no single bump: $f_m$ changes its shape with $m$ (U-shaped,
crowded against the walls, @fig-densities and @fig-voters-wall), so the blur itself
varies across the square. This is the source of the bends near the walls below.

=== Straight versus round: blurring a corner

*Half-plane: straight.* With two candidates each cell is a half-plane, and by
@eq-normal-pairwise its blurred version is $Phi(delta slash sigma)$, a function of the
distance $delta$ from the bisector only. All its contour lines are parallel to the
bisector, and the tie $s_i = s_j = 1/2$ is the bisector itself.

*Corner: round.* Let a cell be the quadrant $Q = {x < 1/2, y < 1/2}$. The two
coordinates of the normal are independent, so its blurred version is exactly

$ s_Q (m) = Phi((1/2 - m_x) / sigma) thin Phi((1/2 - m_y) / sigma) . $ <eq-blur-corner>

Far from the corner one factor is $1$ and the contour lines are straight lines
parallel to the edges of $Q$. At the corner they bend round (@fig-blur-corner, middle).
For example the contour $s_Q = 1/2$ approaches the edges of $Q$ far away, but on the
diagonal it needs $Phi(u)^2 = 1/2$, $u = Phi^(-1)(1 slash sqrt(2)) = 0.545$: it passes
$0.545 sqrt(2) sigma approx 0.77 sigma$ *inside* the corner. Blurring rounds the corner
with a radius of the order of $sigma$, exactly as blurring a photograph of a square
gives a rounded square.

*Vertices: the widest cell wins.* A cell that is a wedge with angle $alpha$ at a
Voronoi vertex receives exactly $alpha slash (2 pi)$ of the voters of the pixel centred at
that vertex (the Gaussian is rotationally symmetric). Where several cells meet, the one
with the largest angle wins at the vertex, and the fronts of the others recede by a
distance of the order of $sigma$ along round contour lines. Far from the vertices
(distance $>> sigma$) only two cells are nearby and the border returns to their
bisector.

#figure(
  image("figures/blur_corner.png", width: 100%),
  caption: [Three candidates whose cells meet at $(1/2, 1/2)$ with angles
    $90 degree$, $135 degree$, $135 degree$ (normal voters, $sigma = 0.25$). Left: the
    quadrant cell of the lower-left candidate. Middle: its share (@eq-blur-corner) with
    contour lines. Right: FPTP. At the vertex the shares are $1/4$, $3/8$, $3/8$, so
    the quadrant candidate loses its corner, along a round front.],
) <fig-blur-corner>

=== Seven candidates

@fig-seven shows FPTP for seven candidates (B removed, three added on the left). With a
small blur ($D = 0.1$) the diagram is the Voronoi diagram with rounded corners at the
vertices. With $D = 0.3$ the blur is wider than several cells:

- the centre candidate $(0.5, 0.5)$ has a thin band as its cell. Blurred, a thin band
  stays dim, so under normal voters it wins only a small patch in the far corner of its
  band, not even its own position;
- borders appear between candidates whose cells do not even touch, e.g. $(0.3, 0.7)$
  and $(0.6, 0.35)$. Such a border has no bisector to follow at all: it is purely a
  contour line of two blurred cells, and it is curved.

The Beta diagram shows the same effects, and in addition the blur changes from pixel to
pixel: the centre candidate, helped by the pull towards the centre, wins a large region in
the upper right, and the borders bend where the shape of the voters changes fastest, near
the walls.

#figure(
  image("figures/seven_fptp.png", width: 100%),
  caption: [FPTP for seven candidates; dashed: Voronoi borders. From left: no blur
    (Voronoi), normal voters with $D = 0.1$ and $D = 0.3$, Beta voters (`rms`) with
    $D = 0.3$.],
) <fig-seven>

=== Flares at the walls (FPTP) <sec-flare>

#figure(
  image("figures/fptp_edge.png", width: 85%),
  caption: [FPTP. With Beta voters the region of B flares out in the last few
    hundredths before the left wall; with normal voters it widens smoothly.],
) <fig-fptp-edge>

B $= (0.25, 0.4)$ is the leftmost candidate and its Voronoi cell widens towards the left
wall: the bisector B|C ($y = x + 0.05$) falls as $x$ decreases, and at $x = 0$ the cell
spans $0.05 < y < 0.596$. The more voters sit close to $x = 0$, the more of them fall
into the wide part of B's cell. Heights $y$ of the borders of B's region in the column
of median $x$ ("–": B wins nowhere in the column):

#align(center, table(
  columns: 6,
  align: right,
  stroke: none,
  table.hline(),
  [median $x$], [$P(X < 0.02)$], [Beta B|E], [Beta C|B], [normal B|E], [normal C|B],
  table.hline(stroke: 0.5pt),
  [0.200], [0.212], [–], [–], [0.455], [0.238],
  [0.080], [0.357], [0.355], [0.272], [0.505], [0.125],
  [0.030], [0.467], [0.398], [0.218], [0.525], [0.075],
  [0.005], [0.581], [0.438], [0.172], [0.535], [0.048],
  table.hline(),
))

With Beta voters B's region only begins near the wall and then widens fast: over the
last $0.075$ the B|E border rises by $0.083$, against $0.030$ with normal voters.
@fig-voters-wall shows why. As the median approaches the wall, the half of the Beta
voters on the wall side is squeezed into an ever thinner strip next to the wall, right
into the wide part of B's cell ($P(X < 0.02)$ grows from $0.21$ to $0.58$), and B's
share grows from $0.22$ to $0.33$ between $x = 0.2$ and $x = 0.02$. The normal cloud keeps its
shape and only moves; its left half continues beyond $x = 0$, where B's cell simply
keeps widening, so nothing piles up at the wall.

#figure(
  image("figures/voters_wall.png", width: 80%),
  caption: [Voters of the pixels $(x, 0.4)$ approaching the left wall, coloured by their
    first choice (FPTP), with the Voronoi borders; titles: shares of B, C and E.],
) <fig-voters-wall>

The flare is a property of any wide Beta, not of the spread rule: it is nearly the
same under `mean_abs` (@ch-spread). It fades with less spread: at
$D = 0.2$ the B|E border rises by only $0.030$, like normal voters at $D = 0.3$.

== Summary

- Normal voters are symmetric about the pixel centre, so pairwise majorities follow
  distance: no cycles, and every Condorcet method draws the Voronoi diagram, whatever
  $sigma$.
- Beta voters have the pixel centre as the median of each coordinate only. Majorities
  between candidates on a diagonal are decided by the skew of the distribution, which
  pulls the effective centre towards the middle of the square: Condorcet regions are
  distorted (14 % of the pixels at $D = 0.3$, 6 % at $D = 0.2$) and cycles appear.
- A first-choice share is the candidate's Voronoi cell blurred by the voter
  distribution. Blurring keeps half-planes straight but rounds corners (radius of order
  $sigma$), so FPTP and IRV borders are curved in both models.
- Beta voters change shape across the square and crowd against the walls, so the blur
  varies from pixel to pixel; near the walls this produces flares such as B's region in
  FPTP.

#pagebreak()

= Choosing the spread rule <ch-spread>

@sec-spreads defined two spread rules. They agree at the centre pixel and differ only
in how the spread changes towards the walls. This chapter shows what that changes in
the diagrams, explains which property of a rule keeps borders straight, and gives the
evidence for the default `rms`. Unless stated otherwise the numbers use the candidates
A–E, $D = 0.3$ and $300 times 300$ pixels. `docs/figures.py` reproduces the figures and
the numbers on the default candidates; the benchmark over random candidate layouts
(@sec-bench) was run with separate scripts.

== The two rules on the default candidates

#figure(
  image("figures/spread_rules.png", width: 100%),
  caption: [The two spread rules of @sec-spreads on the default candidates,
    $D = 0.3$. Only `mean_abs` has the round IRV edge. Black: no Condorcet winner.],
) <fig-spread-rules>

#align(center, table(
  columns: 5,
  align: (left, right, right, center, right),
  stroke: none,
  table.hline(),
  [voter model], [cycle pixels], [Schulze $!=$ Voronoi], [round IRV edge], [flare of B],
  table.hline(stroke: 0.5pt),
  [Beta, `rms`], [1.23 %], [14.2 %], [no], [0.083],
  [Beta, `mean_abs`], [2.06 %], [20.2 %], [yes], [0.096],
  [normal], [0], [0], [no], [0.030],
  table.hline(),
))

(Flare of B: rise of the B|E border over the last $0.075$ before the left wall,
@sec-flare.) Compared with `rms`, `mean_abs` has 1.7 times as many cycle pixels, more
distorted Condorcet regions and a round IRV edge. The FPTP flare is almost the same
under both rules: it comes from the crowding of voters at the wall, which does not
depend on the rule.

== The round IRV edge of `mean_abs` <sec-irv-edge>

#figure(
  image("figures/irv_round.png", width: 100%),
  caption: [Beta IRV (faded) with the three round 3 tie curves among A, D, E, under
    `mean_abs` (left) and `rms` (right). The red curve $s_A = s_D$ bounds D's region.
    Gray: the bisectors D|A and D|E.],
) <fig-irv-round>

In the upper right part B and C are eliminated first. In round 3 A, D and E remain,
and D's cell among them is the diagonal band between the bisectors D|A and D|E that
widens towards the corner $(1, 1)$; A has the bottom right, E the top left. Whoever of
A and D has fewer votes is eliminated:

- D out: the final is A against E, and E wins;
- A out: the final is D against E, and D wins near the band (the pull towards the
  centre again).

So the lower edge of D's region is the round 3 tie $s_A = s_D$ (red in
@fig-irv-round). Along the column $x = 0.55$ (round 3 shares; winner of the pixel):

#align(center, table(
  columns: 9,
  align: right,
  stroke: none,
  table.hline(),
  [], table.cell(colspan: 4, align: center)[`mean_abs`], table.cell(colspan: 4, align: center)[`rms`],
  [$y$], [winner], [$s_A$], [$s_D$], [$P(Y < 0.2)$], [winner], [$s_A$], [$s_D$], [$P(Y < 0.2)$],
  table.hline(stroke: 0.5pt),
  [0.678], [E], [0.327], [0.284], [0.195], [E], [0.316], [0.292], [0.173],
  [0.738], [E], [0.303], [0.286], [0.185], [D], [0.279], [0.300], [0.146],
  [0.798], [E], [0.285], [0.284], [0.182], [D], [0.243], [0.307], [0.123],
  [0.858], [D], [0.274], [0.279], [0.187], [D], [0.209], [0.311], [0.102],
  [0.918], [D], [0.270], [0.270], [0.199], [E], [0.176], [0.312], [0.087],
  [0.978], [E], [0.274], [0.254], [0.226], [E], [0.144], [0.307], [0.079],
  table.hline(),
))

Under `rms`, going up the column, voters leave A's cell for D's band and E's part of
the square, $s_A$ falls steadily and the tie $s_A = s_D$ is crossed once, along a
nearly straight curve. (At the top D still survives round 3 but loses the final to E.)
Under `mean_abs` $s_A$ stops falling and rises again: as the median of $Y$ approaches
$1$, the fixed mean absolute deviation pushes the lower half of the voters down
(@eq-push), $P(Y < 0.2)$ grows from $0.182$ to $0.226$, and those voters are in A's
cell. D is ahead of A only in a middle band, so the tie curve closes around it into the
round edge.

== What keeps a border straight <sec-straight>

Every rule is a curve $kappa(m)$ (@sec-spreads). The border between $c_i$ and $c_j$ is
where the median of the projected voters $nu dot X$ equals the bisector value. That
median lies at some distance from $nu dot m$, the pull of the pixel towards the centre,
and the border is straight when the pull changes linearly along it.

*Borders close to an axis (exact).* For a small tilt $epsilon$,
$F_(X + epsilon Y)(t) = E[F_X (t - epsilon Y)] = F_X (t) - epsilon f_X (t) E[Y] + O(epsilon^2)$,
so

$ "median"(X + epsilon Y) = "median"(X) + epsilon E[Y] + O(epsilon^2), $

and the pull is $epsilon (E[Y] - "median"(Y))$. Such borders are straight exactly when
median minus mean grows in proportion to the distance of the median from the centre,
i.e. when $(m - E X) slash (m - 1/2)$ is constant. No small-spread assumption is needed
(checked numerically). This ratio for several rules:

#align(center, table(
  columns: 6,
  align: (left, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [rule], [$m = 0.6$], [0.7], [0.8], [0.9], [0.98],
  table.hline(stroke: 0.5pt),
  [fixed $a + b$], [0.40], [0.39], [0.38], [0.35], [0.27],
  [`rms`], [0.40], [0.40], [0.39], [0.37], [0.35],
  [`mean_abs`], [0.41], [0.44], [0.48], [0.54], [0.59],
  table.hline(),
))

There are two ways to fail. With a fixed $a + b$ the far tail thins out near a wall,
the mean catches up with the median and the pull stalls. With a fixed $E|X - m|$ the
far half is pushed away (@eq-push) and the pull accelerates. `rms` lies in between,
close to constant.

*Diagonal borders.* At $45 degree$ both coordinates are skewed at once. Exact border
shapes bend one way for fixed $a + b$ and the other way for fixed $E|X - m|$, confirming
the two failure modes. Where the balance between them lies depends on the position in
the square, so no single curve $kappa(m)$ keeps every border straight.

*Why cycles survive.* Straightness depends on how the pull changes with the position
of the pixel, cycles on how it changes with the direction $nu$. Projections of two
skewed coordinates in different directions do not share one effective centre, so the
three borders of three candidates can each be straight and still miss each other. No
spread rule removes cycles without removing the spread.

== Benchmark <sec-bench>

The *bend* of a rule is the largest distance of each pairwise majority border
$pi_(i j) = 1/2$ from its chord, averaged over the pairs (units of the square; $0$ for
normal voters). At the same $D = 0.3$, on the default candidates:

#align(center, table(
  columns: 6,
  align: (left, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [rule], [$kappa(0.7)$], [$kappa(0.9)$], [$kappa(0.98)$], [bend], [cycle pixels],
  table.hline(stroke: 0.5pt),
  [`mean_abs` ($L_1$)], [1.012], [0.554], [0.307], [0.046], [2.06 %],
  [$L_(1.5)$ from median], [1.096], [0.807], [0.557], [0.015], [1.55 %],
  [`rms` ($L_2$)], [1.197], [1.062], [0.799], [0.009], [1.23 %],
  [$L_3$ from median], [1.430], [1.546], [1.250], [0.010], [0.82 %],
  [$L_4$ from median], [1.684], [1.990], [1.667], [0.011], [0.59 %],
  [fixed $a + b$], [1.204], [1.204], [1.204], [0.024], [1.21 %],
  [SD about the mean], [1.086], [0.779], [0.557], [0.015], [1.59 %],
  [far-side mean distance], [1.826], [1.624], [1.045], [0.012], [0.43 %],
  [far-side RMS], [2.112], [2.078], [1.514], [0.010], [0.31 %],
  [far-side median distance], [1.444], [1.096], [0.622], [0.026], [0.75 %],
  [SD of $logit(X)$], [1.320], [2.262], [8.106], [0.038], [0.98 %],
  table.hline(),
))

"Far-side" measures use only the half of the voters towards the centre of the square.
Too little decrease of $kappa$ (fixed $a + b$) and too much (`mean_abs`) both bend
borders. But a comparison at the same $D$ mixes the rule with the amount of spread:
cycles and bend both grow with the spread, and a rule that makes pixels near the walls
narrower (larger $kappa$) gets fewer cycles and less bend together. The fair
comparison is the bend at the *same share of cycle pixels*.

The benchmark therefore swept $D in {0.2, 0.25, 0.3, 0.35, 0.4}$ for every rule and
compared the bend at equal shares of cycle pixels, interpolating in $D$. The score is
the bend of the _visible_ Condorcet borders (where both candidates beat all others),
on layouts of 3–6 candidates drawn uniformly from $[0.1, 0.9]^2$. On a first set of 24
layouts most rules bend more than `rms` (bend relative to `rms`): fixed $a + b$ 2.83,
`mean_abs` 2.41, RMS of $arcsin sqrt(X)$ 2.00, far-side RMS 1.84, $L_3$ 1.35,
$L_(2.5)$ 1.21, $L_(2.25)$ 1.10. The power means below $2$ were checked on 30 new
layouts, with 95 % bootstrap intervals over layouts, at three cycle levels (those of
`rms` at $D = 0.25, 0.3, 0.35$):

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: none,
  table.hline(),
  [rule], [$D = 0.25$ level], [$D = 0.3$ level], [$D = 0.35$ level],
  table.hline(stroke: 0.5pt),
  [$L_(1.75)$], [1.01 [0.96, 1.05]], [0.92 [0.88, 0.98]], [0.77 [0.73, 0.80]],
  [$L_(1.5)$], [1.14 [1.05, 1.24]], [0.94 [0.83, 1.08]], [0.67 [0.60, 0.77]],
  table.hline(),
))

They bend as much as `rms` or more at the lowest level and less at the higher ones. In
absolute terms the bend of `rms` is small: at $D = 0.3$ its visible borders bend by 1.5
pixels on average (90th percentile 3.1 pixels, the single worst border 4.9 pixels).

== Other ways to fix the spread

- *Median absolute deviation.* For every Beta, $"median"|X - m| < min(m, 1 - m)$: the
  near half lies entirely within that distance. A fixed median absolute deviation is
  therefore impossible within $D$ of every wall.
- *Normal voters clamped to the square* (moved onto the nearest edge when they would
  leave it) have neither the round IRV edge nor a strong flare (0.023), few cycles
  (0.10 %) and Schulze $!=$ Voronoi on 7.9 % of the pixels. But they put point masses
  on the edges, are not a Beta, and their deviation shrinks near the walls.

== Conclusion

- *`rms`* is the default, and the only rule of the web UI. It fixes a quantity with a
  clear meaning, pushes the far half of the voters out only by $sqrt(2)$ near a wall,
  has no round IRV edge, and its borders bend by 1.5 pixels on average at $D = 0.3$; of
  the rules compared, only power means slightly below $2$ bend less, and only at larger
  spreads.
- *`mean_abs`* is kept to reproduce results made with it (caches and plots are kept
  apart per rule, `pixels/cache/beta/<spread>/`, `pixels/plots/beta/<spread>/`). By @eq-push it
  pushes the far half of the voters out twice as far near a wall, which bends borders
  the most, adds cycles and makes the round IRV edge.

#pagebreak()

= Shapes of win regions <ch-shapes>

A rule of thumb about Yee diagrams links the shape of the regions with monotonicity:
methods that are not monotone, such as IRV, draw regions that are not convex and
sometimes fall into several pieces. This chapter makes the link precise. The shape of a
region is decided by the method and the voter model together, and the rule of thumb
holds only in special cases. Monotone methods can have non-convex regions and regions in
several pieces, and non-monotone methods can have convex regions.

== Regions, shapes and monotonicity

The *win region* $R_i$ of candidate $c_i$ is the set of pixel centres $m in (0, 1)^2$
where $c_i$ wins. Apart from exact ties, which only happen on curves (@sec-ties), every
method decides its winner by strict inequalities between continuous functions of $m$:
first-choice shares, pairwise shares, IRV totals or Schulze path strengths. So $R_i$ is
an open set. Three shapes, from the strongest to the weakest:

- $R_i$ is *convex* if the segment between any two of its points lies in $R_i$.
- $R_i$ is *star-shaped* around a point $q$ if the segment from $q$ to any point of $R_i$
  lies in $R_i$. Star-shaped around the candidate's own position is the natural property
  between the other two.
- $R_i$ is *connected* if it is not the union of two disjoint non-empty open sets. For an
  open set in the plane this is the same as *path-connected*: any two of its points are
  joined by a continuous path inside it.

A convex region is star-shaped around each of its points, and a star-shaped region is
connected. On the pixel grid a region is *in pieces* when its pixels fall into groups
that do not touch, not even at a corner.

*Curved borders.* Two neighbouring regions can both be convex only if the border between
them is straight: two disjoint open convex sets are separated by a line (the separation
theorem), so their common border lies on that line. Hence *a curved border always has a
non-convex region on at least one side*, and every method that draws curved borders has
non-convex regions.

*Monotonicity.* A method is *monotone* if a winner stays the winner when some voters
raise them on their ballots and the order of the other candidates stays the same. FPTP,
Borda and Schulze are monotone, and a Condorcet winner stays the Condorcet winner when
raised. IRV is not monotone: raising the winner can change who is eliminated first, and
so give the winner a stronger opponent in the final.

== Why monotonicity does not fix the shape <sec-shape-why>

Monotonicity compares two electorates that differ in one way only: some voters raise one
candidate. Moving the pixel is a different change. Every voter moves, and the rankings
change among all candidates, including pairs that do not involve the winner. Even moving
the pixel towards a candidate is not the same as raising that candidate, because the
voters beyond the candidate move away from it. So monotonicity says nothing about the
winners of neighbouring pixels.

Another property that might seem to fit does not help either. A *consistent* method
(FPTP and Borda are consistent) that elects $c_i$ in two electorates also elects $c_i$
when the two electorates are pooled. But the voters of the midpoint between two pixels
are not the pooled voters of the two pixels: for normal voters they are one Gaussian
bump at the midpoint, not two bumps.

== Where the regions are convex

=== Condorcet methods and symmetric voters

For normal voters, and more generally for voters that are centrally symmetric about the
pixel centre, every Condorcet method draws the Voronoi diagram (@sec-normal-majority),
whose regions are convex polygons. This includes Condorcet methods that are not
monotone, such as Baldwin's and Nanson's methods: they elect the Condorcet winner
whenever there is one, and for these voters there always is one. So a non-monotone
method can have convex regions. Their convexity comes from the Condorcet criterion
together with the symmetry of the voters, not from monotonicity.

=== Candidates on a line: FPTP versus IRV <sec-collinear>

Let the candidates lie on a line with unit direction $u$. Their Voronoi cells are strips
perpendicular to the line, $V_i = {p : p dot u in I_i}$, with intervals
$I_1 < I_2 < dots$ in the order of the candidates. For normal voters the projection
$X dot u$ is $cal(N)(t, sigma^2)$ with $t = m dot u$, so every share depends only on $t$,
and every region is a union of strips.

*FPTP regions are single strips.* For $t < t'$ the ratio of the two projected densities,

$ L(x) = phi((x - t') slash sigma) / phi((x - t) slash sigma) = exp(((t' - t) x - (t'^2 - t^2) slash 2) / sigma^2), $

increases in $x$. Moving the pixel from $t$ to $t'$ multiplies the share of cell $k$ by
the average of $L$ over $I_k$, and for $i < j$ the interval $I_j$ lies to the right of
$I_i$, so the share of $c_j$ grows by a larger factor than that of $c_i$: the ratio
$s_j (t) slash s_i (t)$ never decreases. Hence $c_i$ beats $c_j$ on a half-line of $t$,
to the left of some point for $j > i$ and to the right of some point for $j < i$. The
FPTP region of $c_i$ is the intersection of these half-lines, one interval of $t$: a
strip, which is convex. The argument only uses that $L$ increases, which holds whenever
the voters of a pixel are one fixed log-concave density moved to the pixel. Beta voters
are not of this kind, and near the walls they are not even stochastically ordered
(@sec-stochastic), so the argument does not carry over to them.

*IRV splits the middle candidate.* Take L $= (0.2, 0.5)$, M $= (0.5, 0.5)$ and
R $= (0.8, 0.5)$ with normal voters, $D = 0.3$ ($sigma = 0.376$). At the centre, M is the
nearest candidate and the Condorcet winner, but it has the fewest first choices,
$2 Phi(0.15 slash sigma) - 1 = 0.310$ against $0.345$ each for L and R. So M is
eliminated first: the _centre squeeze_. At $x = 0.4$ the shares are L 0.445, M 0.300 and
R 0.254, so R is eliminated first and M beats L in the final. M therefore wins IRV on
two strips, $0.35 < x < 0.46$ and $0.54 < x < 0.65$, but not between them, and the regions
of L and R are in two pieces as well (@fig-collinear). FPTP gives L and R one strip each
and M nothing.

Here the rule of thumb is exactly right: the monotone method keeps every region in one
piece and the non-monotone one does not. Both properties of IRV have the same cause:
who meets whom in the final depends on who is eliminated first.

#figure(
  image("figures/collinear.png", width: 100%),
  caption: [Three candidates on a line, normal voters, $D = 0.3$. FPTP gives every
    candidate at most one strip. In IRV the middle candidate M is squeezed out near the
    centre, where it has the fewest first choices (right), so M wins two separate strips,
    and so do L and R.],
) <fig-collinear>

== Monotone methods with non-convex regions

*How often.* The random layouts below count a region as non-convex when its convex hull
contains pixels that are more than 3 pixels away from the region; pixelation alone
cannot cause that. @tab-shapes counts, on 100 random layouts of 3–6 candidates in
$[0.05, 0.95]^2$ ($150 times 150$ pixels, `docs/figures.py search`), the layouts with at
least one such region, and the layouts with a region in pieces.

#figure(
  table(
    columns: 6,
    align: (left, center, right, right, right, right),
    stroke: none,
    table.hline(),
    [], [], table.cell(colspan: 2, align: center)[normal voters],
    table.cell(colspan: 2, align: center)[Beta voters (`rms`)],
    [method], [monotone], [non-convex], [in pieces], [non-convex], [in pieces],
    table.hline(stroke: 0.5pt),
    [FPTP], [yes], [42 / 21], [0 / 0], [34 / 47], [0 / 0],
    [Borda], [yes], [35 / 24], [0 / 0], [53 / 52], [0 / 0],
    [Schulze], [yes], [0 / 0], [0 / 0], [8 / 49], [0 / 0],
    [IRV], [no], [90 / 95], [25 / 45], [94 / 98], [26 / 45],
    table.hline(),
  ),
  caption: [Layouts out of 100 with at least one non-convex region, or with a region in
    pieces (more than one piece of at least 20 pixels), at $D = 0.2$ / $D = 0.3$.],
) <tab-shapes>

*Normal voters.* The FPTP and Borda borders are contour lines of blurred cells and are
curved (@sec-round-edges), so by the argument above some of their regions are not
convex. The dent is often only a few pixels deep. Schulze draws the Voronoi diagram and
is always convex.

*Beta voters and Schulze.* Under Beta voters Schulze regions are not convex either.
@fig-schulze-notch shows an example with $c_1 = (0.457, 0.407)$, $c_2 = (0.254, 0.289)$,
$c_3 = (0.334, 0.641)$ and $c_4 = (0.235, 0.464)$, `rms` and $D = 0.3$. $c_1$ wins at both
ends of the segment, $c_4$ in most of it. The notch is not a bend of a pairwise border:
along the whole segment $c_4$ beats $c_1$ head to head, with $0.508$ to $0.516$ of the
voters, and in the middle $c_4$ is the Condorcet winner. The two ends lie in pockets of
Condorcet cycles; in each the fourth candidate loses to the other three:

- at the upper end $c_1 > c_3 > c_4 > c_1$, with $0.512$, $0.528$ and $0.508$ of the voters;
- at the lower end $c_1 > c_2 > c_4 > c_1$, with $0.530$, $0.537$ and $0.516$.

In a cycle of three candidates Schulze elects the candidate beaten by the weakest of
the three defeats. Next to the tie line of $c_1$ and $c_4$ the weakest defeat is that of
$c_1$ by $c_4$, barely above one half. So Schulze gives $c_1$ the part of each pocket along that line, on $c_4$'s side of
it. In general Schulze splits a cycle pocket among its three candidates, each winning
the part next to the tie line of its own defeat. Every Schulze region therefore reaches
across its pairwise borders into the neighbouring pockets, and between two such pockets
the region has a notch. Normal voters have no pockets: their three pairwise borders meet
in one point, the Voronoi vertex.

#figure(
  image("figures/schulze_notch.png", width: 100%),
  caption: [Left: Schulze with Beta voters (`rms`, $D = 0.3$). $c_1$ (blue) wins at both
    ends of the segment, $c_4$ (red) in between. Right: detail with the Condorcet winner
    (black: cycles). $c_1$'s Schulze region, to the right of the white line, reaches
    across the tie line of $c_1$ and $c_4$ (yellow) into two cycle pockets.],
) <fig-schulze-notch>

== Monotone methods with regions in pieces <sec-pieces>

FPTP can split a region, already under normal voters. Take seven candidates:
A $= (0.5, 0.5)$; N $= (0.5, 0.6)$ and S $= (0.5, 0.4)$ just above and below it; and
NW $= (0.3, 0.95)$, NE $= (0.7, 0.95)$, SW $= (0.3, 0.05)$, SE $= (0.7, 0.05)$ far above
and below. A's Voronoi cell is the thin strip $0.45 < y < 0.55$.

With normal voters and $D = 0.12$ ($sigma = 0.150$), N and S are strongest in the middle
column: at A's own position A has $0.260$ of the first choices and N and S have $0.320$
each, and along the whole column $x = 1/2$ A trails the best rival by at least $0.060$.
Further out, the voters above the strip are split between N and NW or NE, and those
below between S and SW or SE. Every rival is weaker there, and A wins. So A's region is
in two pieces, $0.13 < x < 0.28$ and $0.72 < x < 0.87$ (both at $0.46 < y < 0.54$),
separated by the column where A never wins (@fig-disconnected). Beta voters (`rms`, same
$D$) give the same picture, with pieces at $0.08 < x < 0.25$ and $0.75 < x < 0.92$; along
the column A trails by at least $0.081$.

A does not win at its own position, so its region is not even star-shaped around A. The
regions of NW, NE, SW and SE are not convex either: the domes of N and S bite into them.
Layouts like this one are rare. In the random layouts of @tab-shapes no region of a
monotone method is in pieces, while IRV has regions in pieces in a quarter to a half of
them.

#figure(
  image("figures/disconnected_fptp.png", width: 90%),
  caption: [FPTP with normal and Beta voters, $D = 0.12$. Dashed: Voronoi borders;
    dotted: the column $x = 1/2$, where A (red) never wins. A's region is in two pieces.],
) <fig-disconnected>

== Condorcet methods under Beta voters <sec-stochastic>

For Condorcet methods under Beta voters, the random layouts gave non-convex regions but
never regions in pieces. Whether Beta voters can split a Condorcet region is open. This
section collects what is known.

*Two candidates.* Every method then elects the majority winner, and the region of $c_i$
is ${pi_(i j) > 1/2}$, bounded by one bent border. Suppose that the voters of a pixel
become stochastically larger in each coordinate as its median grows, i.e. that
$P(X <= t)$ never increases with the median, for any $t$. Then $nu dot X$ becomes
stochastically larger as $m_x$ grows if $nu_x > 0$ and smaller if $nu_x < 0$, and likewise
for $m_y$. So $pi_(i j)$ is monotone in each coordinate of the pixel, and with every pixel
the region contains the whole rectangle between that pixel and one corner of the square.
Any two points of the region are then joined through that corner: the region is
connected, and its border is the graph of a monotone function. Normal voters have this
property, because they only shift.

Beta voters do not quite have it. As the median moves towards a wall, the far half of
the voters is pushed out (@sec-spreads), and $P(X <= t)$ grows for some $t$. The table
gives, for medians up to $1 - 1 slash 800$ (the last pixel of a $400 times 400$ grid), the
median from which $P(X <= t)$ first grows, and the largest increase of $P(X <= t)$
between two medians, over all $t$.

#align(center, table(
  columns: 5,
  align: (left, right, right, right, right),
  stroke: none,
  table.hline(),
  [], table.cell(colspan: 2, align: center)[$D = 0.2$], table.cell(colspan: 2, align: center)[$D = 0.3$],
  [rule], [grows from], [largest rise], [grows from], [largest rise],
  table.hline(stroke: 0.5pt),
  [`rms`], [0.773], [0.011], [0.900], [0.018],
  [`mean_abs`], [0.673], [0.097], [0.654], [0.146],
  table.hline(),
))

So the argument does not apply. Numerically, no two-candidate region was in pieces: 0 of
900 layouts (9 directions of the bisector times 100 positions) for each of `rms` and
`mean_abs` at $D = 0.2$, $0.3$ and $0.4$.

*More candidates.* The Condorcet region of $c_i$ is the intersection of its pairwise
regions. For it to fall into pieces, two of its borders would have to cross each other
twice. They are bent by only a few pixels, so this would need two nearly parallel
borders close together.

== Summary

- Convex regions come from the voter model and the method together. Under normal voters
  every Condorcet method, monotone or not, draws convex regions, and FPTP does for
  candidates on a line.
- Monotone methods can have non-convex regions (FPTP and Borda under both models,
  Schulze under Beta voters, @fig-schulze-notch) and regions in pieces (FPTP,
  @fig-disconnected).
- So neither the shape nor the connectedness of a region proves anything about
  monotonicity. What holds is weaker: in the random layouts only IRV split regions, and
  for candidates on a line FPTP can never split a region, while IRV does.

#pagebreak()

= Real-time diagrams in the web UI <ch-realtime>

The web UI redraws the diagram while a candidate is dragged. @ch-compute computes the
share of every ranking for every pixel, which is exact and serves every method, but
takes too long for that:

#align(center, table(
  columns: 3,
  align: (left, right, right),
  stroke: none,
  table.hline(),
  [step ($D = 0.2$, `rms`, $49 times 49$ nodes)], [5 candidates], [8 candidates],
  table.hline(stroke: 0.5pt),
  [ranking shares at the nodes (`compute_ranking_probabilities`)], [333 ms], [2878 ms],
  [bisector arrangement (`ranking_cells`)], [8 ms], [99 ms],
  [interpolation to $300 times 300$ pixels], [12 ms], [88 ms],
  [IRV / Schulze on the pixels], [61 / 64 ms], [703 / 358 ms],
  table.hline(),
))

Almost all of the time goes to the shares at the $N times N$ nodes, which do not depend
on the number of pixels at all. Drawing the diagram as curves instead of pixels
(@sec-zero-sets) is the right output, but on its own it saves little. The speed comes
from computing fewer integrals (@sec-needs), keeping those a drag does not change
(@sec-edge-cache), making each one cheaper (@sec-tables), and compiling the loops that
remain (@sec-compiled). This chapter describes `margin/` (`shares.py`, `beta_tables.py`
and `regions.py`; `geometric.py` is the subject of @ch-geometric) and the methods built
from blocks in `build/`, which `web/app.py` uses for the UI.
`docs/figures.py` (apart from the examples of @sec-zero-sets), the plots and the tests of
@sec-validation still use the pipeline of @ch-compute, which is in `pixels/`. The two never import each other; what both need
(`ranking_cells.py`, `normal.py`, `voting.py`, `threads.py`) is at the top of the package.

== What each method needs (`margin/shares.py`) <sec-needs>

The arrangement of all $C (C - 1) slash 2$ bisectors has $O(C^4)$ cells and edges: 43
cells and 65 slanted edges for 5 candidates, 272 cells and 468 slanted edges for 8. Most
methods need much less than the share of every cell.

*Pairwise shares.* Schulze, Minimax, Black and the Condorcet winner only use $pi_(i j)$
(@eq-pairwise). Borda does too: $C - 1 - "pos"_r (i)$ is the number of candidates below
$c_i$ in ranking $r$, so

$ "score"_i = sum_r P(r) (C - 1 - "pos"_r (i)) = sum_(j != i) pi_(i j) . $

Baldwin and Nanson take the same sum over the remaining candidates in every round
(@eq-borda-pairwise), so they need nothing else either.

Each $pi_(i j)$ is the share of one half-plane, the part of the square on $c_i$'s side
of the bisector: a convex polygon with one slanted edge. For 8 candidates that makes 28
slanted edges instead of 468. For normal voters no integral is needed at all:
$pi_(i j) = Phi(delta_(i j) (m) slash sigma)$ (@eq-normal-pairwise).

*First-choice shares.* FPTP only uses $s_i$ (@eq-first-choice), the share of the
Voronoi cell of $c_i$: $C$ convex polygons with about $3 C$ edges in all. For normal
voters the cells are taken in the box of @sec-normal and summed with Owen's T.

*The whole profile.* IRV needs the first choices among every set of remaining
candidates. In a benchmark with 8 candidates it visited 106 of the 247 sets of two or
more, so computing sets lazily saves little, and IRV uses the full arrangement.

All three are sums of edge terms: the Green integrals of @sec-edges for Beta voters and
the signed triangles of @sec-normal for normal voters.

*Score shares.* Score uses the part of the top score the voters do not give
each candidate. The regions of these voters have curved borders, so their shares have no
edge terms: they are summed over a grid of rectangles with the exact share of the voters
in each, at the $G + 2$ points per axis where the borders are traced and not at the
nodes (@sec-score). The voters' shares of the rectangles depend only on the voter
model and $G$. They are kept; computing them takes up to 0.2 s for Beta voters. The
rest is done again at every step of a drag: for five to eight candidates the whole grid
takes about 100 to 170 ms with six levels and 30 to 90 ms with two, at $G = 160$ and
$G = 320$ alike. Most of it is the ballots on the voter grid, which do not depend on $G$;
with more levels more rectangles have a border inside and are averaged at $8 times 8$
points.

== Dragging: an edge cache <sec-edge-cache>

An edge term depends only on the edge and the voter model. The terms are therefore
cached with the endpoints as key (rounded to $10^(-9)$, the smaller endpoint first, the
sign from the orientation, as in @sec-green), up to 64 MB, least recently used out
first. Dragging $c_k$ moves only the $C - 1$ bisectors through $c_k$; the half-planes and
Voronoi edges of all other pairs keep their endpoints and come from the cache. For
pairwise shares 4 of 10 half-planes are recomputed with 5 candidates, 7 of 28 with 8.
The arrangement gains less: every other bisector is cut by the moving ones, so for 5
candidates about 60 of its 89 edges are new after each step.

== Tabulated Beta CDF (`margin/beta_tables.py`) <sec-tables>

An edge integral evaluates $G_j$ at $Q$ points for every pair of nodes,
$Q N^2 approx 58 thin 000$ calls of `betainc` at about 400 ns each. The $N$ Beta
distributions of the nodes depend only on $D$ and the spread rule, so their CDFs can be
tabulated once.

The CDF is tabulated against $s = logit y$, with the exact slope

$ (dif G) / (dif s) = g(y) thin y (1 - y) . $

Near a wall $G(y) = y^a slash (a B(a, b)) (1 + O(y)) = e^(a s) slash (a B(a, b)) (1 + O(y))$:
an exponential in $s$, smooth even for $a, b < 1$, where the density itself is infinite.
A cubic Hermite interpolant with the exact slope on a uniform grid ($s in [-40, 40]$,
step $0.01$) is therefore accurate everywhere ($4 dot 10^(-9)$ for the narrowest voters).
Below the table ($y < 4 dot 10^(-18)$) the power law $y^a slash (a B(a, b))$ itself is
used, which is exact to double precision there; above it there is no double $y < 1$.
(A table of $logit G$ instead is smooth enough for a step of $0.02$, but evaluating it
costs an `exp` per point and node, which was most of the time of the compiled kernel.)
Two numerical details matter:

- Of $G$ and $1 - G$ the smaller is computed directly and the other as its complement.
  Using $1 - G(y) = G'(1 - y)$ with the mirrored Beta is not enough: $1 - y$ rounds to $1$
  below $y approx 10^(-16)$, where $G approx y^a$ is still $0.02$ for $a = 0.1$.
- The same grid in $s$ serves all nodes, so a point's interval and its position in it
  are computed once and reused for all $N$ nodes. The remaining work per point and node
  is one cubic, with the coefficients of all nodes in one interval stored together.

The quantile $F^(-1)$ has a table in logit–logit coordinates, $logit x$ against
$logit u$, for $u in [expit(-30), expit(30)]$ with step $0.02$. It is asymptotically
linear at both ends ($x approx (a B u)^(1 slash a)$ near $0$), so outside the table it is
extended linearly. A $u$-interval of width $w$ changes an edge integral by at most $w$, because
the integrand is bounded by 1, so values below $10^(-13)$ do not matter.

Over all offered deviations and spread rules the tables agree with `betainc` to
$4 dot 10^(-9)$ and with `betaincinv` to $2 dot 10^(-8)$ in $x$ (tests:
$10^(-8)$ and $10^(-7)$), and edge integrals agree with the exact ones to
$5 dot 10^(-9)$. With the compiled kernel of @sec-compiled an edge takes about 0.2 ms
instead of 23 ms. A table is built in about 0.2 s, in parallel over the nodes.

== Compiled kernels (numba) <sec-compiled>

With the tables in numpy an edge still took 1.3 ms: about 13 passes over temporary
arrays of $Q N^2$ elements, and threads ran edges only 1.5 times faster than one, since
numpy holds the interpreter lock between its many small operations. The loops that
remain are therefore compiled with numba:

- the edge integrals of @sec-edges on the tables, every form of the edge (vertical,
  horizontal, split, $u$ and $v$) in one kernel for a whole batch of edges; the lookup,
  the cubic and the quadrature sum happen in one pass (about 2 ns per point and node);
- the triangles of normal voters with Owen's T (@sec-normal);
- the clipping of a polygon by a bisector (`ranking_cells`), which with a vectorized
  test of which cells a bisector crosses takes the arrangement of 8 candidates from
  120 ms to 5 ms;
- the rounds of IRV (@ch-methods);
- which candidates a round of Baldwin or Nanson drops (`build/rounds.py`); their Borda
  scores (@eq-borda-pairwise) are one `einsum` of numpy per round.

The kernels release the interpreter lock but are not parallel themselves; `threads.py`
runs chunks of their work (edges, pixels) in a pool of threads. A kernel with numba's
own parallel loops would abort, with numba's default threading layer, as soon as two
threads call it at once, as two requests of the web UI can. Two details decided the
speed: indexing the coefficient table element by element (an array view per point and
node costs more than the cubic), and not calling `exp` in the innermost loop (this
numba has no vectorized `exp`), hence the table of $G$ itself. The first run compiles
the kernels; numba caches them on disk, and the UI loads them at start.

Compute per drag update (in process, best of several runs, $G = 160$), with the kernels
and before them:

#align(center, table(
  columns: 6,
  align: (left, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [], [candidates], [FPTP], [Borda], [Schulze], [IRV],
  table.hline(stroke: 0.5pt),
  [Beta, $D = 0.2$], [5], [5 (12) ms], [11 (23) ms], [15 (26) ms], [14 (170) ms],
  [], [8], [7 (27) ms], [22 (41) ms], [30 (61) ms], [79 (984) ms],
  [Beta, $D = 0.05$], [5], [7 (28) ms], [17 (48) ms], [18 (48) ms], [24 (366) ms],
  [], [8], [10 (52) ms], [31 (95) ms], [39 (96) ms], [162 (2308) ms],
  [normal, $D = 0.2$], [5], [6 ms], [10 ms], [14 ms], [31 ms],
  [], [8], [9 ms], [23 ms], [27 ms], [118 ms],
  table.hline(),
))

== Borders as zero sets (`build/`, `margin/regions.py`) <sec-zero-sets>

The shares are smooth in the median $m$; only the winner jumps. Each method compares
continuous functions of the shares, so its borders are where such a comparison is a tie.
The methods of the web UI return, next to the winner, a *margin* $mu >= 0$
that is continuous in the shares and $0$ on every border between two winners. It is the
smallest gap in a comparison that could change the winner, so it can also vanish where
such a comparison ties but the winner stays; that makes no border, because the sign of
$psi_c$ below changes only with the winner:

#align(center, table(
  columns: 2,
  align: (left, left),
  stroke: none,
  table.hline(),
  [method], [margin $mu$],
  table.hline(stroke: 0.5pt),
  [FPTP, Borda], [lead of the top score over the second],
  [Condorcet winner], [$min_(j != w) (pi_(w j) - pi_(j w))$; in a cycle $-max_i min_(j != i) (pi_(i j) - pi_(j i))$],
  [Schulze], [$min_(e != w) max_f (p_(f e) - p_(e f))$],
  [Minimax], [lead of the top weakest link over the second],
  [Black], [as the Condorcet winner; in a cycle the smaller of that and the margin of Borda],
  [IRV], [smallest gap between the two lowest tallies over all rounds],
  [Baldwin], [smallest gap between the two lowest Borda scores over all rounds],
  [Nanson], [smallest distance of a Borda score from the mean over all rounds],
  [Score], [lead of the top mean score over the second],
  table.hline(),
))

All methods are built from blocks (`yeelab.build`), and the margin is never put together
by hand: every block that decides computes the gap of its decision, and the margin is the
smallest of them. `Highest` has the lead of the top score, `Eliminate` the gap of each
round, `Unbeaten` the gaps below and `Fallback` those of the method that decides.
Baldwin and Nanson are in @sec-baldwin-nanson, Minimax and Black in @sec-minimax-black,
the score methods in @sec-score.

*IRV.* On each side of a curve where some round's two lowest tallies tie, the gap of
that round tends to $0$. So $mu$ is continuous, and it vanishes on every curve where an
elimination changes, whether or not the winner changes there.

*Unbeaten.* The Condorcet winner and Schulze both elect a candidate that no one beats,
from links $ell_(i j) = -ell_(j i)$ in which $c_i$ beats $c_j$ where $ell_(i j) > 0$: the
margins of @eq-link, or the strongest paths below. Let

$ "beaten"_e = max_(f != e) ell_(f e) $

be the strongest link into $c_e$, negative if $c_e$ beats everyone and positive if
someone beats it. The winner $w$ has the smallest $"beaten"_e$ (ties: the first). It
changes only where another candidate becomes unbeaten, that is, where some
$"beaten"_e$ with $e != w$ crosses $0$, so the margin is how far the closest of the
others is from that:

$ mu = min_(e != w) max("beaten"_e, 0). $ <eq-unbeaten>

*Weak and strict winners.* What the winner must be depends on the links. On the
strongest paths, "beats" ($p_(i j) > p_(j i)$) has no cycles, as Schulze proved, so
someone is unbeaten, $"beaten"_w <= 0$, and the winner is a _weak_ winner:
$p_(w e) >= p_(e w)$ for all $e$, as in @ch-methods, which may tie some candidates. The margins
$ell_(i j) = pi_(i j) - pi_(j i)$ can form a cycle. There the winner must be a _strict_
winner, a Condorcet winner with $"beaten"_w < 0$ who beats everyone; where the smallest
$"beaten"_w$ is not negative, the point is a cycle and marked `CYCLE`. That decision
flips where $"beaten"_w$ crosses $0$, so $abs("beaten"_w)$ is a gap as well:

$ mu = min(min_(e != w) max("beaten"_e, 0), abs("beaten"_w)) . $

For antisymmetric links the second term is never larger than the first. With a
Condorcet winner, $"beaten"_e >= ell_(w e) >= min_f ell_(w f) = -"beaten"_w$ for every
other $e$. In a cycle, $"beaten"_e >= "beaten"_w >= 0$. So $mu = abs("beaten"_w)$: the
narrowest win of the Condorcet winner, and, in a cycle, how far every candidate is from
beating everyone, as in the table.
For Schulze's links the gap is left out. There $"beaten"_w$ is $0$ on whole areas, where
the widest paths from $w$ to another candidate and back share their weakest link, and the
margin would vanish on an area where no border runs.

*Strongest paths.* With complete rankings $pi_(i j) + pi_(j i) = 1$, so the margin
$ell_(i j) = 2 pi_(i j) - 1$ orders the links exactly like the winning votes $pi_(i j)$
of @ch-methods, and the winners are the same. Unlike winning votes, the link strengths
$max(ell_(i j), 0)$ are continuous in $pi$, and so are the path strengths $p_(i j)$ built
from them. `StrongestPaths` returns the links $p_(i j) - p_(j i)$, which make
$"beaten"_e = max_(f != e) (p_(f e) - p_(e f))$, the expression of the table.

Where a Condorcet winner $w$ exists it is the Schulze winner: $p_(w e) >= ell_(w e) > 0$,
and no link leads into $w$, so $p_(e w) = 0$. The widest paths are therefore only
computed at the points without a Condorcet winner. Elsewhere the winner and margin of the
margins themselves stand in: the margin is positive, at most the margin on the paths
(as $p_(w e) >= ell_(w e)$), and $0$ on the border of the Condorcet region, so it
vanishes on the same borders.

*Fallback.* `Fallback(first, second)` elects the winner of `first`, and where `first`
elects no one (`CYCLE`), the winner of `second`; Black is the Condorcet winner with Borda
as the second. The winner changes where the decision of the method in charge flips, so
outside the cycles the margin is that of `first`, and inside a cycle the smaller of the
two:

$ mu = cases(mu_"first" & "where" "first" "elects someone", min(mu_"first", mu_"second") & "in a cycle of" "first") . $

In a cycle $mu_"first"$ is how far the point is from having a Condorcet winner, so it
vanishes on the border of the cycle, where `second` hands over to `first`, and
$mu_"second"$ on the borders `second` draws inside the cycle.

=== Example: margins at two medians <sec-margin-example>

The examples use the candidates A–E (@fig-cells) and the defaults of the UI: Beta voters,
`rms` and $D = 0.2$. @tab-margin-shares lists the exact shares at two medians:
$m_1 = (0.5, 0.5)$, the position of D, and $m_2 = (0.32, 0.545)$, in a small pocket
where B, D and E form a cycle. @tab-margins gives the winners and margins. The shares
are rounded to four decimals, while the margins come from the unrounded shares, so a
difference of two entries can be off by one in the last digit.

#figure(
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    stroke: none,
    table.hline(),
    [median], [], [A], [B], [C], [D], [E],
    table.hline(stroke: 0.5pt),
    [$m_1$], [$s_i$], [0.2697], [0.1164], [0.1303], [0.2565], [0.2271],
    [], [$"score"_i$], [2.1051], [1.6919], [1.8157], [2.7301], [1.6572],
    [], [$pi_(A j)$], [–], [0.5783], [0.5819], [0.3636], [0.5812],
    [], [$pi_(B j)$], [0.4217], [–], [0.4436], [0.3095], [0.5172],
    [], [$pi_(C j)$], [0.4181], [0.5564], [–], [0.3111], [0.5301],
    [], [$pi_(D j)$], [0.6364], [0.6905], [0.6889], [–], [0.7143],
    [], [$pi_(E j)$], [0.4188], [0.4828], [0.4699], [0.2857], [–],
    table.hline(stroke: 0.5pt),
    [$m_2$], [$s_i$], [0.1243], [0.2218], [0.1227], [0.1730], [0.3582],
    [], [$"score"_i$], [1.2748], [2.3343], [1.8258], [2.3824], [2.1827],
    [], [$pi_(A j)$], [–], [0.3342], [0.3539], [0.2084], [0.3784],
    [], [$pi_(B j)$], [0.6658], [–], [0.6587], [0.5206], [0.4891],
    [], [$pi_(C j)$], [0.6461], [0.3413], [–], [0.3949], [0.4435],
    [], [$pi_(D j)$], [0.7916], [0.4794], [0.6051], [–], [0.5062],
    [], [$pi_(E j)$], [0.6216], [0.5109], [0.5565], [0.4938], [–],
    table.hline(),
  ),
  caption: [Shares at $m_1$ and $m_2$ (Beta, `rms`, $D = 0.2$): first choices $s_i$
    (@eq-first-choice), Borda scores and pairwise shares $pi_(i j)$ (@eq-pairwise),
    candidate $i$ in the row, $j$ in the column.],
) <tab-margin-shares>

#figure(
  table(
    columns: 5,
    align: (left, left, right, left, right),
    stroke: none,
    table.hline(),
    [method], [winner at $m_1$], [$mu$], [winner at $m_2$], [$mu$],
    table.hline(stroke: 0.5pt),
    [FPTP], [A], [0.0132], [E], [0.1363],
    [Borda], [D], [0.6250], [D], [0.0481],
    [Baldwin], [D], [0.0347], [E], [0.0190],
    [Nanson], [D], [0.1051], [E], [0.0046],
    [Condorcet winner], [D], [0.2727], [none (cycle)], [0.0125],
    [Schulze], [D], [0.2727], [E], [0.0093],
    [Minimax], [D], [0.5455], [E], [0.0093],
    [Black], [D], [0.2727], [D], [0.0125],
    [IRV], [D], [0.0138], [E], [0.0016],
    table.hline(),
  ),
  caption: [Winners and margins at $m_1$ and $m_2$, as returned by the methods of the
    web UI.],
) <tab-margins>

*At $m_1$.* D sits at the median of the voters, but A has the most first choices, so FPTP
elects A with $mu = s_A - s_D = 0.2697 - 0.2565 = 0.0132$. The Borda scores are the row
sums of $pi$ (@sec-needs), and D leads A by $2.7301 - 2.1051 = 0.6250$. D beats every
candidate head to head, most narrowly A: the Condorcet margin is
$pi_(D A) - pi_(A D) = 0.2727$, and Schulze and Black, which elect the Condorcet winner,
use the same margin. Minimax elects D too: its weakest link is $0.2727$, A's is the best
of the others, $-0.2727$, and $mu$ is the lead $0.5455$. IRV eliminates B, C, E and A in
turn:

#align(center, table(
  columns: 8,
  align: (center, right, right, right, right, right, center, right),
  stroke: none,
  table.hline(),
  [round], [A], [B], [C], [D], [E], [out], [gap],
  table.hline(stroke: 0.5pt),
  [1], [0.2697], [0.1164], [0.1303], [0.2565], [0.2271], [B], [0.0138],
  [2], [0.2697], [], [0.2129], [0.2622], [0.2552], [C], [0.0423],
  [3], [0.3636], [], [], [0.3507], [0.2857], [E], [0.0650],
  [4], [0.3636], [], [], [0.6364], [], [A], [0.2727],
  table.hline(),
))

The gap between the two lowest tallies is smallest in the first round, between B and C,
so $mu = 0.0138$. Baldwin drops E, B, C and A, the first with the smallest gap,
$1.6919 - 1.6572 = 0.0347$ to B. Nanson drops B, C and E at once, whose scores are below
the mean $2$, with A closest to it ($mu = 2.1051 - 2 = 0.1051$), and then A. Both elect
the Condorcet winner D.

*At $m_2$.* B, D and E each beat A and C, but among themselves they form a cycle: B beats
D ($pi_(B D) = 0.5206$), D beats E ($0.5062$) and E beats B ($0.5109$). The narrowest
head-to-head results $min_(j != i) (pi_(i j) - pi_(j i))$ are $-0.0218$ for B, $-0.0412$
for D and $-0.0125$ for E (A and C lose by much more). No candidate beats everyone, and
the Condorcet margin is $0.0125$, E's deficit against D.

Schulze has to resolve the cycle. Its links $pi_(i j) - pi_(j i) > 0$ among B, D and E
are B→D $0.0412$, D→E $0.0125$ and E→B $0.0218$. A and C beat none of the three, so the
widest paths between B, D and E stay among them:
$p_(B E) = min(0.0412, 0.0125) = 0.0125$ through D,
$p_(D B) = min(0.0125, 0.0218) = 0.0125$ through E, and
$p_(E D) = min(0.0218, 0.0412) = 0.0218$ through B. E is unbeaten, since
$p_(E B) = 0.0218 > p_(B E)$ and $p_(E D) = 0.0218 > p_(D E) = 0.0125$, so E wins. For
the margin, each other candidate $e$ is beaten by $max_f (p_(f e) - p_(e f))$: B only by E,
by $0.0218 - 0.0125 = 0.0093$; D by B, by $0.0412 - 0.0125 = 0.0287$ (and by E, by
$0.0093$); A and C by $0.5832$ and $0.3175$. The smallest of these, $mu = 0.0093$, says
that B is the candidate closest to being unbeaten.

Minimax elects E as well, whose worst defeat, $0.0125$, is the smallest (B's is $0.0218$,
D's $0.0412$); $mu = -0.0125 - (-0.0218) = 0.0093$ is its lead over B. Black has no
Condorcet winner to elect here and takes the Borda winner D, with
$mu = min(0.0125, 0.0481) = 0.0125$, the smaller of the Condorcet margin and Borda's:
the cycle would end, and D lose its place to a Condorcet winner, if E's deficit against
D closed.

Borda elects D, but Baldwin and Nanson resolve the cycle like Schulze. They drop A and C
first (Baldwin one at a time) and are left with B, D and E, whose Borda scores among the
three are $1.0097$, $0.9856$ and $1.0046$, with mean $1$ (@eq-borda-pairwise). Both drop
D, and then B, which E beats. Nanson's margin $mu = 1.0046 - 1 = 0.0046$ is how close E
came to going out with D; Baldwin's $mu = 1.0046 - 0.9856 = 0.0190$ how close E came to
being the lowest instead of D.

In the first round of IRV, A ($0.1243$) and C ($0.1227$) nearly tie for last place,
so $mu = 0.0016$. E wins whichever of the two goes out first: $m_2$ lies next to a curve
where $mu$ vanishes but the winner does not change. @fig-margins shows $mu$ over the
whole square, with $m_1$ and $m_2$ marked. The FPTP margin vanishes only on the borders.
The IRV margin also vanishes on many curves that are not borders: there two candidates
tie for last place in some round, but the winner does not change.

#figure(
  image("figures/margins.png", width: 100%),
  caption: [The margin $mu$ of FPTP (left) and IRV (right) on the grid of $G = 320$
    points per axis (Beta, `rms`, $D = 0.2$), on a log scale: black is below $10^(-4)$.
    Yellow: the borders between winners, traced as in @sec-tracing. The crosses mark
    $m_1$ (at D) and $m_2$.],
) <fig-margins>

=== Tracing the regions <sec-tracing>

For each winner $c$ let

$ psi_c = cases(mu & "where" c "wins", -mu & "elsewhere") . $

$psi_c$ is continuous, and the region of $c$ is ${psi_c > 0}$. It is traced by marching
squares (`contourpy`) on a grid of $G times G$ points: the centres $(k + 1/2) slash G$
plus both walls, where the medians are clamped to the outermost pixel centres, as the
nodes are. On a grid edge from a point where $c$ wins with margin $alpha$ to a point
where $c'$ wins with margin $beta$, both $psi_c$ and $psi_(c')$ cross zero at
$alpha slash (alpha + beta)$ of the way. So neighbouring regions share their border
points and leave no gap along a border. For example, for IRV at $G = 160$ the
neighbouring grid points $(0.296875, 0.384375)$ and $(0.303125, 0.384375)$ are won by B
with $alpha = 0.00255$ and by C with $beta = 0.00350$. Both the ring of B and the ring
of C get the vertex $0.4215$ of the way along, at $x = 0.299509$ (@fig-polygons, near the
bottom tip of C's island). A gap is left only in a grid cell where three winners meet:
there each of the three regions ends at a straight segment between two edge crossings,
and the small triangle between the three segments belongs to none of them.

The shares come from the interpolant of @sec-interpolation (those of the score methods
are computed at the grid points themselves, @sec-score), so the grid only has
to be fine enough not to miss slivers; its borders do not have the steps of a pixel
image. The
UI uses $G = 160$ while dragging and $G = 320$ once the candidate is dropped. The margins
at grid points next to a change of winner are $O(1 slash G)$, which the tests check.

=== Polygons <sec-polygons>

Every region is a list of polygons, because regions can be in pieces (@sec-pieces) and
can have holes (an island of another winner). Each polygon is an outer ring and its
holes. Can the order of a ring's points tell the region from its complement? In
principle yes: with the convention that the region lies to the left of each edge, a
counter-clockwise triangle is the triangle and a clockwise one is everything else.
GeoJSON and the nonzero fill rule use this convention for holes. But a canvas never
fills a lone clockwise ring as its outside; it fills the inside with winding number
$-1$. So the complement is sent explicitly, as an outer ring (counter-clockwise) with
the triangle as a hole (clockwise). The UI fills with the even-odd rule, which does not
depend on the orientation at all. This matters because flipping $y$ for the canvas
reverses every orientation. Voronoi diagrams need no grid: their cells are the polygons
of @sec-needs.

Each ring is sent as a flat list $[x_0, y_0, x_1, y_1, dots]$ with six decimals. It is
closed: the last point repeats the first, so a ring with $k$ vertices has $2 (k + 1)$
numbers. @tab-polygons lists the IRV regions of @fig-margins as the UI receives them
while dragging ($G = 160$), with the signed (shoelace) area of every ring, which is
positive for a counter-clockwise ring. @fig-polygons draws them.

#figure(
  image("figures/polygons.png", width: 100%),
  caption: [The polygons of @tab-polygons, filled with the even-odd rule as in the UI.
    Solid black: outer rings; dashed red: holes. Right: the frame on the left, with the
    grid points coloured by their winner. The arrows run clockwise along B's two holes:
    the island of C, and at the top right a sliver of C that the grid catches at only
    one point.],
) <fig-polygons>

#figure(
  table(
    columns: 4,
    align: (center, center, right, left),
    stroke: none,
    table.hline(),
    [winner], [polygon], [outer ring], [holes],
    table.hline(stroke: 0.5pt),
    [A], [1], [389~($+0.267367$)], [],
    [], [2], [10~($+0.000155$)], [],
    [], [3], [4~($+0.000020$)], [],
    [], [4], [76~($+0.005068$)], [4~($-0.000001$)],
    [B], [1], [356~($+0.117606$)], [60~($-0.001887$), 4~($-0.000004$)],
    [C], [1], [391~($+0.138644$)], [],
    [], [2], [60~($+0.001887$)], [],
    [], [3], [4~($+0.000004$)], [],
    [D], [1], [443~($+0.233848$)], [],
    [], [2], [4~($+0.000001$)], [],
    [E], [1], [32~($+0.001143$)], [],
    [], [2], [385~($+0.236058$)], [],
    table.hline(),
  ),
  caption: [The IRV regions of the UI's default diagram while dragging (candidates A–E,
    Beta voters, `rms`, $D = 0.2$, $G = 160$): the number of vertices of every ring and,
    in brackets, its signed area.],
) <tab-polygons>

B's region is one polygon with two holes. The larger hole is the island of C next to B.
C's region lists the same 60 vertices again as a polygon of its own, counter-clockwise,
so its area is $+0.001887$ where B's hole has $-0.001887$. The area of B's region is the
sum over its rings, $0.117606 - 0.001887 - 0.000004 = 0.115715$. The rings with 4
vertices are diamonds around a single grid point, the most the grid shows of a sliver
thinner than its step. The smaller hole of B is such a sliver of C, near
$(0.347, 0.472)$. A and E are in several pieces, as IRV regions often are (@sec-pieces).

The signed areas of all rings add up to $0.99991$ instead of $1$. The missing $0.00009$
lies in the grid cells where three winners meet (@sec-tracing). It shrinks with the
grid, to $0.99998$ at $G = 320$.

== Timings

Round trip of one update over HTTP while dragging one candidate ($G = 160$; median of
15 steps, $D = 0.2$, `rms`), which adds a few milliseconds to the compute times of
@sec-compiled. Timings on the test machine, a laptop with performance and efficiency
cores, vary by up to a factor of three between runs.

#align(center, table(
  columns: 7,
  align: (left, right, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [], [candidates], [FPTP], [Borda], [Condorcet], [Schulze], [IRV],
  table.hline(stroke: 0.5pt),
  [Beta], [5], [13 ms], [14 ms], [16 ms], [16 ms], [17 ms],
  [], [8], [10 ms], [21 ms], [29 ms], [26 ms], [75 ms],
  table.hline(),
))

IRV is still the slowest: it needs every cell of the arrangement, a drag changes about
two thirds of their edges, and the shares of about 300 rankings are interpolated to the
grid. All edges of the arrangement lie on the $C (C - 1) slash 2$ bisectors, though. If
the integral of $omega$ along each bisector were tabulated once, as a function of the
position on it, every edge would be a difference of two values, and the cost would grow
with the number of lines (28 for 8 candidates) instead of edges (468). This is not
implemented.

#pagebreak()

= Pixels at the geometric median <ch-geometric>

A Beta pixel is the median of its voters along $x$ and along $y$. That is a median only
in the directions of the axes, which is the root of the distorted Condorcet regions of
@ch-compare. The web UI can draw every election at the *geometric median* of its voters
instead (_Pixel is: the geometric median_), a centre that does not depend on the axes.
This chapter defines it, describes the part of the square that the diagram then covers,
shows what it does to the borders between the candidates, and how it is computed
(`margin/geometric.py`). The numbers and the figure are produced by `docs/figures.py`.

== The map $g$ <sec-g>

The geometric median of the voters $V = (X, Y)$ of a pixel is the point with the smallest
mean distance to them. For the voters with the medians $m = (m_x, m_y)$ along the axes,

$ g(m) = op("arg min", limits: #true)_p d_m (p), quad d_m (p) = E_m |V - p| . $ <eq-geometric>

(In this chapter $g$ is this map, as in the code, and not the density of $Y$.) The voters
do not lie on one line, so $d_m$ is strictly convex and the minimum is unique. In one
dimension the point with the smallest mean distance is the median. In the plane it is a
different point from the pair of medians along the axes, and unlike that pair it turns
with the voters when the plane is rotated.

A diagram of geometric medians shows at the point $p$ the election of the voters with
$g(m) = p$. The voters, their shares, the methods and the margins are those of the pixel
$m$; only the point at which the election is drawn moves from $m$ to $g(m)$. $g$ depends
on the voter model alone ($D$ and the spread rule), never on the candidates.

- *Normal voters* are symmetric about their pixel, so $g(m) = m$ and the diagram is the
  usual one.
- *Symmetry.* For Beta voters $g$ leaves the centre of the square in place and commutes
  with mirroring an axis and with swapping the axes, as the voters do. A median of $1/2$
  along one axis stays $1/2$.
- *Towards the centre.* Everywhere else $g$ moves the pixel towards the centre of the
  square, the more the closer the pixel is to a wall. In the table below it lies between
  the median and the mean of the voters.
- *Both axes at once.* The first coordinate of $g$ depends on $m_y$ too: the same voters
  along $x$ are moved less when $Y$ is crowded against a wall as well. So $g$ cannot be
  found one axis at a time, as the median and the mean can.
- *One to one.* Two different electorates of the model are never drawn at the same
  point. This is checked numerically: the pixel grid moved by $g$ keeps the order of its
  neighbours along both axes and never folds over, for $D = 0.05$, $0.25$ and $0.4$ and
  both spread rules (`tests/test_geometric.py`).

First coordinate of $g$ for the medians $(m, 1/2)$ and $(m, m)$, and the mean of $X$
(`rms`, $D = 0.3$; the last row is the outermost pixel):

#align(center, table(
  columns: 4,
  align: right,
  stroke: none,
  table.hline(),
  [median $m$], [$g$ at $(m, 1/2)$], [$g$ at $(m, m)$], [mean],
  table.hline(stroke: 0.5pt),
  [0.50], [0.500], [0.500], [0.500],
  [0.70], [0.662], [0.665], [0.621],
  [0.90], [0.823], [0.844], [0.750],
  [0.98], [0.896], [0.938], [0.812],
  [$599 slash 600$], [0.930], [0.983], [0.838],
  table.hline(),
))

== The coloured region <sec-g-image>

#figure(
  image("figures/geometric.png", width: 100%),
  caption: [Beta voters (`rms`, $D = 0.3$). Left and middle: the Condorcet winner with
    pixels at the medians along the axes and at the geometric medians. White lines: the
    Voronoi borders (bisectors); black: no Condorcet winner. Right: the lines of equal
    $m_x$ and of equal $m_y$ for the values $1 slash 600$, $0.1$, …, $0.9$,
    $599 slash 600$, moved by $g$; the thick outline belongs to the outermost medians.],
) <fig-geometric>

With pixels at geometric medians the diagram no longer fills the square
(@fig-geometric). The coloured region is exactly the set of points of the unit square
that are the geometric median of some electorate of the model. Each of its points
belongs to exactly one electorate, because $g$ is one to one.

*It is not a part cut out of the usual diagram.* Every election of the usual diagram is
in it and none is missing; each is only moved from its medians along the axes to its
geometric median. The usual diagram has shrunk into the region as a whole, and nothing
was cropped.

*"Of the model" matters.* The region is not a property of the geometric median as such,
but of this family of voters:

- *The shape of the voters:* independent Beta distributions along the axes, with the
  chosen $D$ and spread rule. So the outline changes with the _Deviation_ slider, and
  with the rule.
- *The range of the medians:* from $1 slash 600$ to $1 - 1 slash 600$, the outermost
  pixel centres of the UI's $300$ pixels per axis. Medians closer to a wall would fill
  the corners (under `rms`), but narrow the strip in the middle of a wall only slowly.
- *Other voters* can have their geometric median anywhere in the square. Normal voters
  have it at their pixel and cover the whole square.

The table gives the width of the empty strip, that is the distance of $g$ from the wall
for the outermost medians, in the middle of a wall and at a corner, and the part of the
square that stays empty. The last two columns are the widths if the medians reached to
$10^(-6)$ from the walls.

#align(center, table(
  columns: 7,
  align: (left, right, right, right, right, right, right),
  stroke: none,
  table.hline(),
  [], [], table.cell(colspan: 3, align: center)[medians from $1 slash 600$],
  table.cell(colspan: 2, align: center)[from $10^(-6)$],
  [rule], [$D$], [middle], [corner], [empty], [middle], [corner],
  table.hline(stroke: 0.5pt),
  [`rms`], [0.1], [0.018], [0.007], [6.8 %], [0.007], [0.0001],
  [], [0.2], [0.038], [0.011], [13.4 %], [0.019], [0.0002],
  [], [0.3], [0.070], [0.017], [22.5 %], [0.044], [0.0004],
  [], [0.4], [0.125], [0.032], [35.9 %], [0.094], [0.0012],
  table.hline(stroke: 0.3pt),
  [`mean_abs`], [0.2], [0.064], [0.025], [22.7 %], [0.044], [0.0011],
  [], [0.3], [0.141], [0.087], [45.9 %], [0.120], [0.0270],
  table.hline(),
))

*Why a strip stays empty.* At the minimum of $d_m$ the unit vectors from $p$ towards the
voters cancel,

$ nabla d_m (p) = E_m [(p - V) / (|V - p|)] = 0 . $ <eq-balance>

A median counts voters, while the geometric median balances directions. A pixel next to
the wall $x = 0$ has half of its voters between the wall and $m_x$, which is what holds
the median there. But these voters are spread along the wall in $y$. Seen from a point
near the wall they lie up and down the wall, so their unit vectors nearly cancel and
have almost no component towards the wall. The other half pulls away from the wall with
its full weight. The balance is therefore reached inside the square, and further inside
the further out that half lies. This is the push of @eq-push, which is why `mean_abs`
leaves a wider strip than `rms`. In a corner both coordinates are crowded against their
walls, so many voters sit next to the pixel in every direction and hold $g$ there: the
strip is narrowest in the corners.

No spread rule can fill the square as long as the voters keep a spread at the wall. At a
point $p$ of the wall $x = 0$ the derivative of $d_m$ across the wall is
$-E_m [X slash abs(V - p)]$, which is negative unless all the voters lie on the wall. So
for voters inside the square the geometric median reaches a wall only if their spread
across it vanishes there. Normal voters fill the square because they may leave it.

Nor does a thinner far tail close the strip. The pull counts voters, the spread counts
squared distances, so putting the spread into a few voters on the opposite wall narrows
the strip. This was tried as *ZOIB voters* (a zero-or-one inflated Beta, 5.5 % of a
pixel's voters on the opposite wall at $D = 0.2$): the strip in the middle of a wall
narrowed from $0.038$ to $0.016$. But the voters below $g$ still lie along the wall, and
their pull towards it vanishes as $g$ approaches it, while the far ones keep pulling with
their full weight, so the strip stays. ZOIB voters are not used: they do not make the
diagram easier to interpret. They are there only to fill the picture, nothing about
real voters calls for them, and they bend the borders next to the walls.

== The borders between the candidates <sec-g-borders>

By @ch-compare a majority prefers $c_i$ to $c_j$ where the median of the projected voters
$V dot nu$ lies on $c_i$'s side. That median, the effective centre of the pixel in the
direction $nu$, is shifted from $m dot nu$ towards the mean of the voters, and $g(m)$ is
shifted the same way (@sec-g). Drawing the election at $g(m)$ therefore takes back part
of the shift in every direction at once, and the borders move towards the bisectors
(@fig-geometric).

The table measures how far the pairwise majority borders $pi_(i j) = 1/2$ of all ten
pairs of the candidates A–E lie from their bisectors, at the points where a border
crosses a line of the $300 times 300$ pixel grid: the mean distance and, in brackets,
the largest.

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: none,
  table.hline(),
  [rule], [$D$], [pixels at the medians along the axes], [pixels at the geometric medians],
  table.hline(stroke: 0.5pt),
  [`rms`], [0.2], [0.0165 (0.054)], [0.0106 (0.029)],
  [], [0.3], [0.0411 (0.106)], [0.0255 (0.060)],
  table.hline(stroke: 0.3pt),
  [`mean_abs`], [0.2], [0.0208 (0.089)], [0.0130 (0.054)],
  [], [0.3], [0.0549 (0.223)], [0.0317 (0.132)],
  table.hline(),
))

About 60 % of the mean distance and a little more than half of the largest remain. For
`rms` that is $3.2$ pixels instead of $5.0$ on average at $D = 0.2$, and $7.7$ instead
of $12.3$ at $D = 0.3$. For the border between D and E at $D = 0.25$ the tests check the
same: $0.045$ to $0.075$ from the bisector with pixels at the medians along the axes,
$0.025$ to $0.039$ with pixels at the geometric medians.

What the geometric median does not do:

- *It does not give the Voronoi diagram.* $g(m)$ is not a median in every direction: a
  slanted line through it need not split the voters in half, and a product of skewed
  Beta distributions has no point through which every line does (@ch-compare). The rest
  of the bend is the skew of the voters, and it stays.
- *The cycles stay.* The elections are the same, so every election without a Condorcet
  winner is still there, drawn at another point (black in @fig-geometric).
- *Areas are not comparable.* $g$ compresses the diagram most near the walls, so the
  share of the coloured region that a candidate wins is not its share of the pixels.
- *The spread rule still refers to the medians along the axes.* `rms` fixes
  $sqrt(E(X - m_x)^2)$, not the distance of the voters from $g(m)$. Fixing the spread
  around $g(m)$ instead ($sqrt(E(X - g_x)^2)$ or $E|V - g|$) makes the coloured region
  smaller. And since $g$ depends on both axes, the parameters of $X$ then depend on $m_y$
  too, so the shares are no longer edge integrals of one Beta per column and one per row
  (@sec-edges), and take a much slower computation: two to four times as long per
  diagram while a candidate is dragged, and about $6$~s instead of $0.2$~s to prepare the
  voters of each deviation.

A spread around the geometric median gains little that shows in the diagrams and costs
time, so the UI uses the geometric median only as the point where an election is drawn;
the voters keep the `rms` rule around the medians along the axes.

== Computing $g$ (`margin/geometric.py`) <sec-g-compute>

*The mean distance.* $d_m (p)$ is a sum over the rectangles of the voter grid of
@sec-score. The rectangle $[x_k, x_(k + 1)] times [x_l, x_(l + 1)]$ holds the exact
share $mu_k nu_l$ of the voters, so the infinite density of a Beta with $a < 1$ at a wall
does no harm. Its voters count at their exact mean $(overline(x)_k, overline(y)_l)$ (the
first moment of $Beta(a, b)$ between two lines is $a slash (a + b)$ times the share of
$Beta(a + 1, b)$ between them), at the distance

$ sqrt((overline(x)_k - p_x)^2 + (overline(y)_l - p_y)^2 + s_(k l)^2), quad
  s_(k l)^2 = ((x_(k + 1) - x_k)^2 + (x_(l + 1) - x_l)^2) / 24 , $

where $s_(k l)^2$ is the variance of a uniform distribution over the rectangle along one
axis, averaged over the two axes. For a square cell this is the mean distance of its
voters from $p$ to second order in the size of the cell. It also keeps $d_m$ smooth and
convex: with plain distances the voters next to a corner, half of them in a few cells,
would trap the minimum at the mean of a cell.

*Minimisation.* Newton's method, started at $m$, takes about five steps. A step that
would leave the square is replaced by Weiszfeld's step, the mean of the voters weighted
by one over their distance, which never leaves it.

*Accuracy.* Against a voter grid with cells a quarter as wide, $g$ changes by less than
$5 dot 10^(-5)$ for the narrowest voters of the UI ($D = 0.05$), by $2 dot 10^(-5)$ at
$D = 0.1$ and by $3 dot 10^(-6)$ from $D = 0.25$ on. The geometric median of a million
sampled voters, itself accurate to about $3 dot 10^(-4)$, agrees with it within the
$1.5 dot 10^(-3)$ of the tests.

*Nodes.* $g$ is smooth in $m$, so it is computed at the $N times N$ nodes of
@sec-interpolation and interpolated in $logit m$, as the shares are. A quarter of the
nodes is enough; the others are their mirror images. The interpolant is within
$4 dot 10^(-5)$ of $g$ computed at the point itself. The table of a model is built once,
in about 0.2 s, and kept.

*Drawing (`margin/regions.py`).* The regions are traced as in @sec-tracing, on the grid
moved by $g$: the grid point with the medians $m$ is drawn at $g(m)$, and a crossing is
placed along the moved edge by the same linear interpolation of $psi_c$. The grid points
at and next to a wall share the outermost median and would be drawn at the same point,
so only one of them is kept. The moved grid ends short of the walls, and so do the
regions. The Voronoi diagram is traced on the moved grid as well, with the nearest
candidate as the winner and the difference of the two smallest squared distances as the
margin. That difference is linear along a straight edge, so its borders still lie
exactly on the bisectors.

*Moving between the two diagrams.* When the choice of what a pixel is changes, the UI
moves the diagram instead of jumping. For $0.7$ s it asks for the regions with every grid
point drawn at $q + t (g(m) - q)$, where $q$ is the point at which the usual diagram draws
it ($m$, or the wall for the outermost medians) and $t$ eases from $0$ to $1$ or back
(`shift` of `POST /api/regions`). So every election travels on a straight line between
its two points, and the grid does not fold over on the way (tests). The steps are traced
on the coarser grid, like a drag, at about 15 to 45 ms each for five candidates, and the
last one on the fine grid. Where the system asks for less motion the diagram jumps.

*Hovering (`pixels_at`, `GET /api/geometric`).* To show the voters of a hovered point the
UI needs the inverse of $g$: for the centre of every pixel, the pixel whose voters have
their geometric median nearest to it (a k-d tree over $g$ of all pixels), or none if the
centre lies in the strip, beyond the curve that $g$ follows along the outermost pixels
of a side. The UI gets one quarter of the square per deviation and mirrors it.

*Candidates stay in the coloured region (`outline`).* With the lookup comes the border of
the coloured region, the curve $g$ follows along the outermost pixels, counter-clockwise.
A candidate in the strip would stand where no electorate has its geometric median, so
with pixels at geometric medians the UI keeps every candidate within the border: one
dragged or added beyond it, or left beyond it when the diagram moves to geometric medians
or a larger deviation widens the strip, goes to the closest point of the border. It stays
there when the pixels go back to their medians along the axes.
