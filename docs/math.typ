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
that relates to the monotonicity of the method.

The settings that appear in the mathematics, with their defaults:

#align(center, table(
  columns: 4,
  align: (left, center, left, left),
  stroke: none,
  table.hline(),
  [quantity], [symbol], [set by], [default],
  table.hline(stroke: 0.5pt),
  [pixels per axis], [$n$], [`--pixels` (`plot.py`), `PIXELS` (`main.py`)], [400 / 300],
  [deviation], [$D$], [`--deviation`, _Deviation_ slider], [0.2],
  [voter distribution], [], [`--distribution`, _Voters_], [Beta],
  [Beta spread rule], [], [`--spread`, _Beta spread fixed_], [`rms`],
  [interpolation nodes per axis], [$N$], [`--nodes` (`NODES`)], [49],
  [Gauss–Legendre points per edge], [$Q$], [`QUAD_NODES`], [24],
  [exponent of `tapered`], [$tau$], [`TAPER`], [0.2],
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
(`methods.voronoi`, and deviation $0$ in the web UI). It is the reference against which
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
pushed out near a wall, and this is what the three rules differ in.

=== The three rules

*`rms` (default).* The root mean square distance of the voters from the median is the
same as at the centre pixel,

$ sqrt(E(X - m)^2) = s = 1 / (2 sqrt(2 a_0 + 1)), quad E(X - m)^2 = "Var" X + (mu - m)^2, $ <eq-rms>

where $s$ is the standard deviation of $Beta(a_0, a_0)$ ($s = 0.237$ for $D = 0.2$,
$0.337$ for $D = 0.3$). The square weights distant voters more, so near a wall a few
voters far out make up the spread and the rest of the far half stays where it is. It is
the rule with the clearest meaning, and its borders bend only slightly more than those
of the best rule found (@sec-bench).

*`tapered`.* The concentration is prescribed directly,

$ kappa(m) = 2 a_0 (4 m (1 - m))^tau, quad tau = 0.2 . $ <eq-tapered>

Since $mu (1 - mu)$ is the largest variance any distribution on $[0, 1]$ with mean $mu$
can have, $1 slash (kappa + 1)$ is the share of that room which the electorate of the
pixel uses, and $4 m (1 - m)$ measures the room around the median ($1$ at the centre,
$0$ at a wall). The rule lets electorates near a wall use a little more of their room;
at $D = 0.3$: 45 % at the centre, 50 % at $m = 0.9$, 58 % at $m = 0.98$. The exponent
is empirical: it gave the straightest Condorcet borders at equal numbers of cycles in a
benchmark (@sec-bench). The rule behaves very much like `rms`. Only within about
$10^(-5)$ of a wall, which no practical pixel grid reaches, does $kappa$ become so small
that the spread grows again.

*`mean_abs` (legacy).* The mean absolute deviation is $D$ for every median,
$E|X - m| = D$. This was the first rule, and it is kept to reproduce older results.
Because it keeps the mean absolute deviation fixed, @eq-push applies in full: near a
wall the far half of the voters is pushed twice as far out, towards the opposite wall.
At median $0.98$ the share of voters below $0.1$ is back at $0.192$, more than at the
centre, where `rms` has $0.049$ (@fig-densities). This push bends Condorcet borders,
adds cycles and gives IRV a round edge that the other rules do not have (@ch-spread).

#figure(
  image("figures/spread_densities.png", width: 100%),
  caption: [Density of one coordinate of a pixel's voters, $D = 0.3$, for medians
    moving towards the wall at $1$ (grey: the median). All rules agree at the centre.
    Near the wall `mean_abs` moves many voters to the opposite wall; `rms` and
    `tapered` hardly differ.],
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
  [], [`tapered`], [1.163], [0.619], [0.286], [0.340], [0.099],
  [], [`mean_abs`], [1.012], [0.612], [0.300], [0.355], [0.121],
  table.hline(stroke: 0.3pt),
  [0.90], [`rms`], [1.062], [0.750], [0.229], [0.337], [0.051],
  [], [`tapered`], [0.982], [0.743], [0.237], [0.348], [0.059],
  [], [`mean_abs`], [0.554], [0.685], [0.300], [0.430], [0.146],
  table.hline(stroke: 0.3pt),
  [0.98], [`rms`], [0.799], [0.812], [0.186], [0.337], [0.049],
  [], [`tapered`], [0.724], [0.801], [0.197], [0.353], [0.059],
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
method). `tapered` computes $kappa(m)$ from @eq-tapered and needs nothing else.

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

This chapter computes the shares (@eq-share) for Beta voters (`ranking_cells.py`,
@sec-cells to @sec-interpolation) and for normal voters (`normal.py`, @sec-normal).
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

Only the node probabilities are cached on disk (`cache/beta/<spread>/`,
`cache/normal/`), together with the medians, the parameters and the settings they were
made with; interpolation happens on every load.

== Normal voters (`normal.py`) <sec-normal>

`normal.py` computes the same shares for normal voters (@sec-normal-model), with the
same interface.

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
centres $rho$, $t_s$, $t_e$ are outer sums, so each edge costs two `owens_t` evaluations
per pixel; the cells are computed in parallel threads (`owens_t` releases the GIL).
Five candidates take about $0.1$ s on $49 times 49$ nodes.

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

= Voting methods (`methods.py`) <ch-methods>

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
winner as stopping at a majority. For each of the $2^C$ sets $S$ a 0/1 transfer matrix
$T_S$ of shape $R times C$ is precomputed, so a round is one product $P thin T_S$ for
all pixels with the same $S$.

== Borda count (`borda`)

With $"pos"_r (i) in {0, dots, C - 1}$ the position of $i$ in ranking $r$,

$ "score"_i = sum_r P(r) (C - 1 - "pos"_r (i)), $

and the highest score wins.

== Pairwise majorities

Condorcet methods only use the pairwise shares

$ pi_(i j) = sum_(r: i "above" j "in" r) P(r), quad pi_(i j) + pi_(j i) = 1, $ <eq-pairwise>

the share of voters preferring $c_i$ to $c_j$; $c_i$ beats $c_j$ when
$pi_(i j) > 1/2$.

*Condorcet winner* (`condorcet_cycle`). The candidate who beats every other candidate.
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
other spread rules change them is the subject of @ch-spread.

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
same under `tapered` and `mean_abs` (@ch-spread). It fades with less spread: at
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

@sec-spreads defined three spread rules. They agree at the centre pixel and differ only
in how the spread changes towards the walls. This chapter shows what that changes in
the diagrams, explains which property of a rule keeps borders straight, and gives the
evidence for the default `rms`. Unless stated otherwise the numbers use the candidates
A–E, $D = 0.3$ and $300 times 300$ pixels. `docs/figures.py` reproduces the figures and
the numbers on the default candidates; the benchmark over random candidate layouts
(@sec-bench) was run with separate scripts.

== The three rules on the default candidates

#figure(
  image("figures/spread_rules.png", width: 100%),
  caption: [The three spread rules of @sec-spreads on the default candidates,
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
  [Beta, `tapered`], [1.32 %], [14.9 %], [no], [0.086],
  [Beta, `mean_abs`], [2.06 %], [20.2 %], [yes], [0.096],
  [normal], [0], [0], [no], [0.030],
  table.hline(),
))

(Flare of B: rise of the B|E border over the last $0.075$ before the left wall,
@sec-flare.) `rms` and `tapered` give nearly the same diagrams. `mean_abs` has 1.7
times as many cycle pixels, more distorted Condorcet regions and a round IRV edge. The
FPTP flare is almost the same under all three rules: it comes from the crowding of
voters at the wall, which does not depend on the rule.

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
  [`tapered`, $tau = 0.2$], [0.40], [0.40], [0.40], [0.39], [0.37],
  [`tapered`, $tau = 0.25$], [0.40], [0.41], [0.41], [0.41], [0.40],
  [`mean_abs`], [0.41], [0.44], [0.48], [0.54], [0.59],
  table.hline(),
))

There are two ways to fail. With a fixed $a + b$ the far tail thins out near a wall,
the mean catches up with the median and the pull stalls. With a fixed $E|X - m|$ the
far half is pushed away (@eq-push) and the pull accelerates. `rms` and `tapered` lie in
between, close to constant.

*Diagonal borders.* At $45 degree$ both coordinates are skewed at once. Exact border
shapes bend one way for fixed $a + b$ and the other way for fixed $E|X - m|$, confirming
the two failure modes, but the balance point moves with position: near the centre a
smaller $tau$ is straighter, towards the corners $tau approx 0.2$–$0.25$. Near-axis
borders prefer $0.25$, diagonal ones near the centre less, and the benchmark below puts
the compromise at $0.2$. Above $0.25$ both kinds bend the wrong way, which is why the
optimum is sharp. The textbook skewness approximation of the median is off by about a
factor of two at $D = 0.3$, so the exponent has to be found empirically.

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
on layouts of 3–6 candidates drawn uniformly from $[0.1, 0.9]^2$. A first screen on 24
layouts ruled out most rules (bend relative to `rms`): fixed $a + b$ 2.83, `mean_abs`
2.41, RMS of $arcsin sqrt(X)$ 2.00, $tau = 0.1$ 1.92, far-side RMS 1.84, $L_3$ 1.35,
$L_(2.5)$ 1.21, $L_(2.25)$ 1.10. The remaining rules were checked on 30 new layouts,
with 95 % bootstrap intervals over layouts, at three cycle levels (those of `rms` at
$D = 0.25, 0.3, 0.35$):

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: none,
  table.hline(),
  [rule], [$D = 0.25$ level], [$D = 0.3$ level], [$D = 0.35$ level],
  table.hline(stroke: 0.5pt),
  [`tapered`, $tau = 0.2$], [0.66 [0.62, 0.70]], [0.63 [0.59, 0.70]], [0.56 [0.49, 0.65]],
  [`tapered`, $tau = 0.25$], [0.76 [0.67, 0.87]], [0.90 [0.75, 1.07]], [1.03 [0.81, 1.28]],
  [`tapered`, $tau = 0.3$], [1.52], [1.97], [2.14],
  [`tapered`, $tau = 0.4$], [3.55], [4.49], [4.30],
  [`tapered`, $tau = 0.5$], [5.94], [7.25], [6.80],
  [$L_(1.75)$], [1.01 [0.96, 1.05]], [0.92 [0.88, 0.98]], [0.77 [0.73, 0.80]],
  [$L_(1.5)$], [1.14 [1.05, 1.24]], [0.94 [0.83, 1.08]], [0.67 [0.60, 0.77]],
  table.hline(),
))

The whole pairwise borders (visible or not) give the same picture: $tau = 0.2$ at 0.79,
0.79 and 0.52 of `rms`. In absolute terms the gain is small: at $D = 0.3$ visible
borders bend by 1.5 pixels on average with `rms` and 1.0 with `tapered` (90th
percentile 3.1 and 2.0 pixels; the single worst border 4.9 and 7.7 pixels).

== Other ways to fix the spread

- *Median absolute deviation.* For every Beta, $"median"|X - m| < min(m, 1 - m)$: the
  near half lies entirely within that distance. A fixed median absolute deviation is
  therefore impossible within $D$ of every wall.
- *Normal voters clamped to the square* (moved onto the nearest edge when they would
  leave it) have neither the round IRV edge nor a strong flare (0.023), few cycles
  (0.10 %) and Schulze $!=$ Voronoi on 7.9 % of the pixels. But they put point masses
  on the edges, are not a Beta, and their deviation shrinks near the walls.

== Conclusion

- *`rms`* is the default. It fixes a quantity with a clear meaning, pushes the far half
  of the voters out only by $sqrt(2)$ near a wall, has no round IRV edge, and its
  borders bend only slightly more than those of the best rule found (1.5 against 1.0
  pixels on average).
- *`tapered`* keeps Condorcet borders a little straighter at the same number of cycles
  and otherwise behaves like `rms`; its exponent is empirical.
- *`mean_abs`* is kept to reproduce results made with it (caches and plots are kept
  apart per rule, `cache/beta/<spread>/`, `plots/beta/<spread>/`). By @eq-push it
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
  [`tapered`], [0.885], [0.013], [0.884], [0.036],
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
