#set document(title: "Mathematics of Beta-Yee diagrams")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 15pt)
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: underline

#let P = math.upright("P")
#let E = math.upright("E")
#let Beta = math.op("Beta")
#let logit = math.op("logit")
#let expit = math.op("expit")

#align(center)[
  #text(size: 17pt, weight: "bold")[The mathematics of Beta-Yee diagrams]
  #v(0.2em)
  #text(size: 11pt)[Exact ranking probabilities of Beta and normal voters, and how the two models differ]
]

#v(1em)

#outline(indent: auto, depth: 3)

#pagebreak()

= Generating the ranking probabilities <ch-generate>

== Problem

The political space is the unit square $[0,1]^2$. There are $C$ candidates at fixed
points $c_1, dots, c_C in [0,1]^2$. The square is divided into $n times n$ pixels
(`PIXELS`), and each pixel represents a population of voters, not a single voter.

A voter at position $p$ ranks the candidates by Euclidean distance: the closest
candidate is ranked first, the next closest second, and so on. For each pixel we want

$ P_(k l)(r) = "share of the voters of pixel" (k, l) "whose ranking is" r $

for every strict ranking $r$ that can occur. These numbers feed the voting methods
(Borda, Schulze, …) in `methods.py`. The module computes them *without sampling or
discretising the voters*: the only approximations are Gauss–Legendre quadrature and
polynomial interpolation, both of which converge exponentially fast.

== Voter distribution of a pixel

=== Model

The voters of pixel $(k, l)$ have positions $(X, Y)$ with $X$ and $Y$ independent,

$ X tilde Beta(a_k, b_k), quad Y tilde Beta(a_l, b_l). $

The Beta family is used because it lives on $[0,1]$, so every voter stays inside the
political space. Parameters $(a, b)$ are chosen per coordinate so that

+ the *median* is the pixel centre $m = (k + 1/2) slash n$ (`pixel_medians`), and
+ the *mean absolute deviation from the median* is a fixed constant $d$ (`DEVIATION`):
  $E|X - m| = d$.

Condition 2 keeps the "spread" of every pixel the same, independently of where the
pixel lies. Near the edges of the square this forces $a < 1$ or $b < 1$, so the density
is singular (infinite) at $0$ or $1$; much of the later numerics is shaped by this.

We write $f, F$ for the pdf and CDF of $X$ and $g, G$ for those of $Y$. The CDF of
$Beta(a, b)$ is the regularised incomplete beta function $I_x (a, b)$
(`scipy.special.betainc`), its inverse is `betaincinv`.

=== The two equations (`_solve_beta`)

The median condition is simply

$ I_m (a, b) = 1/2. $ <eq-median>

For the absolute deviation, split at the median and use $F(m) = 1/2$:

$
E|X - m| &= E[X - m] - 2 E[(X - m) bb(1){X < m}] \
         &= mu - m - 2 E[X bb(1){X < m}] + 2 m F(m) \
         &= mu - 2 E[X bb(1){X < m}],
$

where $mu = a/(a+b)$. Since $x dot x^(a-1)(1-x)^(b-1) slash B(a,b) = mu dot$
(density of $Beta(a+1, b)$),

$ E[X bb(1){X < m}] = mu I_m (a+1, b), $

and therefore

$ E|X - m| = a/(a+b) (1 - 2 I_m (a+1, b)) = d. $ <eq-mad>

@eq-median and @eq-mad are two equations in the two unknowns $(a, b)$ and are solved with
Levenberg–Marquardt (`scipy.optimize.root(method="lm")`).

=== Symmetry and continuation (`beta_params_at`)

If $X tilde Beta(a, b)$ then $1 - X tilde Beta(b, a)$. A median $m < 1/2$ is therefore
the mirror of median $1 - m$ with $(a, b)$ swapped, so only the medians
$max(m, 1 - m) >= 1/2$ are solved. They are solved in increasing order, starting from
$(1, 1)$ (the uniform distribution, median $1/2$) and using each solution as the initial
guess for the next median. This continuation keeps the solver on the correct branch
even far from the centre, where $(a, b)$ change quickly.

== Ranking cells (`ranking_cells`)

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
Points on a bisector have a tie, but lines have zero area and so zero probability, and
ties are ignored.

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

@fig-cells shows the arrangement for the candidates in `const.py`.

// Generated from ranking_cells(CANDIDATES) with the candidates of const.py.
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

== Cell probability as a boundary integral

Computing the double integral @eq-cell-prob directly for every cell and every pixel
pair would be expensive. Green's theorem turns it into a sum of integrals along the
cell's edges, and each edge is shared by two cells.

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
$a < 1$ or $b < 1$, and quadrature would converge slowly. Substituting $u = F(x)$,
$dif u = f(x) dif x$ absorbs the density:

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

With $n$ = `PIXELS` large, $O(Q n^2)$ per edge is still costly. But the cell
probabilities are *smooth* functions of the medians $(m_x, m_y)$: $(a, b)$ depends
smoothly on $m$, and the integrals @eq-cell-prob depend smoothly on $(a, b)$. (The
*winner* of a voting method jumps between pixels, but the probabilities do not.) So they
are computed exactly only on an $N times N$ grid of medians (`NODES`, default $N = 49$)
and interpolated to all $n times n$ pixels. If $N >= n$ no interpolation is used.

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
cluster at the ends of the interval, which further concentrates nodes near the edges
of the square and avoids the Runge phenomenon.

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
probabilities of each pixel still sum to one up to rounding.

Only the node probabilities are cached on disk (`generate_ranking_probabilities`),
together with the medians and parameters; interpolation happens on every load.

== Validation (`monte_carlo_check`)

For selected pixels, $M$ voters (default $2 dot 10^6$) are sampled from the pixel's
$Beta times Beta$ distribution, each is ranked directly by sorting distances, and the
empirical ranking frequencies are compared with the exact probabilities. The maximum
absolute difference should be of the order of the Monte Carlo error
$sqrt(p(1-p) slash M) lt.eq 1/(2 sqrt(M)) approx 3.5 dot 10^(-4)$.

== Summary of the pipeline

+ For each node median solve @eq-median–@eq-mad for $(a, b)$ (with mirror symmetry and
  continuation).
+ Build the arrangement of the $binom(C, 2)$ bisectors in the unit square; each cell
  has one ranking.
+ For every distinct cell edge, compute $integral omega$ for all pairs of node parameters
  (closed form, $u = F(x)$ or $v = G(y)$ substitution, Gauss–Legendre quadrature).
+ Sum signed edge integrals into cell probabilities (Green's theorem).
+ Interpolate from the $N times N$ nodes to the $n times n$ pixels with barycentric
  Chebyshev interpolation in $logit(m)$.

== Normal voters (`normal.py`) <sec-normal>

`normal.py` computes the same ranking probabilities for the original Yee model, in
which the voters of a pixel are normally distributed. Chapter 2 compares the two models;
this section only describes the computation.

=== Model

The voters of the pixel with centre $m = ((k + 1/2) slash n, (l + 1/2) slash n)$ are

$ (X, Y) tilde cal(N)(m, sigma^2 I), $

isotropic and *not* restricted to the unit square. To make the two models comparable,
$sigma$ is set from the same constant $d$ (`DEVIATION`): for $X tilde cal(N)(m_x, sigma^2)$,
$E|X - m_x| = sigma sqrt(2 slash pi)$, so

$ sigma = d sqrt(pi / 2) quad (d = 0.3 => sigma approx 0.376), $ <eq-sigma>

(`sigma_from_deviation`). Both models then have the same mean absolute deviation from
the median in each coordinate; for the normal the median is also the mean.

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
the edge line into two right triangles with legs $d$ (from $m$ to $q$) and $t$ (from
$q$ along the line). The isotropic normal is invariant under rotations, so in units of
$sigma$ a right triangle is ${0 <= z_1 <= h, 0 <= z_2 <= a z_1}$ with $h = d slash sigma$,
$a = t slash d$, for standard normal $(Z_1, Z_2)$. The wedge
${z_1 > 0, 0 < z_2 < a z_1}$ holds $arctan(a) slash (2 pi)$ of the mass, and the part of
it beyond $z_1 = h$ is by definition Owen's T function,

$
T(h, a) = 1/(2 pi) integral_0^a exp(-h^2 (1 + x^2) slash 2) / (1 + x^2) dif x =
P(Z_1 > h, 0 < Z_2 < a Z_1) quad (h, a >= 0).
$

Therefore the right triangle has probability

$ R(d, t) = arctan(t slash d) / (2 pi) - T(d slash sigma, t slash d). $ <eq-owen>

Both terms are odd in $t$; $T$ is even in $h$ and odd in $a$, so @eq-owen with a
*signed* $d$ is the signed probability. For the edge $s -> e$ with unit direction $u$
and left normal $u^perp = (-u_y, u_x)$ let

$
d = (m - s) dot u^perp, quad t_s = (s - m) dot u, quad t_e = t_s + |e - s|,
$

where $d > 0$ when $m$ lies left of the edge, i.e. inside a CCW cell. Then

$ P_plus.minus (m, s, e) = R(d, t_e) - R(d, t_s), $

and $0$ when $d = 0$ (the triangle degenerates). For the $N times N$ grid of pixel
centres $d$, $t_s$, $t_e$ are outer sums, so each edge costs two `owens_t` evaluations
per pixel; the cells are computed in parallel threads (`owens_t` releases the GIL).
Five candidates take about $0.1$ s on $49 times 49$ nodes. The pairwise shares match
their closed form (@eq-normal-pairwise in Chapter 2) to $10^(-14)$ in the tests.

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

#pagebreak()

= Beta versus normal voters <ch-compare>

The original Yee diagrams put normally distributed voters around every point; this
project replaced them by Beta distributed voters (Chapter 1). With the same candidates
the two models give visibly different diagrams, and the difference is largest where one
would expect it least: for Condorcet methods. On
#link("https://votingmethods.net/yee/")[votingmethods.net/yee] (normal voters) the
River method splits the square into one clean region per candidate, while Schulze on
the Beta model has distorted regions and even pixels without a Condorcet winner
(@fig-compare). This chapter explains why.

All numbers below use the candidates A–E of `const.py` (@fig-cells), $d = 0.3$
(`DEVIATION`), $300 times 300$ pixels and $49 times 49$ nodes; they are produced by
`docs/figures.py`.

#figure(
  image("figures/compare.png", width: 100%),
  caption: [Beta (top) and normal voters (bottom) with the same $d = 0.3$. Normal
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
  [spread], [$E|X - m_x| = d$], [$E|X - m_x| = d$, i.e. @eq-sigma],
  [symmetric about $m$], [only for $m = (1/2, 1/2)$], [always],
  table.hline(),
))

votingmethods.net uses a standard deviation of $0.4$ by default ($d approx 0.319$ here)
and samples only 140 voters per pixel, so its diagrams also carry Monte Carlo noise
(ragged IRV borders, small islands). The probabilities here are exact.

== Pairwise majorities

By @eq-bisector a voter at $p$ prefers $c_i$ to $c_j$ exactly when
$p dot nu < omega$ with $nu = c_j - c_i$ and $omega = (|c_j|^2 - |c_i|^2) slash 2$.
The share of the voters of pixel $m$ who prefer $c_i$ to $c_j$ is therefore

$ pi_(i j)(m) = P_m (X dot nu < omega), $

and it depends only on the one-dimensional projection $X dot nu$ of the voters onto the
direction from $c_i$ to $c_j$. Every Condorcet method only looks at these numbers.

=== Normal voters: the majority is decided by distance

The projection of an isotropic normal is normal,
$X dot nu tilde cal(N)(m dot nu, sigma^2 |nu|^2)$, so

$
pi_(i j)(m) = Phi(delta_(i j)(m) / sigma), quad
delta_(i j)(m) = (omega - m dot nu) / (|nu|),
$ <eq-normal-pairwise>

where $delta_(i j)(m)$ is the signed distance of $m$ from the bisector, positive on the
side of $c_i$. A majority prefers $c_i$ to $c_j$ if and only if $delta_(i j)(m) > 0$, that
is, if and only if $|m - c_i| < |m - c_j|$. Consequently:

+ The majority relation of every pixel is the order of the candidates by distance from
  $m$. It is transitive: *there are no Condorcet cycles.*
+ The Condorcet winner always exists and is the candidate nearest to $m$, so *every*
  Condorcet method (Schulze, River, Ranked Pairs, Minimax, …) draws exactly the Voronoi
  diagram of the candidates, `methods.ideal`.
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
line through $m$ perpendicular to $nu$ splits the voters exactly in half
(`test_median_splits_voters_in_half`). For an oblique $nu$ this fails: the median of
$nu_x X + nu_y Y$ is in general *not* $nu_x m_x + nu_y m_y$, because medians are not
additive for skewed variables. For every direction $nu$ there is a different
"effective centre" of the pixel, shifted from $m$ towards the mean of the voters.
Consequently:

+ The Condorcet winner is no longer the candidate nearest to $m$, so Condorcet regions
  are distorted.
+ Different pairs of candidates are compared from different effective centres, so the
  majority relation need not be transitive: *Condorcet cycles appear* (black in
  @fig-compare). If some point were a median in every direction (a "total median",
  Davis, DeGroot and Hinich, 1972), the argument of the normal case would apply with
  that point in place of $m$; a product of skewed Beta marginals generally has none.

=== How skewed the Beta is at $d = 0.3$ <sec-skew>

The uniform distribution on $[0, 1]$ has $E|X - 1/2| = 1/4$, and $Beta(1/2, 1/2)$ has
$1 slash pi approx 0.318$. So $d = 0.3$ is more spread than uniform and the Beta is
U-shaped ($a, b < 1$, mass piled at both ends) *even at the centre*:

#align(center, table(
  columns: 6,
  align: right,
  stroke: none,
  table.hline(),
  [median], [$a$], [$b$], [mean], [$P(X < 0.1)$], [$P(X > 0.9)$],
  table.hline(stroke: 0.5pt),
  [0.50], [0.602], [0.602], [0.500], [0.18], [0.18],
  [0.70], [0.619], [0.393], [0.612], [0.12], [0.32],
  [0.90], [0.380], [0.175], [0.685], [0.15], [0.50],
  [0.98], [0.215], [0.093], [0.698], [0.19], [0.58],
  table.hline(),
))

Near the edges this is forced by the constraints. With median $m$ close to $1$ the upper
half of the voters lies in $[m, 1]$ and contributes at most $(1 - m) slash 2$ to
$E|X - m|$, so the lower half has to supply the rest:

$ E[m - X | X < m] >= 2d - (1 - m), $

which for $m = 0.98$ means that the lower half of the voters lies on average at least
$0.58$ below the median. Moving the median towards the edge *pushes the other half of
the voters further away* (@sec-round-edges uses this twice).

=== Example: the pull towards the centre

The pixels below lie on A's side of the bisector of A $= (0.6, 0.35)$ and the centre
candidate D $= (0.5, 0.5)$; $delta$ is their distance from it. Share of voters preferring A:

#align(center, table(
  columns: 5,
  align: right,
  stroke: none,
  table.hline(),
  [pixel $m$], [$delta$], [normal], [Beta], [Beta mean],
  table.hline(stroke: 0.5pt),
  [(0.95, 0.55)], [0.118], [0.623], [*0.477*], [(0.694, 0.530)],
  [(0.90, 0.60)], [0.049], [0.551], [*0.441*], [(0.685, 0.559)],
  [(0.80, 0.45)], [0.118], [0.623], [0.523], [(0.655, 0.470)],
  table.hline(),
))

In the first two pixels the median voter is closer to A, yet D wins the pairwise
contest: the long tails of the Beta point towards the middle of the square, and so does
the effective centre.

=== The whole diagram

Leaving out the 357 pixels whose centre lies exactly on a bisector (where both
candidates are correct winners):

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: none,
  table.hline(),
  [], [Schulze $!=$ Voronoi], [no Condorcet winner], [area of D (Voronoi: 0.208)],
  table.hline(stroke: 0.5pt),
  [Beta], [18 125 pixels (20.2 %)], [1 844 pixels (2.1 %)], [0.351],
  [normal], [0], [0], [0.208],
  table.hline(),
))

The centre candidate D gains most: the Beta electorates of the whole upper right part
of the square are pulled towards the centre, where D is.

== Round edges <sec-round-edges>

Under normal voters the borders of Condorcet methods are bisectors, i.e. straight
lines. The borders of FPTP and IRV are curved in both models, and the Beta model adds
bends and hooks near the edges of the square. The key is to look at a first-choice
share as a function of the pixel: it is the Voronoi cell of the candidate, *blurred*.

=== First-choice shares are blurred Voronoi cells

The FPTP winner changes where two first-choice shares are equal,

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
piled against the walls, @fig-voters-irv and @fig-voters-wall), so the blur itself
varies across the square. This is the source of the bends near the edges below.

=== Straight versus round: a proof by blurring a corner

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
    quadrant cell of the lower-left candidate. Middle: its share @eq-blur-corner with
    contour lines. Right: FPTP. At the vertex the shares are $1/4$, $3/8$, $3/8$, so
    the quadrant candidate loses its corner, along a round front.],
) <fig-blur-corner>

=== Seven candidates

@fig-seven shows FPTP for seven candidates (B removed, three added on the left). With a
small blur ($d = 0.1$) the diagram is the Voronoi diagram with rounded corners at the
vertices. With $d = 0.3$ the blur is wider than several cells:

- the centre candidate $(0.5, 0.5)$ has a thin band as its cell. Blurred, a thin band
  stays dim everywhere, so it wins *nowhere*, not even at its own position;
- borders appear between candidates whose cells do not even touch, e.g. $(0.3, 0.7)$
  and $(0.6, 0.35)$. Such a border has no bisector to follow at all: it is purely a
  contour line of two blurred cells, and it is curved.

The Beta diagram shows the same effects, plus the distortions of a blur that changes
from pixel to pixel (the bowl-shaped border under $(0.3, 0.7)$).

#figure(
  image("figures/seven_fptp.png", width: 100%),
  caption: [FPTP for seven candidates; dashed: Voronoi borders. From left: no blur
    (Voronoi), normal voters with $d = 0.1$ and $d = 0.3$, Beta voters with $d = 0.3$.],
) <fig-seven>

=== The round IRV edge

#figure(
  image("figures/irv_round.png", width: 62%),
  caption: [Beta IRV (faded) with the three round 3 tie curves among A, D, E. The red
    curve $s_A = s_D$ is the round edge of D's region; all three curves meet at the
    kink near $(0.71, 0.75)$. Gray: the bisectors D|A and D|E.],
) <fig-irv-round>

In the upper right part B and C are eliminated first. In round 3 A, D and E remain,
and D's cell $V_D^({A, D, E})$ is the diagonal band between the bisectors D|A and D|E
that widens towards the corner $(1, 1)$; A has the bottom right, E the top left.
Whoever of A and D has fewer votes is eliminated:

- D out: the final is A against E, and E wins;
- A out: the final is D against E, and D wins (the pull towards the centre again).

So the edge of D's region is the tie curve $s_A = s_D$ of round 3 (@fig-irv-round). It
bends back on itself because along the column $x = 0.55$ the two shares are not
monotonic:

#align(center, table(
  columns: 6,
  align: right,
  stroke: none,
  table.hline(),
  [$y$], [winner], [$s_A$], [$s_D$], [$s_E$], [$P(Y < 0.2)$],
  table.hline(stroke: 0.5pt),
  [0.678], [E], [0.327], [0.284], [0.389], [0.195],
  [0.738], [E], [0.303], [0.286], [0.411], [0.185],
  [0.798], [E], [0.285], [0.284], [0.431], [0.182],
  [0.858], [D], [0.274], [0.279], [0.447], [0.187],
  [0.918], [D], [0.270], [0.270], [0.460], [0.199],
  [0.978], [E], [0.274], [0.254], [0.472], [0.226],
  table.hline(),
))

- Low in the column many voters are still below the D|A bisector, so A is stronger.
- Going up, voters leave A's cell into D's band: $s_A$ falls, $s_D$ rises.
- Higher up, voters cross the D|E bisector into E's part of the top edge, and $s_D$
  falls again, while $s_A$ *rises* again: with the median of $Y$ approaching $1$ the
  fixed deviation pushes the lower half of the voters down (@sec-skew), $P(Y < 0.2)$ grows from $0.182$ to $0.226$, and those voters are in A's cell.

D is ahead of A only in a middle band, so the tie curve closes around it. At the kink
the three shares are equal; below it the edge follows the tie $s_D = s_E$ (dashed).

@fig-voters-irv shows this directly: sampled voters of four pixels of the column,
coloured by their first choice among A, D and E. With Beta voters the cloud does not
move, it is squeezed against the walls: a dense row forms at the top wall, mostly E
(purple), and another at the bottom wall, which is A's (blue). So $s_A$ stops falling and
rises again ($0.317, 0.285, 0.274, 0.275$) while D's share drops ($0.285$ to $0.253$).
With normal voters the whole cloud simply moves up: $s_A$ falls and $s_E$ rises
monotonically.

#figure(
  image("figures/voters_irv.png", width: 100%),
  caption: [Voters of the pixels $(0.55, y)$ coloured by their first choice among A
    (blue), D (red) and E (purple), with the bisectors D|A and D|E; $+$: pixel median
    (Beta) or mean (normal). Titles: the exact round 3 shares.],
) <fig-voters-irv>

With normal voters D also beats A in round 3 around $y approx 0.74$, but a little higher
D loses the final against E, because the pixel is closer to E; the edge there is the
straight D|E bisector and the curved round 3 tie is not visible:

#align(center, table(
  columns: 5,
  align: right,
  stroke: none,
  table.hline(),
  [$y$ (normal)], [winner], [$s_A$], [$s_D$], [$s_E$],
  table.hline(stroke: 0.5pt),
  [0.678], [E], [0.288], [0.266], [0.445],
  [0.738], [D], [0.245], [0.265], [0.490],
  [0.798], [E], [0.205], [0.260], [0.535],
  [0.858], [E], [0.169], [0.251], [0.579],
  table.hline(),
))

=== Hooks at the edge of the square (FPTP)

#figure(
  image("figures/fptp_edge.png", width: 85%),
  caption: [FPTP. With Beta voters the region of B flares out in the last few
    hundredths before the left edge; with normal voters it widens smoothly.],
) <fig-fptp-edge>

B $= (0.25, 0.4)$ is the leftmost candidate and its Voronoi cell widens towards the left
edge: the bisector B|C ($y = x + 0.05$) falls as $x$ decreases, and at $x = 0$ the cell
spans $0.05 < y < 0.596$. The more voters sit close to $x = 0$, the more of them fall into
the wide part of B's cell. The Beta squeezes the voters against the wall as the median
approaches it — the same effect of the fixed deviation as above:

#align(center, table(
  columns: 6,
  align: right,
  stroke: none,
  table.hline(),
  [median $x$], [$P(X < 0.02)$], [Beta B|E], [Beta C|B], [normal B|E], [normal C|B],
  table.hline(stroke: 0.5pt),
  [0.200], [0.259], [–], [–], [0.455], [0.238],
  [0.080], [0.400], [0.362], [0.278], [0.505], [0.125],
  [0.030], [0.478], [0.422], [0.245], [0.525], [0.075],
  [0.005], [0.547], [0.458], [0.218], [0.535], [0.048],
  table.hline(),
))

(heights $y$ of the borders of B's region in the column; "–": B wins nowhere). With Beta
voters B's share jumps within the last few hundredths of the square and the borders hook
(B|E bends up, C|B down). With normal voters the widening is smooth: the voters just
continue beyond $x = 0$ into the wider part of the cell, and nothing piles up at the wall.

@fig-voters-wall shows why. Between $x = 0.2$ and $x = 0.02$ the Beta voters of the
pixel collapse onto the wall, into the wide part of B's cell: B's share grows from
$0.22$ to $0.29$ while C's falls from $0.19$ to $0.16$, most of it in the last few
hundredths. The normal cloud keeps its shape and only moves (B's share grows smoothly,
$0.32$, $0.40$, $0.44$), so its borders have no hook.

#figure(
  image("figures/voters_wall.png", width: 80%),
  caption: [Voters of the pixels $(x, 0.4)$ approaching the left wall, coloured by their
    first choice (FPTP), with the Voronoi borders; titles: shares of B, C and E.],
) <fig-voters-wall>

In short: *curved* FPTP and IRV borders are generic (ties between masses of polygons),
while the *hooks and bends near the edges* are a property of the Beta model with a
fixed deviation, the same property that distorts the Condorcet diagrams.

== Summary

- Normal voters are symmetric about the pixel centre, so pairwise majorities follow
  distance: no cycles, and every Condorcet method draws the Voronoi diagram, whatever
  $sigma$.
- Beta voters have the pixel centre as the median of each coordinate only. Majorities
  between candidates on a diagonal are decided by the skew of the distribution, which
  pulls the effective centre towards the middle of the square: Condorcet regions are
  distorted (20 % of the pixels for $d = 0.3$) and cycles appear.
- A first-choice share is the candidate's Voronoi cell blurred by the voter
  distribution. Blurring keeps half-planes straight but rounds corners (radius of order
  $sigma$), so FPTP and IRV borders are curved in both models.
- At $d = 0.3$ the Beta is U-shaped everywhere, and near the edges the fixed deviation
  pushes the other half of the voters away: the blur changes shape across the square.
  This produces the round IRV edge and the FPTP hooks at the walls of the square.
