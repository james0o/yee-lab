#set document(title: "Mathematics of ranking_cells.py")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: underline

#let P = math.upright("P")
#let E = math.upright("E")
#let Beta = math.op("Beta")
#let logit = math.op("logit")
#let expit = math.op("expit")

#align(center)[
  #text(size: 17pt, weight: "bold")[The mathematics of `ranking_cells.py`]
  #v(0.2em)
  #text(size: 11pt)[Exact ranking probabilities of a continuous voter population on every pixel]
]

#v(1em)

= Problem

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

= Voter distribution of a pixel

== Model

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

== The two equations (`_solve_beta`)

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

== Symmetry and continuation (`beta_params_at`)

If $X tilde Beta(a, b)$ then $1 - X tilde Beta(b, a)$. A median $m < 1/2$ is therefore
the mirror of median $1 - m$ with $(a, b)$ swapped, so only the medians
$max(m, 1 - m) >= 1/2$ are solved. They are solved in increasing order, starting from
$(1, 1)$ (the uniform distribution, median $1/2$) and using each solution as the initial
guess for the next median. This continuation keeps the solver on the correct branch
even far from the centre, where $(a, b)$ change quickly.

= Ranking cells (`ranking_cells`)

== Bisectors

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

== Construction

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

= Cell probability as a boundary integral

Computing the double integral @eq-cell-prob directly for every cell and every pixel
pair would be expensive. Green's theorem turns it into a sum of integrals along the
cell's edges, and each edge is shared by two cells.

== Green's theorem

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

== Sharing edges (`compute_ranking_probabilities`)

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

= Edge integrals (`_edge_integral`) <sec-edges>

Let the edge go from $s = (x_s, y_s)$ to $e = (x_e, y_e)$.

== Axis-parallel edges (closed form)

- *Vertical* ($x_s = x_e$): $dif x = 0$, so $integral omega = 0$.
- *Horizontal* ($y_s = y_e$): $G(y)$ is constant, so
  $ integral_s^e omega = -G(y_s) (F(x_e) - F(x_s)). $

These include all edges on the boundary of the square.

== General edges: substitution $u = F(x)$

On the line through $s$ and $e$, $y = ell(x) = y_s + (x - x_s)(y_e - y_s) slash (x_e - x_s)$.
Integrating @eq-omega directly is a bad idea: $f(x)$ is infinite at $0$ or $1$ when
$a < 1$ or $b < 1$, and quadrature would converge slowly. Substituting $u = F(x)$,
$dif u = f(x) dif x$ absorbs the density:

$ integral_s^e omega = -integral_(F(x_s))^(F(x_e)) G(ell(F^(-1)(u))) dif u . $ <eq-u>

The integrand is now bounded by $1$ and smooth unless the edge reaches $y = 0$ or
$y = 1$, where $G$ itself behaves like $y^a$ or $1 - (1-y)^b$ and is not smooth.

== Edges touching $y = 0$ or $y = 1$: substitution $v = G(y)$

For those edges the roles of the axes are swapped. Integrate $omega'$ from
@eq-omega-prime with $v = G(y)$, $dif v = g(y) dif y$, and $x = ell^(-1)(y)$:

$ integral_s^e omega' = integral_(G(y_s))^(G(y_e)) F(ell^(-1)(G^(-1)(v))) dif v , $

and convert back to $omega$ with @eq-switch. Now the endpoint on $y in {0, 1}$ is harmless
because it is mapped to $v in {0, 1}$ and the singular behaviour of $G$ is absorbed by
the substitution, while $x$ stays in the interior of $(0, 1)$ so $F$ is smooth there.

== Edges touching both kinds of boundary

An edge can run from a vertical side of the square ($x in {0,1}$) to a horizontal one
($y in {0,1}$). Neither substitution handles both endpoints, so the edge is split at its
midpoint; each half touches only one kind of boundary and is handled as above.
(An edge ending exactly at a corner of the square is integrated with the $v$ form.)

== Quadrature

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

= Interpolation in the pixel median

With $n$ = `PIXELS` large, $O(Q n^2)$ per edge is still costly. But the cell
probabilities are *smooth* functions of the medians $(m_x, m_y)$: $(a, b)$ depends
smoothly on $m$, and the integrals @eq-cell-prob depend smoothly on $(a, b)$. (The
*winner* of a voting method jumps between pixels, but the probabilities do not.) So they
are computed exactly only on an $N times N$ grid of medians (`NODES`, default $N = 49$)
and interpolated to all $n times n$ pixels. If $N >= n$ no interpolation is used.

== Nodes (`node_medians`)

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

== Barycentric interpolation (`_interpolation_matrix`)

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

= Validation (`monte_carlo_check`)

For selected pixels, $M$ voters (default $2 dot 10^6$) are sampled from the pixel's
$Beta times Beta$ distribution, each is ranked directly by sorting distances, and the
empirical ranking frequencies are compared with the exact probabilities. The maximum
absolute difference should be of the order of the Monte Carlo error
$sqrt(p(1-p) slash M) lt.eq 1/(2 sqrt(M)) approx 3.5 dot 10^(-4)$.

= Summary of the pipeline

+ For each node median solve @eq-median–@eq-mad for $(a, b)$ (with mirror symmetry and
  continuation).
+ Build the arrangement of the $binom(C, 2)$ bisectors in the unit square; each cell
  has one ranking.
+ For every distinct cell edge, compute $integral omega$ for all pairs of node parameters
  (closed form, $u = F(x)$ or $v = G(y)$ substitution, Gauss–Legendre quadrature).
+ Sum signed edge integrals into cell probabilities (Green's theorem).
+ Interpolate from the $N times N$ nodes to the $n times n$ pixels with barycentric
  Chebyshev interpolation in $logit(m)$.
