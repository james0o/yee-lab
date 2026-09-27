# Beta-Yee diagram

Yee diagram used to visual voting systems. Basic idea is every point (in unit squre $[0,1] \times [0,1]$) represents election with same voting system. Basic asumtion in original Yee diagrams:
-  Every candidate is point in unit square
-  Every voter is voting for nearest candiste.
-  Voters in point $[x, y] \sim \mathcal{N}((x, y),\Sigma)$
- Winner of each election (point) is then colour in for given candidate.

Yee diagram can be viewed as generalization of [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram).
Using random variable (not constant) and specific voting system.

## Beta distribution
In this repository is changed the distribution from normal to beta.
So voters in point $[x, y] \sim \text{Beta}(\alpha, \beta)$ so that $[x, y]$ is medians of that distribution. Because if candidates exist on unit squrare is logical to expect same constrain also for voters.

Median is used because each voter have same weight and many voting method is just ordinal.

### Parametrization for median
Beta distribution cannot use median it must be parametrized for given median $m$. So then we need to solve this two equstion numericaly (F is CDF, f is PDF):

-  $F(m; \alpha, \beta) = 0.5$
-  $\mathbb{E}[|X - m|]= D= \int_0^1 |m-x|f(x;\alpha, \beta) = \frac{a}{a+b} (1-2F(m; a + 1, b)) $

First one is easy, just describe median property. Second equsion guaranties average absolute distance from median is constant $D$.
The constant is selected by heuristic. If is too small all voting method will generate same diagram, if it is large then result diagram is too random and computation is needed. We choose $D=0.2$

Depending on number of pixel in diagram (in your case $n=1000$) for all $m=\frac{1}{2 n},\frac{1}{2 n} +\frac{1}{n} , \dots, 1-\frac{1}{2 n}$ points is need to calculate $\alpha_i, \beta_i$.

At the end we just create cartesian product of all $\alpha_i, \beta_i$ so we have squre $n\times n$.

### Spread rules
The second equation is one of three spread rules (`--spread` in the CLI, *Beta spread* in the UI). All give the centre pixel the same distribution, $\text{Beta}(a_0, a_0)$ with $\mathbb{E}|X - \tfrac12| = D$, and differ in what stays the same towards the edges:

- `mean_abs` (default, legacy): $\mathbb{E}|X - m| = D$ for every pixel, the equation above. Near the edges it pushes the voters on the far side of the median away, which bends borders.
- `rms`: $\sqrt{\mathbb{E}(X - m)^2}$ equals that of the centre pixel.
- `tapered`: $\alpha + \beta = 2a_0\,(4m(1-m))^{0.2}$. The exponent is the result of an optimisation for straight Condorcet borders; the rule behaves very much like `rms`.

`rms` and `tapered` keep Condorcet cycles without the round edges of `mean_abs`. How they were found is described in chapter 3 of `docs/math.pdf`.

## Normal distribution
`--distribution normal` (and *Voters: normal* in the UI) uses the original model instead, implemented in `normal.py`: voters in point $[x, y] \sim \mathcal{N}((x, y), \sigma^2 I)$, not limited to the unit square, with $\sigma = D\sqrt{\pi/2}$ so that $\mathbb{E}[|X - x|] = D$ as for the beta distribution. Cell probabilities are exact (Owen's T function), no quadrature.

With it, every Condorcet method draws exactly the Voronoi diagram (`ideal`), for any $D$: the distribution is symmetric around $[x, y]$, so every line through $[x, y]$ has half of the voters on each side, and in every pair a majority prefers the candidate closer to $[x, y]$. With the beta distribution this holds only for bisectors parallel to an axis (the median is taken per coordinate). For diagonal bisectors the skew of the distribution decides the majority, so the regions of Condorcet methods are pulled towards the centre of the square and Condorcet cycles appear.
