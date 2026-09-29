"""Beta CDF and quantile of the node distributions, exact or from tables.

The edge integrals of pixels/beta.py evaluate, for a fixed list of N Beta
distributions (the voters of the interpolation nodes along one axis), the CDF of
every node at every quadrature point: O(N^2) `betainc` calls per edge, which is
nearly all of their cost. The N distributions never change for a given deviation
and spread rule, so their CDFs can be tabulated once.

The CDF is tabulated against s = logit(y), with the exact slope

    dG/ds = g(y) y (1 - y),

which is smooth for every a, b > 0: near a wall G ~ y^a / (a B(a, b)) = e^(a s) / (a B),
an exponential in s even where the density itself is infinite. Cubic Hermite
interpolation on a uniform grid in s is therefore accurate everywhere (step 0.01:
4e-9 for the narrowest voters). Below the table (y < 4e-18) the power law itself is
used, exact to double precision; above it no double y < 1 exists. Evaluating the
table is one cubic per point and node, without exp.

The quantile has the inverse table in logit-logit coordinates, logit(x) against
logit(u), which is asymptotically linear at both ends (x ~ (a B u)^(1/a) near 0), so
it is extended linearly. It is only needed for u in the edge integrals, where
u < 1e-13 or 1 - u < 1e-13 can change a result by at most that much.

The tables are built with scipy and evaluated by numba kernels. The grid in s is the
same for all nodes, so a point's interval and position in it are computed once and
serve every node. The kernels release the GIL but are not parallel themselves:
callers run several edges in threads (a parallel numba kernel called from two
threads at once would abort with numba's default threading layer).
"""

import math
import os

import numpy as np
from numba import njit
from scipy.special import betainc, betaincinv, betaln, log_expit

from yeelab import threads

CDF_RANGE = 40.0  # the CDF table spans logit(y) in [-CDF_RANGE, CDF_RANGE] ...
CDF_STEP = 0.01   # ... with this step
PPF_RANGE = 30.0  # the quantile table spans logit(u) in [-PPF_RANGE, PPF_RANGE] ...
PPF_STEP = 0.02   # ... with this step
QUANTILE_CLAMP = 700.0  # |logit(x)| beyond this (x or 1 - x < 1e-304) is cut off

_kernel = njit(cache=True, nogil=True, error_model="numpy")
_inline = njit(inline="always", error_model="numpy")  # helpers of the kernels


def _broadcast(values, extra):
    """(N,) parameters shaped (N, 1, ..., 1) with `extra` trailing axes."""
    return values.reshape(-1, *(1,) * extra)


class ExactBeta:
    """Beta CDF and quantile of the nodes, straight from scipy."""

    def __init__(self, params):
        self.a, self.b = params[:, 0], params[:, 1]

    def __len__(self):
        return len(self.a)

    def cdf(self, x):
        """F_i(x) for every node i at every x, shape (N, *x.shape)."""
        x = np.asarray(x, dtype=np.float64)
        return betainc(_broadcast(self.a, x.ndim), _broadcast(self.b, x.ndim), x)

    def ppf(self, u):
        """F_i^-1(u[i, ...]): row i of u with node i, shape of u."""
        u = np.asarray(u, dtype=np.float64)
        extra = u.ndim - 1
        return betaincinv(_broadcast(self.a, extra), _broadcast(self.b, extra), u)

    def cdf_sums(self, points, weights):
        """S[r, n] = sum_q weights[r, q] F_n(points[r, q]), shape (R, N)."""
        return np.einsum("rq,nrq->rn", weights, self.cdf(points), optimize=True)

# ---------------------------------------------------------------- Kernels


@_inline
def _cubic(coef, k, t, n):
    """Hermite cubic of function n in interval k at position t (0..1). Scalar
    indexing: an array view per call would cost more than the cubic."""
    return ((coef[k, n, 3] * t + coef[k, n, 2]) * t + coef[k, n, 1]) * t + coef[k, n, 0]


@_inline
def _cdf_values(y, coef, tail_a, tail_scale, values):
    """values[n] = F_n(y) for every node n; y is located in the table once."""
    if y <= 0.0 or y >= 1.0:
        values[:] = 0.0 if y <= 0.0 else 1.0
        return
    log_y = math.log(y)
    s = log_y - math.log1p(-y)
    if s < -CDF_RANGE:  # y < 4e-18: the power law is exact to double precision
        for n in range(coef.shape[1]):
            values[n] = math.exp(tail_a[n] * log_y - tail_scale[n])
        return
    t = (s + CDF_RANGE) / CDF_STEP  # s < 37 for every double y < 1: inside the table
    k = min(int(t), coef.shape[0] - 1)
    t -= k
    for n in range(coef.shape[1]):
        values[n] = _cubic(coef, k, t, n)


@_inline
def _quantile(v, n, coef, left, right):
    """F_n^-1(v)."""
    if v <= 0.0 or v >= 1.0:
        return 0.0 if v <= 0.0 else 1.0
    last = coef.shape[0] - 1
    z = math.log(v) - math.log1p(-v)
    inside = min(max(z, -PPF_RANGE), -PPF_RANGE + PPF_STEP * (last + 1))
    t = (inside + PPF_RANGE) / PPF_STEP
    k = min(int(t), last)
    mu = _cubic(coef, k, t - k, n) + (left[n] if z < inside else right[n]) * (z - inside)
    return 1.0 / (1.0 + math.exp(-mu))


@_kernel
def _cdf_every(points, coef, tail_a, tail_scale):
    """out[p, n] = F_n(points[p]) for every node n."""
    out = np.empty((points.size, coef.shape[1]))
    for p in range(points.size):
        _cdf_values(points[p], coef, tail_a, tail_scale, out[p])
    return out


@_kernel
def _ppf_own(u, coef, left, right):
    """out[n, q] = F_n^-1(u[n, q]): row n with node n."""
    out = np.empty_like(u)
    for n in range(u.shape[0]):
        for q in range(u.shape[1]):
            out[n, q] = _quantile(u[n, q], n, coef, left, right)
    return out


@_inline
def _on_wall(v):
    return abs(v) < 1e-9 or abs(v - 1.0) < 1e-9


@_inline
def _edge_part(xs, ys, xe, ye, out, cdf, tail_a, tail_scale, ppf, left, right, nodes,
               weights, scratch):
    """out += integral of omega over (xs, ys) -> (xe, ye), a segment that is not
    vertical and does not touch both kinds of wall: the forms of
    pixels.beta._edge_integral with the same quadrature, on the tables."""
    size = out.shape[0]
    start, end, values = scratch[0], scratch[1], scratch[2]
    if abs(ys - ye) < 1e-9:  # horizontal: -G(y) (F(xe) - F(xs))
        _cdf_values(xs, cdf, tail_a, tail_scale, start)
        _cdf_values(xe, cdf, tail_a, tail_scale, end)
        _cdf_values(ys, cdf, tail_a, tail_scale, values)
        for i in range(size):
            for j in range(size):
                out[i, j] -= (end[i] - start[i]) * values[j]
        return
    if not (_on_wall(ys) or _on_wall(ye)):
        # u = F_i(x): -int G_j(l(F_i^-1(u))) du
        _cdf_values(xs, cdf, tail_a, tail_scale, start)
        _cdf_values(xe, cdf, tail_a, tail_scale, end)
        slope = (ye - ys) / (xe - xs)
        for i in range(size):
            du = end[i] - start[i]
            for q in range(nodes.size):
                x = _quantile(start[i] + du * 0.5 * (nodes[q] + 1.0), i, ppf, left, right)
                y = min(max(ys + (x - xs) * slope, 0.0), 1.0)
                _cdf_values(y, cdf, tail_a, tail_scale, values)
                w = 0.5 * du * weights[q]
                for j in range(size):
                    out[i, j] -= w * values[j]
        return
    # v = G_j(y): int F_i(l^-1(G_j^-1(v))) dv, then omega = omega' - d(F G)
    _cdf_values(ys, cdf, tail_a, tail_scale, start)
    _cdf_values(ye, cdf, tail_a, tail_scale, end)
    inverse = (xe - xs) / (ye - ys)
    for j in range(size):
        dv = end[j] - start[j]
        for q in range(nodes.size):
            y = _quantile(start[j] + dv * 0.5 * (nodes[q] + 1.0), j, ppf, left, right)
            x = min(max(xs + (y - ys) * inverse, 0.0), 1.0)
            _cdf_values(x, cdf, tail_a, tail_scale, values)
            w = 0.5 * dv * weights[q]
            for i in range(size):
                out[i, j] += w * values[i]
    x_start, x_end = scratch[3], scratch[4]
    _cdf_values(xs, cdf, tail_a, tail_scale, x_start)
    _cdf_values(xe, cdf, tail_a, tail_scale, x_end)
    for i in range(size):
        for j in range(size):
            out[i, j] -= x_end[i] * end[j] - x_start[i] * start[j]


@_kernel
def _edge_terms(segments, out, cdf, tail_a, tail_scale, ppf, left, right, nodes, weights):
    """out[e] += integral of omega over segment e = (xs, ys, xe, ye), each (N, N):
    pixels.beta._edge_integral for many edges at once."""
    scratch = np.empty((5, cdf.shape[1]))
    for e in range(segments.shape[0]):
        xs, ys, xe, ye = segments[e, 0], segments[e, 1], segments[e, 2], segments[e, 3]
        if abs(xs - xe) < 1e-9:  # vertical: dx = 0
            continue
        x_wall, y_wall = _on_wall(xs) or _on_wall(xe), _on_wall(ys) or _on_wall(ye)
        corner = (_on_wall(xs) and _on_wall(ys)) or (_on_wall(xe) and _on_wall(ye))
        if abs(ys - ye) >= 1e-9 and x_wall and y_wall and not corner:
            # from a vertical to a horizontal wall: split, each half touches one kind
            xm, ym = 0.5 * (xs + xe), 0.5 * (ys + ye)
            _edge_part(xs, ys, xm, ym, out[e], cdf, tail_a, tail_scale, ppf, left, right,
                       nodes, weights, scratch)
            _edge_part(xm, ym, xe, ye, out[e], cdf, tail_a, tail_scale, ppf, left, right,
                       nodes, weights, scratch)
        else:
            _edge_part(xs, ys, xe, ye, out[e], cdf, tail_a, tail_scale, ppf, left, right,
                       nodes, weights, scratch)

# ---------------------------------------------------------------- Tables


def _node_tables(a, b):
    """Table values and slopes for the nodes (a, b), each shape (n, 1):
    G(expit(s)) against s, and mu(t) = logit F^-1(expit(t)) against t."""
    log_beta = betaln(a, b)
    # The small one of G and 1 - G is computed directly, the other as its complement.
    # Mirroring (1 - G(y) = G'(1 - y)) is no substitute: 1 - y rounds to 1 below
    # y ~ 1e-16, where G ~ y^a can still be far from 0 for small a.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.arange(-CDF_RANGE, CDF_RANGE + CDF_STEP / 2, CDF_STEP)
        log_y, log_1my = log_expit(s), log_expit(-s)
        G, one_minus_G = betainc(a, b, np.exp(log_y)), betainc(b, a, np.exp(log_1my))
        G = np.where(G < 0.5, G, 1 - one_minus_G)
        G_slope = np.exp(a * log_y + b * log_1my - log_beta)  # g(y) y (1 - y)

        # The same care for x and 1 - x: the quantile of u < 1/2 directly, of u >= 1/2
        # through the mirrored Beta at 1 - u.
        t = np.arange(-PPF_RANGE, PPF_RANGE + PPF_STEP / 2, PPF_STEP)
        log_u, log_1mu = log_expit(t), log_expit(-t)
        low = t < 0
        x = betaincinv(a, b, np.exp(log_u[low]))
        one_minus_x = betaincinv(b, a, np.exp(log_1mu[~low]))
        log_x = np.concatenate([np.log(x), np.log1p(-one_minus_x)], axis=1)
        log_1mx = np.concatenate([np.log1p(-x), np.log(one_minus_x)], axis=1)
        mu = log_x - log_1mx
        # d mu / d t = u (1 - u) / (g(x) x (1 - x))
        mu_slope = np.exp(log_u + log_1mu + log_beta - a * log_x - b * log_1mx)
        # cut off where x or 1 - x is below 1e-304, which also removes infinities where
        # those underflow
        cut = ~(np.abs(mu) < QUANTILE_CLAMP)
        mu = np.clip(np.nan_to_num(mu), -QUANTILE_CLAMP, QUANTILE_CLAMP)
        mu_slope = np.where(cut, 0.0, mu_slope)
    return G, G_slope, mu, mu_slope


def _hermite(values, slopes, step):
    """Cubic Hermite coefficients of N functions per interval, shape (K - 1, N, 4):
    interval-major, so a kernel reads every node's coefficients of one interval at once."""
    d0, d1 = slopes[:, :-1] * step, slopes[:, 1:] * step
    v0, v1 = values[:, :-1], values[:, 1:]
    coef = np.stack([v0, d0, 3 * (v1 - v0) - 2 * d0 - d1, 2 * (v0 - v1) + d0 + d1], axis=-1)
    return np.ascontiguousarray(coef.transpose(1, 0, 2))


class TabulatedBeta:
    """Beta CDF and quantile of the nodes from the tables; same interface as
    ExactBeta, absolute error below 1e-7 (tests/test_beta_tables.py)."""

    def __init__(self, params):
        a, b = params[:, 0:1], params[:, 1:2]
        self._size = len(params)
        # G = y^a / (a B(a, b)) (1 + O(y)) below the table
        self._tail = a[:, 0].copy(), (np.log(a) + betaln(a, b))[:, 0]
        # scipy's betainc / betaincinv release the GIL: build blocks of nodes in parallel
        blocks = np.array_split(np.arange(self._size), min(self._size, os.cpu_count() or 1))
        parts = list(threads.pool.map(lambda rows: _node_tables(a[rows], b[rows]), blocks))
        G, G_slope, mu, mu_slope = (np.concatenate(p) for p in zip(*parts))
        self._cdf = _hermite(G, G_slope, CDF_STEP)
        # quantile extended linearly with its end slopes
        self._ppf = _hermite(mu, mu_slope, PPF_STEP), mu_slope[:, 0].copy(), mu_slope[:, -1].copy()

    def __len__(self):
        return self._size

    def cdf(self, x):
        """F_i(x) for every node i at every x, shape (N, *x.shape)."""
        x = np.asarray(x, dtype=np.float64)
        values = _cdf_every(np.ascontiguousarray(x).ravel(), self._cdf, *self._tail)
        return np.moveaxis(values, -1, 0).reshape(self._size, *x.shape)

    def ppf(self, u):
        """F_i^-1(u[i, ...]): row i of u with node i, shape of u."""
        u = np.asarray(u, dtype=np.float64)
        rows = np.ascontiguousarray(u.reshape(self._size, -1))
        return _ppf_own(rows, *self._ppf).reshape(u.shape)

    def edge_terms(self, segments, nodes, weights):
        """Integrals of omega over the segments (E, 4) = (xs, ys, xe, ye), shape
        (E, N, N): pixels.beta._edge_integral with the Gauss-Legendre `nodes` and
        `weights`, compiled, on chunks of the edges in parallel threads."""
        segments = np.ascontiguousarray(segments, dtype=np.float64).reshape(-1, 4)
        out = np.zeros((len(segments), self._size, self._size))
        chunk = max(1, -(-len(segments) // (os.cpu_count() or 1)))
        threads.in_chunks(lambda a, b: _edge_terms(segments[a:b], out[a:b], self._cdf, *self._tail,
                                                   *self._ppf, nodes, weights), len(segments), chunk)
        return out
