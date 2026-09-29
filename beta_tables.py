"""Beta CDF and quantile of the node distributions, exact or from tables.

The edge integrals of ranking_cells.py evaluate, for a fixed list of N Beta
distributions (the voters of the interpolation nodes along one axis), the CDF of
every node at every quadrature point: O(N^2) `betainc` calls per edge, which is
nearly all of their cost. The N distributions never change for a given deviation
and spread rule, so their CDFs can be tabulated once.

Tables are kept in logit-logit coordinates, s = logit(y) and lambda = logit(G(y)):

    lambda'(s) = g(y) y (1 - y) / (G (1 - G)),

which is smooth for every a, b > 0. Near the walls G ~ y^a / (a B(a, b)), so lambda is
asymptotically linear with slope a as s -> -inf (slope b as s -> +inf), even where
the density itself is infinite. Cubic Hermite interpolation with the exact slope is
therefore accurate on a uniform grid in s. Below the table (y < 4e-18) the power
law itself is used, exact to double precision; above it no double y < 1 exists.
The quantile has the inverse table, logit(x) against logit(u), with the reciprocal
slope, extended linearly. It is only needed for u in the edge integrals, where
u < 1e-13 or 1 - u < 1e-13 can change a result by at most that much.

The grid in s is the same for all nodes, so a point's interval and position in it
are computed once and serve every node.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import betainc, betaincinv, betaln, log_expit, logit

CDF_RANGE = 40.0  # forward table spans logit(y) in [-CDF_RANGE, CDF_RANGE]
PPF_RANGE = 30.0  # inverse table spans logit(u) in [-PPF_RANGE, PPF_RANGE]
STEP = 0.02       # grid step in logit coordinates
LOGIT_CLAMP = 50.0  # |logit(G)| beyond this (G < 2e-22 or 1 - G < 2e-22) is cut off
QUANTILE_CLAMP = 700.0  # |logit(x)| beyond this (x or 1 - x < 1e-304) is cut off
FAR = 1e4         # logit arguments are clipped here, so 0 and 1 stay finite


def _clamp(values, slopes, limit):
    """Values cut off at +-limit with flat slopes there. The cut parts are beyond
    anything that matters: G or 1 - G below 2e-22, or a quantile below 1e-304; this
    also removes the infinities where those underflow."""
    cut = ~(np.abs(values) < limit)
    return np.clip(np.nan_to_num(values), -limit, limit), np.where(cut, 0.0, slopes)


def _expit(values):
    """1 / (1 + exp(-values)), in place; several times faster than scipy's expit."""
    with np.errstate(over="ignore"):
        values = np.exp(np.negative(values, out=values), out=values)
    values += 1.0
    return np.reciprocal(values, out=values)


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


class _Table:
    """Cubic Hermite interpolant of N functions on one uniform grid, extended
    linearly past both ends. Coefficients per interval, shape (4, N, K - 1)."""

    def __init__(self, start, values, slopes):
        self.start, self.stop = start, start + STEP * (values.shape[1] - 1)
        d0, d1 = slopes[:, :-1] * STEP, slopes[:, 1:] * STEP
        v0, v1 = values[:, :-1], values[:, 1:]
        self.coef = np.stack([v0, d0, 3 * (v1 - v0) - 2 * d0 - d1, 2 * (v0 - v1) + d0 + d1])
        self.end_slopes = slopes[:, 0], slopes[:, -1]

    def _evaluate(self, z, coef, left, right):
        """Hermite cubic at z clipped to the grid (coef = coefficients gathered for
        z's intervals), plus the linear extension with slopes left / right beyond."""
        inside = np.clip(z, self.start, self.stop)
        t = (inside - self.start) / STEP
        k = np.minimum(t.astype(np.intp), self.coef.shape[2] - 1)
        c = coef(k)
        t -= k
        value = ((c[3] * t + c[2]) * t + c[1]) * t + c[0]
        beyond = z - inside
        if beyond.any():
            value = value + np.where(beyond < 0, left, right) * beyond
        return value

    def every_row(self, z):
        """All N functions at every z, shape (N, *z.shape)."""
        left, right = (_broadcast(s, z.ndim) for s in self.end_slopes)
        return self._evaluate(z, lambda k: self.coef[:, :, k], left, right)

    def own_row(self, z):
        """Function i at z[i, ...], shape of z."""
        rows = _broadcast(np.arange(z.shape[0]), z.ndim - 1)
        left, right = (s[rows] for s in self.end_slopes)
        return self._evaluate(z, lambda k: self.coef[:, rows, k], left, right)


def _node_tables(a, b):
    """Table values and slopes for the nodes (a, b), each shape (n, 1):
    lambda(s) = logit G(expit(s)) and mu(t) = logit F^-1(expit(t))."""
    log_beta = betaln(a, b)
    # The small one of G and 1 - G is computed directly, the other as its complement.
    # Mirroring (1 - G(y) = G'(1 - y)) is no substitute: 1 - y rounds to 1 below
    # y ~ 1e-16, where G ~ y^a can still be far from 0 for small a.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.arange(-CDF_RANGE, CDF_RANGE + STEP / 2, STEP)
        log_y, log_1my = log_expit(s), log_expit(-s)
        G, one_minus_G = betainc(a, b, np.exp(log_y)), betainc(b, a, np.exp(log_1my))
        low = G < 0.5
        log_G = np.where(low, np.log(G), np.log1p(-one_minus_G))
        log_1mG = np.where(low, np.log1p(-G), np.log(one_minus_G))
        lam = log_G - log_1mG
        lam_slope = np.exp(a * log_y + b * log_1my - log_beta - log_G - log_1mG)

        # The same care for x and 1 - x: the quantile of u < 1/2 directly, of u >= 1/2
        # through the mirrored Beta at 1 - u.
        t = np.arange(-PPF_RANGE, PPF_RANGE + STEP / 2, STEP)
        log_u, log_1mu = log_expit(t), log_expit(-t)
        low = t < 0
        x = betaincinv(a, b, np.exp(log_u[low]))
        one_minus_x = betaincinv(b, a, np.exp(log_1mu[~low]))
        log_x = np.concatenate([np.log(x), np.log1p(-one_minus_x)], axis=1)
        log_1mx = np.concatenate([np.log1p(-x), np.log(one_minus_x)], axis=1)
        mu = log_x - log_1mx
        # d mu / d t = 1 / lambda'(mu)
        mu_slope = np.exp(log_u + log_1mu + log_beta - a * log_x - b * log_1mx)
    return lam, lam_slope, mu, mu_slope


class TabulatedBeta:
    """Beta CDF and quantile of the nodes from logit-logit tables; same interface
    as ExactBeta, absolute error below 1e-7 (tests/test_beta_tables.py)."""

    def __init__(self, params):
        a, b = params[:, 0:1], params[:, 1:2]
        self._size = len(params)
        # G = y^a / (a B(a, b)) (1 + O(y)) below the forward table
        self._tail = a, np.log(a) + betaln(a, b)
        # scipy's betainc / betaincinv release the GIL: build blocks of nodes in parallel
        blocks = np.array_split(np.arange(self._size), min(self._size, os.cpu_count() or 1))
        with ThreadPoolExecutor(len(blocks)) as pool:
            parts = list(pool.map(lambda rows: _node_tables(a[rows], b[rows]), blocks))
        lam, lam_slope, mu, mu_slope = (np.concatenate(p) for p in zip(*parts))
        self._cdf = _Table(-CDF_RANGE, *_clamp(lam, lam_slope, LOGIT_CLAMP))
        self._ppf = _Table(-PPF_RANGE, *_clamp(mu, mu_slope, QUANTILE_CLAMP))

    def __len__(self):
        return self._size

    def cdf(self, x):
        """F_i(x) for every node i at every x, shape (N, *x.shape)."""
        x = np.asarray(x, dtype=np.float64)
        z = np.clip(logit(x), -FAR, FAR)
        G = _expit(self._cdf.every_row(z))
        tail = z < -CDF_RANGE  # y < 4e-18: the power law is exact to double precision
        if tail.any():
            a, log_scale = self._tail
            with np.errstate(divide="ignore"):
                G[:, tail] = np.exp(a * np.log(x[tail]) - log_scale)
        return G

    def ppf(self, u):
        """F_i^-1(u[i, ...]): row i of u with node i, shape of u."""
        z = np.clip(logit(np.asarray(u, dtype=np.float64)), -FAR, FAR)
        return _expit(self._ppf.own_row(z))
