"""Checks of the tabulated Beta CDF and quantile (margin/beta_tables.py).

The tables are compared with scipy over the whole range of doubles, for the node
distributions the web UI uses (the extremes of the deviation slider and every spread
rule), and the edge integrals built on them with the exact ones.
"""

import numpy as np
import pytest
from scipy.special import betainc, betaincinv, expit

from yeelab.margin.beta_tables import TabulatedBeta
from yeelab.margin.shares import PIXELS, node_count
from yeelab.ranking_cells import SPREADS, node_params


class ExactBeta:
    """Direct scipy CDF/quantile reference for the compiled tabulated implementation."""

    def __init__(self, params):
        self.a, self.b = params[:, 0], params[:, 1]

    def __len__(self):
        return len(self.a)

    def cdf(self, values):
        values = np.asarray(values)
        shape = (-1,) + (1,) * values.ndim
        return betainc(self.a.reshape(shape), self.b.reshape(shape), values)

    def ppf(self, values):
        values = np.asarray(values)
        shape = (-1,) + (1,) * (values.ndim - 1)
        return betaincinv(self.a.reshape(shape), self.b.reshape(shape), values)

    def cdf_sums(self, points, weights):
        return np.einsum("rq,nrq->rn", weights, self.cdf(points), optimize=True)


def _on(value, target):
    return abs(value - target) < 1e-9


def _touches_x(point):
    return _on(point[0], 0.0) or _on(point[0], 1.0)


def _touches_y(point):
    return _on(point[1], 0.0) or _on(point[1], 1.0)


def _edge_integral(start, end, beta, nodes, weights):
    """Independent scipy quadrature for one Beta-voter polygon edge."""
    (xs, ys), (xe, ye) = start, end
    if _on(xs, xe):
        return np.zeros((len(beta), len(beta)))
    if _on(ys, ye):
        return -np.outer(beta.cdf(xe) - beta.cdf(xs), beta.cdf(ys))
    touches_x = _touches_x(start) or _touches_x(end)
    touches_y = _touches_y(start) or _touches_y(end)
    corner = (_touches_x(start) and _touches_y(start)) or (_touches_x(end) and _touches_y(end))
    if touches_x and touches_y and not corner:
        middle = 0.5 * (np.asarray(start) + np.asarray(end))
        return (_edge_integral(start, middle, beta, nodes, weights)
                + _edge_integral(middle, end, beta, nodes, weights))
    half = 0.5 * (nodes + 1)
    if not touches_y:
        us, ue = beta.cdf(xs), beta.cdf(xe)
        u = us[:, None] + (ue - us)[:, None] * half
        w = 0.5 * (ue - us)[:, None] * weights
        x = beta.ppf(u)
        y = np.clip(ys + (x - xs) * (ye - ys) / (xe - xs), 0, 1)
        return -beta.cdf_sums(y, w)
    vs, ve = beta.cdf(ys), beta.cdf(ye)
    v = vs[:, None] + (ve - vs)[:, None] * half
    w = 0.5 * (ve - vs)[:, None] * weights
    y = beta.ppf(v)
    x = np.clip(xs + (y - ys) * (xe - xs) / (ye - ys), 0, 1)
    return beta.cdf_sums(x, w).T - (np.outer(beta.cdf(xe), ve) - np.outer(beta.cdf(xs), vs))

# every double y in (0, 1), including the tails below the table, and both walls
POINTS = np.concatenate([expit(np.linspace(-745, 37, 4001)), [0.0, 1.0]])
# probabilities the edge integrals invert; below 1e-13 the quantile does not matter
PROBABILITIES = expit(np.linspace(-30, 30, 1201))


@pytest.fixture(scope="module", params=[(d, s) for d in (0.05, 0.2, 0.4) for s in SPREADS],
                ids=lambda p: f"{p[0]}-{p[1]}")
def betas(request):
    deviation, spread = request.param
    params = node_params(PIXELS, node_count("beta", deviation), deviation, spread)[1]
    return ExactBeta(params), TabulatedBeta(params)


def test_cdf_matches_scipy(betas):
    exact, table = betas
    np.testing.assert_allclose(table.cdf(POINTS), exact.cdf(POINTS), rtol=0, atol=1e-8)


def test_quantile_matches_scipy(betas):
    exact, table = betas
    u = np.broadcast_to(PROBABILITIES, (len(exact), len(PROBABILITIES)))
    np.testing.assert_allclose(table.ppf(u), exact.ppf(u), rtol=0, atol=1e-7)


def test_scalar_and_walls(betas):
    _, table = betas
    np.testing.assert_array_equal(table.cdf(0.0), 0.0)
    np.testing.assert_array_equal(table.cdf(1.0), 1.0)
    assert table.cdf(0.3).shape == (len(table),)


EDGES = {
    "interior": ((0.1, 0.2), (0.7, 0.6)),     # u = F(x)
    "y-wall": ((0.2, 0.0), (0.5, 0.6)),       # from the bottom wall: v = G(y)
    "both-walls": ((0.0, 0.3), (0.6, 1.0)),   # left wall to top wall: split
    "horizontal": ((0.2, 0.4), (0.8, 0.4)),   # closed form
    "vertical": ((0.3, 0.1), (0.3, 0.9)),     # dx = 0
    "corner": ((0.0, 0.0), (0.7, 0.4)),       # from a corner: v form, not split
    "backwards": ((1.0, 0.2), (0.0, 0.9)),    # wall to wall, right to left
}


def test_edge_integrals_match_exact(betas):
    """Check compiled edge integrals against the independent scipy reference."""
    exact, table = betas
    nodes, weights = np.polynomial.legendre.leggauss(24)
    edges = np.array(list(EDGES.values()), dtype=np.float64)
    got = table.edge_terms(edges.reshape(-1, 4), nodes, weights)
    for name, (s, e), term in zip(EDGES, edges, got):
        np.testing.assert_allclose(term, _edge_integral(s, e, exact, nodes, weights),
                                   rtol=0, atol=1e-7, err_msg=name)
