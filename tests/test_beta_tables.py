"""Checks of the tabulated Beta CDF and quantile (beta_tables.py).

The tables are compared with scipy over the whole range of doubles, for the node
distributions the web UI uses (the extremes of the deviation slider and every spread
rule), and the edge integrals built on them with the exact ones.
"""

import numpy as np
import pytest
from scipy.special import expit

from beta_tables import ExactBeta, TabulatedBeta
from ranking_cells import SPREADS, _edge_integral, node_params
from shares import PIXELS, node_count

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


@pytest.mark.parametrize("edge", [
    ((0.1, 0.2), (0.7, 0.6)),  # interior: u = F(x)
    ((0.2, 0.0), (0.5, 0.6)),  # from the bottom wall: v = G(y)
    ((0.0, 0.3), (0.6, 1.0)),  # left wall to top wall: split
    ((0.2, 0.4), (0.8, 0.4)),  # horizontal: closed form
], ids=["interior", "y-wall", "both-walls", "horizontal"])
def test_edge_integrals_match_exact(betas, edge):
    exact, table = betas
    s, e = np.array(edge, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    np.testing.assert_allclose(_edge_integral(s, e, table, nodes, weights),
                               _edge_integral(s, e, exact, nodes, weights), rtol=0, atol=1e-7)
