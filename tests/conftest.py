"""The ranking profiles the methods are checked on (test_build.py, test_strategic.py)."""

import sys
from itertools import product

import numpy as np
import pytest

from yeelab.margin.shares import Model, ranking_shares

@pytest.fixture(scope="session", autouse=True)
def _warmed_up():
    """The web app warms its caches in a thread when it is imported (at collection, by the
    tests of the API): tests that count cache misses must not see its computations."""
    app = sys.modules.get("yeelab.web.app")
    if app is not None:
        app.WARM_UP.join()


# voters (distribution, deviation, Beta spread) and random candidates (number, seed)
VOTERS = [("beta", 0.1, "rms"), ("beta", 0.3, "mean_abs"), ("normal", 0.1, None), ("normal", 0.3, None)]
CANDIDATES = [(3, 0), (5, 1), (7, 2)]


@pytest.fixture(scope="session", params=list(product(VOTERS, CANDIDATES)),
                ids=lambda p: f"{p[0][0]}-{p[0][1]}-{p[1][0]}")
def profile(request):
    """(rankings (R, C), node shares (N, N, R)) from the shared margin integrator."""
    (distribution, deviation, spread), (n, seed) = request.param
    candidates = np.random.default_rng(seed).random((n, 2))
    model = Model(distribution, deviation, spread if distribution == "beta" else None)
    return ranking_shares(candidates, model)
