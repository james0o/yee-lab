"""Ranking probabilities of every pixel for Beta voters, exact and cached.

Every cell of the bisector arrangement (ranking_cells.ranking_cells) has one ranking,
and its probability is (Green's theorem, cell boundary oriented counter-clockwise)

    P(cell) = iint f(x) g(y) dx dy = oint omega,   omega = -f(x) G(y) dx

where f, F are the pdf / CDF of X and g, G of Y. With a, b < 1 the densities are
singular at 0 and 1, so the line integrals are computed after the substitution
u = F(x), i.e. -int G(l(F^-1(u))) du. On an edge that touches y = 0 or y = 1 the
equivalent form omega' = F(x) g(y) dy (omega' = omega + d(F G)) is integrated
with v = G(y) instead, which moves the singular point away from the integrand.
Every edge integral is shared by two cells (with opposite sign), so it is computed once.

Each edge integral costs O(pixels^2) Beta CDF evaluations, so they are computed at the
Chebyshev nodes (ranking_cells.node_medians) and interpolated to the pixels.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaincinv, logit

from yeelab.pixels.cache import (
    DEFAULT_CACHE_ROOT,
    candidate_hash,
    load_node_probabilities,
    metadata,
    save_node_probabilities,
    value_token,
)
from yeelab.ranking_cells import (
    NODES,
    QUAD_NODES,
    SPREAD,
    TAPER,
    effective_nodes,
    interpolate_to,
    node_params,
    pixel_medians,
    ranking_cells,
)


def _broadcast(values, extra):
    """(N,) parameters shaped (N, 1, ..., 1) with `extra` trailing axes."""
    return values.reshape(-1, *(1,) * extra)


class ExactBeta:
    """Beta CDF and quantile of the nodes, straight from scipy (margin.beta_tables
    has the same interface from tables)."""

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


def interpolate_to_pixels(probs, medians, pixels, transform=logit):
    """Probabilities (pixels, pixels, R) from probabilities (N, N, R) at `medians`
    (see ranking_cells.interpolate_to), clipped to [0, 1]."""
    probs = interpolate_to(probs, medians, pixel_medians(pixels), transform)
    return np.clip(probs, 0.0, 1.0, out=probs)

# ---------------------------------------------------------------- Edge integrals

def _on(value, target):
    return abs(value - target) < 1e-9


def _touches_y(p):
    return _on(p[1], 0.0) or _on(p[1], 1.0)


def _touches_x(p):
    return _on(p[0], 0.0) or _on(p[0], 1.0)


def _edge_integral(s, e, beta, nodes, weights):
    """Integral of omega = -f(x) G(y) dx over the segment s -> e.

    beta: the CDF and quantile of the pixels' Beta distributions (ExactBeta;
    margin.beta_tables.TabulatedBeta.edge_terms is this function compiled for the tables). Result has
    shape (pixels, pixels): [i, j] uses X-params of pixel i and Y-params of pixel j.
    """
    (xs, ys), (xe, ye) = s, e
    pixels = len(beta)

    if _on(xs, xe):  # vertical: dx = 0
        return np.zeros((pixels, pixels))
    if _on(ys, ye):  # horizontal: -G(y) (F(xe) - F(xs))
        dF = beta.cdf(xe) - beta.cdf(xs)
        return -np.outer(dF, beta.cdf(ys))

    ts, te = _touches_x(s) or _touches_x(e), _touches_y(s) or _touches_y(e)
    corner = (_touches_x(s) and _touches_y(s)) or (_touches_x(e) and _touches_y(e))
    if ts and te and not corner:
        mid = 0.5 * (np.asarray(s) + np.asarray(e))
        return (_edge_integral(s, mid, beta, nodes, weights)
                + _edge_integral(mid, e, beta, nodes, weights))

    half = 0.5 * (nodes + 1.0)
    if not te:
        # u = F_i(x): -int G_j(l(F_i^-1(u))) du
        us, ue = beta.cdf(xs), beta.cdf(xe)
        u = us[:, None] + (ue - us)[:, None] * half
        w = 0.5 * (ue - us)[:, None] * weights
        x = beta.ppf(u)
        y = np.clip(ys + (x - xs) * (ye - ys) / (xe - xs), 0.0, 1.0)
        return -beta.cdf_sums(y, w)  # [i, j] = sum_q w[i, q] G_j(y[i, q])

    # v = G_j(y): int F_i(l^-1(G_j^-1(v))) dv, then omega = omega' - d(F G)
    vs, ve = beta.cdf(ys), beta.cdf(ye)
    v = vs[:, None] + (ve - vs)[:, None] * half
    w = 0.5 * (ve - vs)[:, None] * weights
    y = beta.ppf(v)
    x = np.clip(xs + (y - ys) * (xe - xs) / (ye - ys), 0.0, 1.0)
    omega_prime = beta.cdf_sums(x, w).T  # [i, j] = sum_q w[j, q] F_i(x[j, q])
    FG_end = np.outer(beta.cdf(xe), ve)
    FG_start = np.outer(beta.cdf(xs), vs)
    return omega_prime - (FG_end - FG_start)

# ---------------------------------------------------------------- Probabilities

def compute_ranking_probabilities(candidates, params, quad_nodes=QUAD_NODES, progress=None):
    """Returns (rankings (R, C), probabilities (pixels, pixels, R)).

    progress: optional callable(done, total), called after each edge integral.
    """
    polygons, rankings = ranking_cells(candidates)
    beta = ExactBeta(params)
    nodes, weights = np.polynomial.legendre.leggauss(quad_nodes)

    def key(p):
        return (round(float(p[0]), 9), round(float(p[1]), 9))

    # edge -> (start, end, [(cell, sign), ...]); each edge is shared by up to two cells
    edges = {}
    for r, poly in enumerate(polygons):
        for k in range(len(poly)):
            s, e = poly[k], poly[(k + 1) % len(poly)]
            ks, ke = key(s), key(e)
            forward = ks < ke
            edge_key = (ks, ke) if forward else (ke, ks)
            if edge_key not in edges:
                edges[edge_key] = ((s, e) if forward else (e, s)) + ([],)
            edges[edge_key][2].append((r, 1.0 if forward else -1.0))

    probs = np.zeros((len(params), len(params), len(polygons)))
    lock = threading.Lock()

    def integrate(start, end, uses):
        integral = _edge_integral(start, end, beta, nodes, weights)
        with lock:
            for r, sign in uses:
                probs[..., r] += sign * integral

    # betainc / betaincinv release the GIL, so edges are integrated in parallel.
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(integrate, *edge) for edge in edges.values()]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if progress is not None:
                progress(done, len(edges))

    # Guard against duplicate rankings from sliver cells (should not happen).
    unique, inverse = np.unique(rankings, axis=0, return_inverse=True)
    if len(unique) != len(rankings):
        merged = np.zeros((*probs.shape[:2], len(unique)))
        np.add.at(merged, (..., inverse.ravel()), probs)
        rankings, probs = unique, merged
    return rankings, np.clip(probs, 0.0, 1.0)


def ranking_probabilities(candidates, pixels, deviation, nodes=NODES,
                          spread=SPREAD, quad_nodes=QUAD_NODES, progress=None):
    """(rankings, probabilities (pixels, pixels, R)) computed at the node medians
    and interpolated, without the cache."""
    medians, params = node_params(pixels, nodes, deviation, spread)
    rankings, probs = compute_ranking_probabilities(candidates, params, quad_nodes, progress)
    return rankings, interpolate_to_pixels(probs, medians, pixels)

# ---------------------------------------------------------------- Cache

def rankings_path(pixels, deviation, spread, candidates, quad_nodes, nodes,
                  cache_root=DEFAULT_CACHE_ROOT):
    name = (f"P{pixels}_D{value_token(deviation)}_Q{quad_nodes}_N{effective_nodes(pixels, nodes)}"
            f"_C{candidate_hash(candidates)}.npz")
    return Path(cache_root) / "beta" / spread / name


def _rankings_metadata(pixels, deviation, spread, candidates, quad_nodes, nodes):
    return metadata(
        "rankings",
        pixels=int(pixels),
        deviation=float(deviation),
        spread=spread,
        **({"taper": TAPER} if spread == "tapered" else {}),
        quad_nodes=int(quad_nodes),
        nodes=effective_nodes(pixels, nodes),
        candidate_hash=candidate_hash(candidates),
    )


def read_cached_ranking_probabilities(
    candidates,
    pixels,
    deviation,
    nodes=NODES,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    spread=SPREAD,
):
    """(rankings, probabilities) from the cache, or None if not cached."""
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, spread, candidates, quad_nodes, nodes, cache_root)
    expected = _rankings_metadata(pixels, deviation, spread, candidates, quad_nodes, nodes)
    saved = load_node_probabilities(path, expected, candidates)
    if saved is None:
        return None
    rankings, probs, medians = saved
    return rankings, interpolate_to_pixels(probs, medians, pixels)


def load_ranking_probabilities(
    candidates,
    pixels,
    deviation,
    nodes=NODES,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    spread=SPREAD,
):
    """Cached (rankings (R, C), probabilities (pixels, pixels, R)).

    probabilities[x, y, r] is the share of voters of pixel (x, y) whose ranking
    (best to worst) is rankings[r]. Pixel index x follows the x axis.
    """
    cached = read_cached_ranking_probabilities(
        candidates, pixels, deviation, nodes, quad_nodes, cache_root, spread
    )
    if cached is not None:
        return cached
    return generate_ranking_probabilities(
        candidates, pixels, deviation, nodes, quad_nodes, cache_root, spread=spread
    )


def generate_ranking_probabilities(
    candidates,
    pixels,
    deviation,
    nodes=NODES,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    progress=None,
    spread=SPREAD,
):
    """Compute (rankings, probabilities) and save them to the cache.

    Only the probabilities at the node medians are saved; they are interpolated
    to the pixels on every load.
    progress: optional callable(done, total), see compute_ranking_probabilities.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, spread, candidates, quad_nodes, nodes, cache_root)
    expected = _rankings_metadata(pixels, deviation, spread, candidates, quad_nodes, nodes)
    medians, params = node_params(pixels, nodes, deviation, spread)
    rankings, probs = compute_ranking_probabilities(
        candidates, params, quad_nodes, progress
    )
    save_node_probabilities(path, expected, rankings, probs, medians, candidates, params=params)
    return rankings, interpolate_to_pixels(probs, medians, pixels)

# ---------------------------------------------------------------- Validation

def monte_carlo_check(candidates, params, rankings, probs, pixel_ids, samples=2_000_000, seed=0):
    """Max absolute difference between exact and sampled ranking probabilities."""
    rng = np.random.default_rng(seed)
    candidates = np.asarray(candidates, dtype=np.float64)
    n = len(candidates)
    codes = rankings.astype(np.int64) @ (n ** np.arange(n))
    lookup = {int(c): r for r, c in enumerate(codes)}
    worst = 0.0
    for i, j in pixel_ids:
        x = rng.beta(params[i, 0], params[i, 1], samples)
        y = rng.beta(params[j, 0], params[j, 1], samples)
        dist = np.hypot(x[:, None] - candidates[:, 0], y[:, None] - candidates[:, 1])
        sample_codes = np.argsort(dist, axis=1) @ (n ** np.arange(n))
        values, counts = np.unique(sample_codes, return_counts=True)
        empirical = np.zeros(len(rankings))
        for value, count in zip(values, counts):
            empirical[lookup[int(value)]] = count / samples
        diff = np.abs(empirical - probs[i, j]).max()
        worst = max(worst, diff)
        print(f"pixel ({i:3d}, {j:3d}): max |exact - MC| = {diff:.2e}")
    return worst
