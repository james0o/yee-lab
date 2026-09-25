"""Exact ranking probabilities for every pixel (no voter discretization).

Voters of a pixel are distributed as X ~ Beta(a_x, b_x), Y ~ Beta(a_y, b_y)
(independent), where (a, b) are chosen so that the median is the pixel centre and the
mean absolute deviation from the median is `deviation`.

A voter's strict ranking of the candidates changes only when the voter crosses
the perpendicular bisector of some pair of candidates. The bisectors cut the unit
square into convex cells, each cell has one fixed ranking, and the cells are the
same for every pixel. Only the probability of each cell changes between pixels.
(Ties have probability zero and are ignored.)

Cell probability (Green's theorem, cell boundary oriented counter-clockwise):

    P(cell) = iint f(x) g(y) dx dy = oint omega,   omega = -f(x) G(y) dx

where f, F are the pdf / CDF of X and g, G of Y. With a, b < 1 the densities are
singular at 0 and 1, so the line integrals are computed after the substitution
u = F(x), i.e. -int G(l(F^-1(u))) du. On an edge that touches y = 0 or y = 1 the
equivalent form omega' = F(x) g(y) dy (omega' = omega + d(F G)) is integrated
with v = G(y) instead, which moves the singular point away from the integrand.
Every edge integral is shared by two cells (with opposite sign), so it is computed once.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import root
from scipy.special import betainc, betaincinv

from cache import DEFAULT_CACHE_ROOT, candidate_hash, metadata, read_metadata
from const import CANDIDATES, DEVIATION, PIXELS

QUAD_NODES = 24
EPS = 1e-12

# ---------------------------------------------------------------- Beta parameters

def _solve_beta(median, deviation, x0):
    """(a, b) with Beta median `median` and E|X - median| = `deviation`."""

    def equations(x):
        a, b = x
        if a <= 0 or b <= 0:
            return [1e10, 1e10]
        eq1 = betainc(a, b, median) - 0.5
        eq2 = a / (a + b) * (1 - 2 * betainc(a + 1, b, median)) - deviation
        return [eq1, eq2]

    sol = root(equations, x0, tol=1e-14, method="lm")
    return sol.x


def beta_params(pixels, deviation=DEVIATION):
    """Parameters (a, b) for pixel medians (k + 1/2) / pixels, shape (pixels, 2)."""
    medians = (np.arange(pixels) + 0.5) / pixels
    params = np.empty((pixels, 2), dtype=np.float64)
    x0 = [1.0, 1.0]
    upper = np.flatnonzero(medians >= 0.5)
    for k in upper:
        x0 = _solve_beta(medians[k], deviation, x0)
        params[k] = x0
    # Beta(a, b) mirrored around 1/2 is Beta(b, a).
    lower = np.flatnonzero(medians < 0.5)
    params[lower] = params[pixels - 1 - lower][:, ::-1]
    return params

# ---------------------------------------------------------------- Arrangement

def _clip(poly, normal, offset):
    """Part of convex CCW polygon `poly` with normal . p <= offset."""
    vals = poly @ normal - offset
    vals[np.abs(vals) < EPS] = 0.0
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        vp, vq = vals[k], vals[(k + 1) % n]
        if vp <= 0:
            out.append(p)
        if (vp < 0 < vq) or (vq < 0 < vp):
            out.append(p + vp / (vp - vq) * (q - p))
    if len(out) < 3:
        return None
    out = np.array(out)
    keep = np.linalg.norm(out - np.roll(out, -1, axis=0), axis=1) > 1e-10
    out = out[keep]
    if len(out) < 3:
        return None
    x, y = out[:, 0], out[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return out if area > 1e-14 else None


def ranking_cells(candidates):
    """Split the unit square by all pairwise bisectors.

    Returns (polygons, rankings): list of CCW vertex arrays and array (R, C) of
    candidate indices from best to worst, one row per polygon.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    n = len(candidates)
    polygons = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    for i in range(n):
        for j in range(i + 1, n):
            # closer to i  <=>  2 (c_j - c_i) . p < |c_j|^2 - |c_i|^2
            normal = 2 * (candidates[j] - candidates[i])
            offset = candidates[j] @ candidates[j] - candidates[i] @ candidates[i]
            split = []
            for poly in polygons:
                for part in (_clip(poly, normal, offset), _clip(poly, -normal, -offset)):
                    if part is not None:
                        split.append(part)
            polygons = split

    rankings = []
    for poly in polygons:
        centre = poly.mean(axis=0)
        dist = np.linalg.norm(candidates - centre, axis=1)
        rankings.append(np.argsort(dist))
    return polygons, np.array(rankings, dtype=np.uint8)

# ---------------------------------------------------------------- Edge integrals

def _on(value, target):
    return abs(value - target) < 1e-9


def _touches_y(p):
    return _on(p[1], 0.0) or _on(p[1], 1.0)


def _touches_x(p):
    return _on(p[0], 0.0) or _on(p[0], 1.0)


def _edge_integral(s, e, a, b, nodes, weights):
    """Integral of omega = -f(x) G(y) dx over the segment s -> e.

    Result has shape (pixels, pixels): [i, j] uses X-params of pixel i and
    Y-params of pixel j.
    """
    (xs, ys), (xe, ye) = s, e
    pixels = len(a)

    if _on(xs, xe):  # vertical: dx = 0
        return np.zeros((pixels, pixels))
    if _on(ys, ye):  # horizontal: -G(y) (F(xe) - F(xs))
        dF = betainc(a, b, xe) - betainc(a, b, xs)
        return -np.outer(dF, betainc(a, b, ys))

    ts, te = _touches_x(s) or _touches_x(e), _touches_y(s) or _touches_y(e)
    corner = (_touches_x(s) and _touches_y(s)) or (_touches_x(e) and _touches_y(e))
    if ts and te and not corner:
        mid = 0.5 * (np.asarray(s) + np.asarray(e))
        return (_edge_integral(s, mid, a, b, nodes, weights)
                + _edge_integral(mid, e, a, b, nodes, weights))

    half = 0.5 * (nodes + 1.0)
    if not te:
        # u = F_i(x): -int G_j(l(F_i^-1(u))) du
        us, ue = betainc(a, b, xs), betainc(a, b, xe)
        u = us[:, None] + (ue - us)[:, None] * half
        w = 0.5 * (ue - us)[:, None] * weights
        x = betaincinv(a[:, None], b[:, None], u)
        y = np.clip(ys + (x - xs) * (ye - ys) / (xe - xs), 0.0, 1.0)
        G = betainc(a[None, None, :], b[None, None, :], y[:, :, None])
        return -np.einsum("iq,iqj->ij", w, G, optimize=True)

    # v = G_j(y): int F_i(l^-1(G_j^-1(v))) dv, then omega = omega' - d(F G)
    vs, ve = betainc(a, b, ys), betainc(a, b, ye)
    v = vs[:, None] + (ve - vs)[:, None] * half
    w = 0.5 * (ve - vs)[:, None] * weights
    y = betaincinv(a[:, None], b[:, None], v)
    x = np.clip(xs + (y - ys) * (xe - xs) / (ye - ys), 0.0, 1.0)
    F = betainc(a[:, None, None], b[:, None, None], x[None, :, :])
    omega_prime = np.einsum("jq,ijq->ij", w, F, optimize=True)
    FG_end = np.outer(betainc(a, b, xe), ve)
    FG_start = np.outer(betainc(a, b, xs), vs)
    return omega_prime - (FG_end - FG_start)

# ---------------------------------------------------------------- Probabilities

def compute_ranking_probabilities(candidates, params, quad_nodes=QUAD_NODES, progress=None):
    """Returns (rankings (R, C), probabilities (pixels, pixels, R)).

    progress: optional callable(done, total), called after each edge integral.
    """
    polygons, rankings = ranking_cells(candidates)
    a, b = params[:, 0], params[:, 1]
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
    for done, (start, end, uses) in enumerate(edges.values(), start=1):
        integral = _edge_integral(start, end, a, b, nodes, weights)
        for r, sign in uses:
            probs[..., r] += sign * integral
        if progress is not None:
            progress(done, len(edges))

    # Guard against duplicate rankings from sliver cells (should not happen).
    unique, inverse = np.unique(rankings, axis=0, return_inverse=True)
    if len(unique) != len(rankings):
        merged = np.zeros((*probs.shape[:2], len(unique)))
        np.add.at(merged, (..., inverse.ravel()), probs)
        rankings, probs = unique, merged
    return rankings, np.clip(probs, 0.0, 1.0)


def rankings_path(pixels, deviation, candidates, quad_nodes, cache_root=DEFAULT_CACHE_ROOT):
    token = format(float(deviation), ".12g").replace("-", "m").replace(".", "p")
    name = f"P{pixels}_D{token}_Q{quad_nodes}_C{candidate_hash(candidates)}.npz"
    return Path(cache_root) / "rankings" / name


def _rankings_metadata(pixels, deviation, candidates, quad_nodes):
    return metadata(
        "rankings",
        pixels=int(pixels),
        deviation=float(deviation),
        quad_nodes=int(quad_nodes),
        candidate_hash=candidate_hash(candidates),
    )


def read_cached_ranking_probabilities(
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
):
    """(rankings, probabilities) from the cache, or None if not cached."""
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, candidates, quad_nodes, cache_root)
    if not path.exists():
        return None
    expected = _rankings_metadata(pixels, deviation, candidates, quad_nodes)
    with np.load(path) as archive:
        if read_metadata(archive) == json.loads(expected) and \
                np.array_equal(archive["candidates"], candidates):
            return archive["rankings"], archive["probabilities"]
    return None


def load_ranking_probabilities(
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
):
    """Cached (rankings (R, C), probabilities (pixels, pixels, R)).

    probabilities[x, y, r] is the share of voters of pixel (x, y) whose ranking
    (best to worst) is rankings[r]. Pixel index x follows the x axis.
    """
    cached = read_cached_ranking_probabilities(
        candidates, pixels, deviation, quad_nodes, cache_root
    )
    if cached is not None:
        return cached
    return generate_ranking_probabilities(
        candidates, pixels, deviation, quad_nodes, cache_root
    )


def generate_ranking_probabilities(
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    progress=None,
):
    """Compute (rankings, probabilities) and save them to the cache.

    progress: optional callable(done, total), see compute_ranking_probabilities.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, candidates, quad_nodes, cache_root)
    expected = _rankings_metadata(pixels, deviation, candidates, quad_nodes)
    params = beta_params(pixels, deviation)
    rankings, probs = compute_ranking_probabilities(
        candidates, params, quad_nodes, progress
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rankings=rankings,
        probabilities=probs,
        params=params,
        candidates=candidates,
        metadata=np.array(expected),
    )
    return rankings, probs

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


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    rankings, probs = load_ranking_probabilities()
    print(f"{len(rankings)} rankings, probabilities {probs.shape}, "
          f"{time.perf_counter() - start:.1f} s")
