"""Ranking probabilities for every pixel (no voter discretization).

Voters of a pixel are distributed as X ~ Beta(a_x, b_x), Y ~ Beta(a_y, b_y)
(independent), where (a, b) are chosen so that the median is the pixel centre. The
spread rule (`spread`) fixes one more quantity for every pixel:

    "mean_abs"  E|X - m| = deviation (legacy: pushes voters away near the walls)
    "rms"       sqrt(E (X - m)^2) is the same as at the centre pixel
    "tapered"   a + b is the centre value times (4 m (1 - m))^TAPER, TAPER = 0.2

All give the same centre pixel, Beta(a0, a0) with E|X - 1/2| = deviation.
Near the walls only "mean_abs" keeps the spread fixed, which pushes the voters on
the far side of the median away (docs/math.typ).

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

Each edge integral costs O(pixels^2) Beta CDF evaluations. The probabilities are
smooth in the pixel median (only the winners jump), so they are computed exactly
on NODES x NODES Chebyshev-Lobatto points in logit(median) and interpolated to
the pixels (barycentric formula). The points cluster towards 0 and 1, where the
Beta parameters change fastest; the error decreases exponentially with NODES.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import betainc, betaincinv, betaln, expit, logit

from cache import DEFAULT_CACHE_ROOT, candidate_hash, metadata, read_metadata
from const import CANDIDATES, DEVIATION, PIXELS

QUAD_NODES = 24
NODES = 49
EPS = 1e-12

Spread = Literal["mean_abs", "rms", "tapered"]
SPREADS = get_args(Spread)
SPREAD: Spread = "mean_abs"
CONTINUATION_STEP = 0.25  # largest step in logit(median) of the mean_abs solver
# Exponent of the "tapered" rule, found by optimisation: it gave the straightest
# Condorcet borders at equal numbers of cycle pixels in a benchmark (docs/math.typ).
TAPER = 0.2

# ---------------------------------------------------------------- Beta parameters

def centre_shape(deviation):
    """a0 of the symmetric Beta(a0, a0) with E|X - 1/2| = `deviation`.

    E|X - 1/2| = 2^(-2 a0) / (a0 B(a0, a0)) decreases from 1/2 (a0 -> 0) to 0.
    """
    if not 0 < deviation < 0.5:
        raise ValueError("deviation must be between 0 and 1/2")
    log_deviation = np.log(deviation)
    return brentq(
        lambda a: -2 * a * np.log(2) - np.log(a) - betaln(a, a) - log_deviation,
        1e-12, 1e12, xtol=1e-15,
    )


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


def _b_for_median(kappa, median):
    """b of Beta(kappa - b, b) with median `median` >= 1/2. For a fixed a + b the
    median decreases as b grows, from 1 (b -> 0) to 1/2 (b = kappa / 2)."""
    if median == 0.5:
        return kappa / 2
    return brentq(lambda b: betainc(kappa - b, b, median) - 0.5,
                  1e-12 * kappa, kappa / 2, xtol=1e-15)


def _rms_from_median(a, b, median):
    """sqrt(E (X - median)^2) = sqrt(Var X + (E X - median)^2)."""
    mean = a / (a + b)
    variance = a * b / ((a + b) ** 2 * (a + b + 1))
    return np.sqrt(variance + (mean - median) ** 2)


def _solve_rms(upper, a0):
    """(a, b) for the increasing medians `upper` >= 1/2 whose RMS distance from the
    median equals that of the centre pixel Beta(a0, a0).

    Along the Beta distributions with a given median, the RMS distance decreases as
    a + b grows. It is solved over log(a + b), bracketing outwards from the solution
    at the previous median.
    """
    target = np.log(_rms_from_median(a0, a0, 0.5))
    solved, log_kappa = [], np.log(2 * a0)

    def excess(t, median):
        kappa = np.exp(t)
        b = _b_for_median(kappa, median)
        return np.log(_rms_from_median(kappa - b, b, median)) - target

    for median in upper:
        f0, step = excess(log_kappa, median), 0.05
        while f0 != 0:
            lower, higher = log_kappa - step, log_kappa + step
            if np.sign(excess(lower, median)) != np.sign(f0):
                log_kappa = brentq(excess, lower, log_kappa, args=(median,), xtol=1e-14)
                break
            if np.sign(excess(higher, median)) != np.sign(f0):
                log_kappa = brentq(excess, log_kappa, higher, args=(median,), xtol=1e-14)
                break
            step *= 2
            if step > 40:
                raise RuntimeError(f"no Beta with median {median} and this RMS distance")
        kappa = np.exp(log_kappa)
        b = _b_for_median(kappa, median)
        solved.append((kappa - b, b))
    return np.array(solved, dtype=np.float64)


def _upper_params(upper, deviation, spread):
    """(a, b) for the increasing medians `upper` >= 1/2, shape (len(upper), 2)."""
    if spread not in SPREADS:
        raise ValueError(f"unknown spread {spread!r}; choose from {', '.join(SPREADS)}")
    a0 = centre_shape(deviation)
    if spread == "mean_abs":
        # Continuation from the centre, Beta(a0, a0), outwards, with steps of at
        # most CONTINUATION_STEP in logit(median) so the solver stays on its branch.
        solved = np.empty((len(upper), 2), dtype=np.float64)
        x0, z0 = [a0, a0], 0.0
        for k, median in enumerate(upper):
            z = logit(median)
            steps = int(np.ceil((z - z0) / CONTINUATION_STEP))
            for t in np.linspace(z0, z, steps + 1)[1:-1]:
                x0 = _solve_beta(expit(t), deviation, x0)
            x0 = _solve_beta(median, deviation, x0)
            solved[k], z0 = x0, z
        return solved
    if spread == "tapered":
        # 4 m (1 - m) is 1 at the centre and falls to 0 at the walls. Only within
        # ~1e-5 of a wall does a + b get so small that the spread grows again.
        kappa = 2 * a0 * (4 * upper * (1 - upper)) ** TAPER
        b = np.array([_b_for_median(k, median) for k, median in zip(kappa, upper)])
        return np.column_stack([kappa - b, b])
    return _solve_rms(upper, a0)


def pixel_medians(pixels):
    """Medians (k + 1/2) / pixels, the pixel centres."""
    return (np.arange(pixels) + 0.5) / pixels


def beta_params_at(medians, deviation=DEVIATION, spread=SPREAD):
    """Parameters (a, b) for each median, shape (len(medians), 2)."""
    if not 0 < deviation < 0.5:
        raise ValueError("deviation must be between 0 and 1/2")
    medians = np.asarray(medians, dtype=np.float64)
    upper, inverse = np.unique(np.maximum(medians, 1 - medians), return_inverse=True)
    params = _upper_params(upper, deviation, spread)[inverse]
    # Beta(a, b) mirrored around 1/2 is Beta(b, a).
    lower = medians < 0.5
    params[lower] = params[lower, ::-1]
    return params


def beta_params(pixels, deviation=DEVIATION, spread=SPREAD):
    """Parameters (a, b) for pixel medians (k + 1/2) / pixels, shape (pixels, 2)."""
    return beta_params_at(pixel_medians(pixels), deviation, spread)

# ---------------------------------------------------------------- Interpolation

def effective_nodes(pixels, nodes):
    """Nodes per axis actually used; 0 means exact at every pixel, which is
    also used when interpolation would not save work (nodes >= pixels)."""
    if nodes < 0 or nodes == 1:
        raise ValueError("nodes must be 0 (exact) or at least 2")
    return 0 if nodes >= pixels else nodes


def node_medians(pixels, nodes=NODES):
    """Medians at which probabilities are computed exactly: `nodes`
    Chebyshev-Lobatto points in logit(median) spanning the pixel medians,
    or the pixel medians themselves if effective_nodes(...) is 0."""
    nodes = effective_nodes(pixels, nodes)
    if nodes == 0:
        return pixel_medians(pixels)
    t = np.sin(np.pi * (2 * np.arange(nodes) - (nodes - 1)) / (2 * (nodes - 1)))
    return expit(-logit(0.5 / pixels) * t)


def _interpolation_matrix(nodes, targets):
    """L (targets, nodes) with f(targets) ~ L @ f(nodes) for Chebyshev-Lobatto
    `nodes` (barycentric formula, weights (-1)^k halved at both ends)."""
    weights = (-1.0) ** np.arange(len(nodes))
    weights[[0, -1]] *= 0.5
    diff = targets[:, None] - nodes[None, :]
    hit = diff == 0
    diff[hit] = 1.0
    matrix = weights / diff
    matrix /= matrix.sum(axis=1, keepdims=True)
    rows = hit.any(axis=1)
    matrix[rows] = hit[rows]
    return matrix


def interpolate_to_pixels(probs, medians, pixels, transform=logit):
    """Probabilities (pixels, pixels, R) from probabilities (N, N, R) at `medians`,
    which are Chebyshev-Lobatto points in transform(median).

    The result is float32: it is the largest array by far and only feeds the
    voting methods, where float32 halves memory and time. Its rounding (~1e-7)
    can only flip pixels whose winning margin is that small.
    """
    targets = pixel_medians(pixels)
    if np.array_equal(medians, targets):
        return probs.astype(np.float32)
    matrix = _interpolation_matrix(transform(medians), transform(targets))
    # columns[i, q, r] = sum_j matrix[q, j] probs[i, j, r]; small, so kept in float64
    columns = (matrix @ probs).reshape(len(medians), -1).astype(np.float32)
    # one (pixels, N) @ (N, pixels * R) product for the large result
    probs = (matrix.astype(np.float32) @ columns).reshape(pixels, pixels, -1)
    return np.clip(probs, 0.0, 1.0, out=probs)

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
    lock = threading.Lock()

    def integrate(start, end, uses):
        integral = _edge_integral(start, end, a, b, nodes, weights)
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


def _token(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def rankings_path(pixels, deviation, spread, candidates, quad_nodes, nodes,
                  cache_root=DEFAULT_CACHE_ROOT):
    name = (f"P{pixels}_D{_token(deviation)}_S{spread}_Q{quad_nodes}_N{effective_nodes(pixels, nodes)}"
            f"_C{candidate_hash(candidates)}.npz")
    return Path(cache_root) / "rankings" / name


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
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
    nodes=NODES,
    quad_nodes=QUAD_NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    spread=SPREAD,
):
    """(rankings, probabilities) from the cache, or None if not cached."""
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, spread, candidates, quad_nodes, nodes, cache_root)
    if not path.exists():
        return None
    expected = _rankings_metadata(pixels, deviation, spread, candidates, quad_nodes, nodes)
    with np.load(path) as archive:
        if read_metadata(archive) == json.loads(expected) and \
                np.array_equal(archive["candidates"], candidates):
            probs = interpolate_to_pixels(
                archive["node_probabilities"], archive["medians"], pixels
            )
            return archive["rankings"], probs
    return None


def load_ranking_probabilities(
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
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
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
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
    medians = node_medians(pixels, nodes)
    params = beta_params_at(medians, deviation, spread)
    rankings, probs = compute_ranking_probabilities(
        candidates, params, quad_nodes, progress
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rankings=rankings,
        node_probabilities=probs,
        medians=medians,
        params=params,
        candidates=candidates,
        metadata=np.array(expected),
    )
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


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    rankings, probs = load_ranking_probabilities()
    print(f"{len(rankings)} rankings, probabilities {probs.shape}, "
          f"{time.perf_counter() - start:.1f} s")
