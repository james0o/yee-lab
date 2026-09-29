"""Beta voters of a pixel, the cells of equal ranking, and interpolation from nodes.

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
(Ties have probability zero and are ignored.) The probabilities are edge integrals
over the cell boundaries: exact for every cell in pixels/beta.py, from tables and
compiled in margin/beta_tables.py.

The probabilities are smooth in the pixel median (only the winners jump), so they
are computed exactly on NODES x NODES Chebyshev-Lobatto points in logit(median) and
interpolated (barycentric formula). The points cluster towards 0 and 1, where the
Beta parameters change fastest; the error decreases exponentially with NODES.
"""

import math
from functools import lru_cache
from typing import Literal, get_args

import numpy as np
from numba import njit
from scipy.optimize import brentq, root
from scipy.special import betainc, betaln, expit, logit

QUAD_NODES = 24  # Gauss-Legendre points per edge integral
NODES = 49
EPS = 1e-12

Spread = Literal["mean_abs", "rms", "tapered"]
SPREADS = get_args(Spread)
SPREAD: Spread = "rms"
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


def beta_params_at(medians, deviation, spread=SPREAD):
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


def beta_params(pixels, deviation, spread=SPREAD):
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


@lru_cache(maxsize=None)
def node_params(pixels, nodes, deviation, spread=SPREAD):
    """(node medians, Beta parameters at them), read-only. Kept in memory: they
    depend only on the grid and the spread rule, never on the candidates."""
    medians = node_medians(pixels, nodes)
    params = beta_params_at(medians, deviation, spread)
    medians.setflags(write=False)
    params.setflags(write=False)
    return medians, params


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


def interpolate_to(probs, medians, targets, transform=logit):
    """Values (T, T, ...) at the medians `targets` on both axes from values
    (N, N, ...) at `medians`, which are Chebyshev-Lobatto points in transform(median).

    The result is float32: it is the largest array by far and only feeds the
    voting methods, where float32 halves memory and time. Its rounding (~1e-7)
    can only flip pixels whose winning margin is that small.
    """
    shape = probs.shape[2:]
    if np.array_equal(medians, targets):
        return probs.astype(np.float32)
    matrix = _interpolation_matrix(transform(medians), transform(targets))
    probs = probs.reshape(len(medians), len(medians), -1)
    # columns[i, q, r] = sum_j matrix[q, j] probs[i, j, r]; small, so kept in float64
    columns = (matrix @ probs).reshape(len(medians), -1).astype(np.float32)
    # one (T, N) @ (N, T * R) product for the large result
    return (matrix.astype(np.float32) @ columns).reshape(len(targets), len(targets), *shape)

# ---------------------------------------------------------------- Arrangement

@njit(cache=True, error_model="numpy")
def _clip_kernel(poly, nx, ny, offset):
    """_clip with an empty (0, 2) array for None; compiled, since the arrangement
    of 8 candidates clips a few thousand small polygons."""
    n = poly.shape[0]
    vals = np.empty(n)
    for k in range(n):
        v = poly[k, 0] * nx + poly[k, 1] * ny - offset
        vals[k] = 0.0 if abs(v) < EPS else v
    out = np.empty((2 * n, 2))
    m = 0
    for k in range(n):
        j = (k + 1) % n
        vp, vq = vals[k], vals[j]
        if vp <= 0:
            out[m] = poly[k]
            m += 1
        if (vp < 0 < vq) or (vq < 0 < vp):
            f = vp / (vp - vq)
            out[m, 0] = poly[k, 0] + f * (poly[j, 0] - poly[k, 0])
            out[m, 1] = poly[k, 1] + f * (poly[j, 1] - poly[k, 1])
            m += 1
    # drop vertices that repeat the next one
    keep = np.empty(m, dtype=np.bool_)
    for k in range(m):
        j = (k + 1) % m
        keep[k] = math.hypot(out[k, 0] - out[j, 0], out[k, 1] - out[j, 1]) > 1e-10
    out = out[:m][keep]
    m = out.shape[0]
    area = 0.0
    for k in range(m):
        j = (k + 1) % m
        area += out[k, 0] * out[j, 1] - out[j, 0] * out[k, 1]
    return out if m >= 3 and 0.5 * area > 1e-14 else out[:0]


def _clip(poly, normal, offset):
    """Part of convex CCW polygon `poly` with normal . p <= offset, or None."""
    out = _clip_kernel(poly, float(normal[0]), float(normal[1]), float(offset))
    return out if len(out) else None


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
            # Most polygons lie on one side of the bisector (up to EPS) and stay whole;
            # one product over all vertices finds the few it crosses.
            vals = np.concatenate(polygons) @ normal - offset
            starts = np.cumsum([0] + [len(poly) for poly in polygons[:-1]])
            lowest, highest = np.minimum.reduceat(vals, starts), np.maximum.reduceat(vals, starts)
            split = []
            for poly, low, high in zip(polygons, lowest, highest):
                if high < EPS or low > -EPS:
                    split.append(poly)
                    continue
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
