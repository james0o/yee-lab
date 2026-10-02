"""A pixel at the geometric median of its voters instead of at their median along each axis.

The voters of a pixel have the pixel as their median along x and along y
(ranking_cells.py). That is a median only in the directions of the axes: a slanted line
through the pixel need not split its voters in half, so the slanted borders of a diagram
bend. The geometric median, the point with the smallest mean distance to the voters, does
not depend on the axes.

g(m) is the geometric median of the voters whose medians along the axes are
m = (m_x, m_y). A diagram of geometric medians shows at a point p the election of the
voters with g(m) = p. The voters, their shares, the methods and the margins stay as they
are; only the point an election is drawn at moves from m to g(m): regions.py traces the
borders on the grid moved by g. g depends on the voters only (deviation and spread rule),
never on the candidates, so it is computed once per model and kept.

g is the identity at the centre of the square and commutes with mirroring an axis and
with swapping the axes, as the voters do. Everywhere else it pulls a pixel towards the
centre, the more the closer the pixel is to a wall. It is one to one, but not onto the
square: the medians of a model stop half a pixel from the walls (shares.PIXELS), and g of
those lies well inside. Between that and the walls is a strip that is the geometric
median of no voters of the model; nothing is drawn there. (Medians closer to a wall would
fill the corners, but narrow the strip in the middle of a wall only slowly: with the rms
rule and deviation 0.25 from 0.052 for this half pixel, 1/600, to 0.029 for a median of
1e-6.)

Normal voters are symmetric around their pixel, so it is their geometric median too and
g is the identity.

g(m) minimises F(p) = E|V - p| over the voters V = (X, Y) of m. The expectation is a sum
over the rectangles of the voter grid of shares.py: each holds its exact share of the
voters, from the CDF, so the infinite density of a Beta with a < 1 at a wall does no
harm. A rectangle counts as its voters at their exact mean, at the distance

    sqrt(|mean - p|^2 + s^2),    s^2 the variance of a uniform cell along one axis,

which is their mean distance to second order in the size of the rectangle. It also keeps
F smooth and convex: with plain distances the voters next to a corner, half of them in a
few cells, would trap the minimum at the mean of a cell. Newton's method takes about five
steps from m. Against a voter grid with cells a quarter as wide this is within 5e-5 for
the narrowest voters of the UI (deviation 0.05), 2e-5 at deviation 0.1 and 3e-6 from 0.25
on (tests/test_geometric.py).

g is smooth in m, so it is computed at NODES x NODES Chebyshev-Lobatto points in
logit(m), as the shares are, and interpolated; a quarter of them is enough, the others
are its mirror images. The interpolant is within 4e-5 of g computed at the point itself.
"""

import math
import threading
from functools import lru_cache
from typing import Literal, get_args

import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from scipy.special import betainc, logit

from yeelab import ranking_cells, threads
from yeelab.margin.shares import Model, _voter_grid
from yeelab.ranking_cells import NODES, _interpolation_matrix, pixel_medians

# What a pixel is: the medians of its voters along the axes, or their geometric median.
PixelMedian = Literal["marginal", "geometric"]
PIXEL_MEDIANS = get_args(PixelMedian)
PIXEL_MEDIAN: PixelMedian = "marginal"

EMPTY = 1e-17      # cells at both ends of an axis with a smaller share of the voters are skipped
SPARSE = 1e-9      # a cell with a smaller share counts at its middle: its mean is rounding noise
STEPS = 60         # of Newton's method, which needs about five
TOLERANCE = 1e-12  # the last step is shorter than this
CHUNK = 16         # medians per thread

# ---------------------------------------------------------------- Kernel


@njit(cache=True, nogil=True, error_model="numpy")
def _minimise(rows, mass, mean, variance, start, out):
    """out[n] = the point with the smallest mean distance to the voters of
    rows[n] = (i, j): mass[i, k] * mass[j, l] of them at (mean[i, k], mean[j, l]), spread
    by (variance[k] + variance[l]) / 2 around it. Started at start[n]; NaN if Newton's
    method does not settle."""
    cells = variance.size
    for n in range(rows.shape[0]):
        wx, wy = mass[rows[n, 0]], mass[rows[n, 1]]
        x, y = mean[rows[n, 0]], mean[rows[n, 1]]
        k0, k1, l0, l1 = 0, cells, 0, cells
        while wx[k0] < EMPTY:
            k0 += 1
        while wx[k1 - 1] < EMPTY:
            k1 -= 1
        while wy[l0] < EMPTY:
            l0 += 1
        while wy[l1 - 1] < EMPTY:
            l1 -= 1
        px, py = start[n, 0], start[n, 1]
        out[n, 0] = out[n, 1] = np.nan
        for _ in range(STEPS):
            # gradient (gx, gy) and Hessian of F at (px, py); `weight` is E 1 / distance
            gx = gy = hxx = hxy = hyy = weight = 0.0
            for k in range(k0, k1):
                dx = px - x[k]
                sx = sy = sxx = sxy = syy = s = 0.0
                for l in range(l0, l1):
                    dy = py - y[l]
                    r2 = dx * dx + dy * dy + 0.5 * (variance[k] + variance[l])
                    w = wy[l] / math.sqrt(r2)
                    s += w
                    sx += w * dx
                    sy += w * dy
                    w /= r2
                    sxx += w * (r2 - dx * dx)
                    sxy -= w * dx * dy
                    syy += w * (r2 - dy * dy)
                gx += wx[k] * sx
                gy += wx[k] * sy
                hxx += wx[k] * sxx
                hxy += wx[k] * sxy
                hyy += wx[k] * syy
                weight += wx[k] * s
            det = hxx * hyy - hxy * hxy
            step_x = (hxy * gy - hyy * gx) / det
            step_y = (hxy * gx - hxx * gy) / det
            if not (0.0 < px + step_x < 1.0 and 0.0 < py + step_y < 1.0):
                # Weiszfeld's step: shorter, and it never leaves the square
                step_x, step_y = -gx / weight, -gy / weight
            px += step_x
            py += step_y
            if abs(step_x) + abs(step_y) < TOLERANCE:
                out[n, 0], out[n, 1] = px, py
                break

# ---------------------------------------------------------------- g at given medians


def _cell_means(model: Model, medians, lines, mass):
    """Mean of the voters with each of the `medians` (M,) within each cell of the voter
    grid along one axis, shape (M, K): the first moment of Beta(a, b) between two lines
    is a / (a + b) times the share of Beta(a + 1, b) between them."""
    params = ranking_cells.beta_params_at(medians, model.deviation, model.spread)
    a, b = params[:, :1], params[:, 1:]
    first = a / (a + b) * np.diff(betainc(a + 1, b, lines), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = first / mass
    inside = (mass > SPARSE) & (mean > lines[:-1]) & (mean < lines[1:])
    return np.where(inside, mean, 0.5 * (lines[1:] + lines[:-1]))


def _solve(model: Model, medians, rows):
    """g at the medians (medians[i], medians[j]) for each (i, j) of `rows`, shape
    (len(rows), 2): computed there, on the threads of the pool."""
    medians = np.ascontiguousarray(medians, dtype=np.float64)
    rows = np.ascontiguousarray(rows, dtype=np.intp)
    lines, mass = _voter_grid(model, medians)
    mean = _cell_means(model, medians, lines, mass)
    variance = np.diff(lines) ** 2 / 12
    start, out = medians[rows], np.empty((len(rows), 2))
    threads.in_chunks(
        lambda a, b: _minimise(rows[a:b], mass, mean, variance, start[a:b], out[a:b]),
        len(rows), CHUNK,
    )
    if np.isnan(out).any():
        raise RuntimeError("no geometric median found for some voters")
    return out


def geometric_median_at(model: Model, points) -> np.ndarray:
    """g at each of the `points` (K, 2), the medians of voters along the axes, shape
    (K, 2): computed at the points themselves, not interpolated."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if model.distribution != "beta":
        return points.copy()
    medians, rows = np.unique(points, return_inverse=True)
    return _solve(model, medians, rows.reshape(points.shape))

# ---------------------------------------------------------------- g from the nodes


@lru_cache(maxsize=32)  # 38 kB each, built in ~0.2 s
def _node_table(model: Model):
    medians = ranking_cells.node_medians(model.pixels, NODES)
    size = len(medians)
    lower = (size + 1) // 2  # nodes up to 1/2: the others are their mirror images
    k, l = np.tril_indices(lower)
    solved = _solve(model, medians, np.column_stack([k, l]))
    quarter = np.empty((lower, lower, 2))
    quarter[k, l], quarter[l, k] = solved, solved[:, ::-1]  # g(y, x) is g(x, y) swapped
    table = np.empty((size, size, 2))
    table[:lower, :lower] = quarter
    table[size - lower:, :lower] = quarter[::-1] * [-1, 1] + [1, 0]  # g(1 - x, y)
    table[:, size - lower:] = table[:, lower - 1::-1] * [1, -1] + [0, 1]  # g(x, 1 - y)
    medians.setflags(write=False)
    table.setflags(write=False)
    return medians, table


_table_lock = threading.Lock()


def node_table(model: Model):
    """(node medians (N,), g at them (N, N, 2)), read-only and built once: the lock keeps
    the threads that ask first from each building their own."""
    with _table_lock:
        return _node_table(model)


def geometric_medians(model: Model, medians) -> np.ndarray:
    """g[i, j] = the geometric median of the voters with the medians
    (medians[i], medians[j]) along the axes, shape (M, M, 2), interpolated from the
    nodes. The `medians` are within those of the model: at least half a pixel from the
    walls."""
    medians = np.asarray(medians, dtype=np.float64)
    if model.distribution != "beta":
        return np.stack(np.meshgrid(medians, medians, indexing="ij"), axis=-1)
    nodes, table = node_table(model)
    matrix = _interpolation_matrix(logit(nodes), logit(medians))
    return np.stack([matrix @ table[..., axis] @ matrix.T for axis in (0, 1)], axis=-1)

# ---------------------------------------------------------------- The inverse of g


def pixels_at(model: Model) -> np.ndarray:
    """The inverse of g for the pixels: at[i, j] = k * pixels + l is the pixel (k, l)
    whose voters have their geometric median closest to the centre of the pixel (i, j),
    or -1 where that centre is the geometric median of no voters of the model (the strip
    along the walls). Shape (pixels, pixels); symmetric like g.

    The centre is one of someone's geometric medians if it is not beyond the image of a
    wall: the curve g follows as the median runs along the outermost pixels of one
    side."""
    centres = pixel_medians(model.pixels)
    g = geometric_medians(model, centres)
    x, y = np.meshgrid(centres, centres, indexing="ij")
    _, nearest = cKDTree(g.reshape(-1, 2)).query(np.column_stack([x.ravel(), y.ravel()]))
    nearest = nearest.reshape(x.shape)
    inside = (
        (x >= np.interp(y, g[0, :, 1], g[0, :, 0])) & (x <= np.interp(y, g[-1, :, 1], g[-1, :, 0]))
        & (y >= np.interp(x, g[:, 0, 0], g[:, 0, 1])) & (y <= np.interp(x, g[:, -1, 0], g[:, -1, 1]))
    )
    return np.where(inside, nearest, -1)
