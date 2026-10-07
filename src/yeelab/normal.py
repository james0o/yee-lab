"""Normally distributed voters (original Yee diagrams).

The counterpart of the Beta model of ranking_cells.py: the voters of the pixel with
centre m = (m_x, m_y) are

    (X, Y) ~ N(m, sigma^2 I),

not restricted to the unit square. sigma comes from the same `deviation` as the
Beta model, the mean absolute deviation from the median of each coordinate:
E|X - m_x| = sigma sqrt(2 / pi).

Why Condorcet methods draw the Voronoi diagram here: the distribution is
symmetric about m, so every line through m has half of the voters on each side.
A voter prefers c_i to c_j on c_i's side of their bisector, so a majority prefers
c_i to c_j exactly when m is closer to c_i. The pairwise majorities rank the
candidates by distance from m: there is never a Condorcet cycle and the
Condorcet winner is the candidate nearest to m, for any sigma.
The Beta median halves the voters only along lines parallel to the axes, so
there the majorities between candidates on a diagonal differ from the distance.

Cells: the same bisector arrangement as ranking_cells.py, but voters can leave
the unit square, so it is built in the box [-L, 1 + L]^2, L = 10 sigma, which
holds all but ~1e-23 of the voters of every pixel. Scaling the box onto the unit
square maps bisectors to bisectors, so ranking_cells() builds it unchanged.

Cell probabilities are exact, without quadrature. A convex CCW cell is the
signed sum of the triangles (m, s, e) over its edges s -> e. Split at the foot
of the perpendicular from m, each is a signed difference of right triangles with
one leg d from m to the edge line and the other leg t along it, and

    P(right triangle) = R(d, t) = arctan(t / d) / (2 pi) - T(d / sigma, t / d),

where the wedge {0 < y < (t / d) x} holds arctan(t / d) / (2 pi) of the voters
and Owen's T function T(h, a) is the part of it beyond x = h (in units of sigma).

Owen's T is compiled (numba) rather than scipy's owens_t, so that the triangles of
many edges run in one call without the GIL. For 0 <= a <= 1 it is its integral

    T(h, a) = a / (2 pi) int_0^1 exp(-h^2 (1 + a^2 x^2) / 2) / (1 + a^2 x^2) dx

with 12-point Gauss-Legendre (the integrand is analytic; 12 points agree with scipy
to 1e-16 for all h), and for a > 1 the identity
T(h, a) = (Phi(h) + Phi(a h)) / 2 - Phi(h) Phi(a h) - T(a h, 1 / a) reduces it to that.
"""

import math
import os

import numpy as np
from numba import njit

from yeelab import threads
from yeelab.ranking_cells import (
    NODES,
    effective_nodes,
    pixel_medians,
    ranking_cells,
)

BOX = 10  # half-width of the margin around the unit square, in sigma
# 12-point Gauss-Legendre on [0, 1] for Owen's T
_T_NODES, _T_WEIGHTS = np.polynomial.legendre.leggauss(12)
_T_NODES, _T_WEIGHTS = 0.5 * (_T_NODES + 1), 0.5 * _T_WEIGHTS


def sigma_from_deviation(deviation):
    """sigma with E|X - m| = `deviation` for X ~ N(m, sigma^2)."""
    return deviation * np.sqrt(np.pi / 2)

# ---------------------------------------------------------------- Interpolation

def node_medians(pixels, nodes=NODES):
    """Means at which probabilities are computed exactly: `nodes`
    Chebyshev-Lobatto points in the mean itself spanning the pixel centres,
    or the pixel centres if effective_nodes(...) is 0. Unlike the Beta model,
    nothing changes faster near the edges, so the points are not in logit."""
    nodes = effective_nodes(pixels, nodes)
    if nodes == 0:
        return pixel_medians(pixels)
    t = np.sin(np.pi * (2 * np.arange(nodes) - (nodes - 1)) / (2 * (nodes - 1)))
    return 0.5 + (0.5 - 0.5 / pixels) * t

# ---------------------------------------------------------------- Probabilities

def normal_cells(candidates, sigma):
    """ranking_cells(candidates) in the box [-L, 1 + L]^2, L = BOX * sigma."""
    margin = BOX * sigma
    size = 1 + 2 * margin
    polygons, rankings = ranking_cells((np.asarray(candidates) + margin) / size)
    return [poly * size - margin for poly in polygons], rankings


@njit(inline="always")
def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(inline="always")
def _owens_t_unit(h, a, nodes, weights):
    """T(h, a) for 0 <= a <= 1, by quadrature of its integral."""
    total = 0.0
    for q in range(nodes.size):
        r = 1.0 + (a * nodes[q]) ** 2
        total += weights[q] * math.exp(-0.5 * h * h * r) / r
    return a * total / (2 * math.pi)


@njit(inline="always")
def _right_triangle(d, t, sigma, nodes, weights):
    """R(d, t), signed like d * t; zero where d = 0 (m on the edge line)."""
    if d == 0.0:
        return 0.0
    a = t / d
    h, b = abs(d / sigma), abs(a)  # T is even in h and odd in a
    if b <= 1.0:
        T = _owens_t_unit(h, b, nodes, weights)
    else:
        ph, pbh = _phi(h), _phi(b * h)
        T = 0.5 * (ph + pbh) - ph * pbh - _owens_t_unit(b * h, 1.0 / b, nodes, weights)
    return math.atan(a) / (2 * math.pi) - math.copysign(T, a)


@njit(cache=True, nogil=True, error_model="numpy")
def _triangle_terms(segments, medians, sigma, nodes, weights, out):
    """out[e, i, j] = signed probability of the triangle (m, s, e) for segment e =
    (xs, ys, xe, ye) and m = (medians[i], medians[j]); positive when m is left of s -> e."""
    for e in range(segments.shape[0]):
        xs, ys, xe, ye = segments[e, 0], segments[e, 1], segments[e, 2], segments[e, 3]
        length = math.hypot(xe - xs, ye - ys)
        ux, uy = (xe - xs) / length, (ye - ys) / length
        for i in range(medians.size):
            for j in range(medians.size):
                d = (xs - medians[i]) * uy - (ys - medians[j]) * ux  # > 0 when m is left
                ts = (xs - medians[i]) * ux + (ys - medians[j]) * uy  # s on the line, 0 at the foot
                out[e, i, j] = (_right_triangle(d, ts + length, sigma, nodes, weights)
                                - _right_triangle(d, ts, sigma, nodes, weights))


def triangle_terms(segments, medians, sigma):
    """Signed triangle probabilities (m, s, e) of the segments (E, 4) = (xs, ys, xe, ye),
    shape (E, N, N): [e, i, j] has mean (medians[i], medians[j]). Chunks of the edges
    run in parallel threads."""
    segments = np.ascontiguousarray(segments, dtype=np.float64).reshape(-1, 4)
    medians = np.ascontiguousarray(medians, dtype=np.float64)
    out = np.empty((len(segments), len(medians), len(medians)))
    chunk = max(1, -(-len(segments) // (os.cpu_count() or 1)))
    threads.in_chunks(lambda a, b: _triangle_terms(segments[a:b], medians, sigma, _T_NODES,
                                                   _T_WEIGHTS, out[a:b]), len(segments), chunk)
    return out
