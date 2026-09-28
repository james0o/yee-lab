"""Ranking probabilities for normally distributed voters (original Yee diagrams).

Alternative to the Beta model of ranking_cells.py with the same interface: the
voters of the pixel with centre m = (m_x, m_y) are

    (X, Y) ~ N(m, sigma^2 I),

not restricted to the unit square. sigma comes from the same `deviation` as the
Beta model, the mean absolute deviation from the median of each coordinate:
E|X - m_x| = sigma sqrt(2 / pi).

Why Condorcet methods draw the Voronoi diagram here: the distribution is
symmetric about m, so every line through m has half of the voters on each side.
A voter prefers c_i to c_j on c_i's side of their bisector, so a majority prefers
c_i to c_j exactly when m is closer to c_i. The pairwise majorities rank the
candidates by distance from m: there is never a Condorcet cycle and the
Condorcet winner is the candidate nearest to m (methods.voronoi), for any sigma.
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
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import owens_t

from cache import (
    DEFAULT_CACHE_ROOT,
    candidate_hash,
    load_node_probabilities,
    metadata,
    save_node_probabilities,
    value_token,
)
from const import CANDIDATES, DEVIATION, PIXELS
from ranking_cells import (
    NODES,
    effective_nodes,
    pixel_medians,
    ranking_cells,
)
from ranking_cells import interpolate_to_pixels as _interpolate_to_pixels

BOX = 10  # half-width of the margin around the unit square, in sigma


def sigma_from_deviation(deviation=DEVIATION):
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


def interpolate_to_pixels(probs, medians, pixels):
    """Probabilities (pixels, pixels, R) from probabilities (N, N, R) at node_medians."""
    return _interpolate_to_pixels(probs, medians, pixels, transform=lambda m: m)

# ---------------------------------------------------------------- Probabilities

def normal_cells(candidates, sigma):
    """ranking_cells(candidates) in the box [-L, 1 + L]^2, L = BOX * sigma."""
    margin = BOX * sigma
    size = 1 + 2 * margin
    polygons, rankings = ranking_cells((np.asarray(candidates) + margin) / size)
    return [poly * size - margin for poly in polygons], rankings


def _right_triangles(d, t, sigma):
    """R(d, t), signed like d * t; zero where d = 0 (m on the edge line)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = t / d
        r = np.arctan(a) / (2 * np.pi) - owens_t(d / sigma, a)
    return np.where(d == 0, 0.0, r)


def _triangle(s, e, medians, sigma):
    """Signed probability of the triangle (m, s, e), positive when m is left of
    s -> e. Shape (N, N): [i, j] has mean (medians[i], medians[j])."""
    length = np.linalg.norm(e - s)
    ux, uy = (e - s) / length
    x, y = medians[:, None], medians[None, :]
    d = (s[0] - x) * uy - (s[1] - y) * ux  # distance of m from the line, > 0 on its left
    ts = (s[0] - x) * ux + (s[1] - y) * uy  # position of s along the line, 0 at the foot
    return _right_triangles(d, ts + length, sigma) - _right_triangles(d, ts, sigma)


def compute_ranking_probabilities(candidates, medians, deviation=DEVIATION, progress=None):
    """Returns (rankings (R, C), probabilities (N, N, R)); [i, j] is the pixel with
    mean (medians[i], medians[j]).

    progress: optional callable(done, total), called after each cell.
    """
    sigma = sigma_from_deviation(deviation)
    polygons, rankings = normal_cells(candidates, sigma)
    medians = np.asarray(medians, dtype=np.float64)

    def cell_probability(poly):
        edges = zip(poly, np.roll(poly, -1, axis=0))
        return sum(_triangle(s, e, medians, sigma) for s, e in edges)

    probs = np.empty((len(medians), len(medians), len(polygons)))
    # owens_t releases the GIL, so cells are computed in parallel.
    with ThreadPoolExecutor() as pool:
        for r, cell in enumerate(pool.map(cell_probability, polygons)):
            probs[..., r] = cell
            if progress is not None:
                progress(r + 1, len(polygons))
    return rankings, np.clip(probs, 0.0, 1.0)


def ranking_probabilities(candidates, pixels=PIXELS, deviation=DEVIATION, nodes=NODES,
                          progress=None):
    """(rankings, probabilities (pixels, pixels, R)) computed at the node medians
    and interpolated, without the cache."""
    medians = node_medians(pixels, nodes)
    rankings, probs = compute_ranking_probabilities(candidates, medians, deviation, progress)
    return rankings, interpolate_to_pixels(probs, medians, pixels)

# ---------------------------------------------------------------- Cache

def rankings_path(pixels, deviation, candidates, nodes, cache_root=DEFAULT_CACHE_ROOT):
    name = (f"P{pixels}_D{value_token(deviation)}_N{effective_nodes(pixels, nodes)}"
            f"_C{candidate_hash(candidates)}.npz")
    return Path(cache_root) / "normal" / name


def _rankings_metadata(pixels, deviation, candidates, nodes):
    return metadata(
        "rankings",
        distribution="normal",
        pixels=int(pixels),
        deviation=float(deviation),
        nodes=effective_nodes(pixels, nodes),
        candidate_hash=candidate_hash(candidates),
    )


def read_cached_ranking_probabilities(
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
    nodes=NODES,
    cache_root=DEFAULT_CACHE_ROOT,
):
    """(rankings, probabilities) from the cache, or None if not cached."""
    candidates = np.asarray(candidates, dtype=np.float64)
    path = rankings_path(pixels, deviation, candidates, nodes, cache_root)
    saved = load_node_probabilities(
        path, _rankings_metadata(pixels, deviation, candidates, nodes), candidates
    )
    if saved is None:
        return None
    rankings, probs, medians = saved
    return rankings, interpolate_to_pixels(probs, medians, pixels)


def generate_ranking_probabilities(
    candidates=CANDIDATES,
    pixels=PIXELS,
    deviation=DEVIATION,
    nodes=NODES,
    cache_root=DEFAULT_CACHE_ROOT,
    progress=None,
):
    """Compute (rankings, probabilities) and save the node probabilities to the cache."""
    candidates = np.asarray(candidates, dtype=np.float64)
    medians = node_medians(pixels, nodes)
    rankings, probs = compute_ranking_probabilities(candidates, medians, deviation, progress)
    path = rankings_path(pixels, deviation, candidates, nodes, cache_root)
    save_node_probabilities(
        path, _rankings_metadata(pixels, deviation, candidates, nodes),
        rankings, probs, medians, candidates,
    )
    return rankings, interpolate_to_pixels(probs, medians, pixels)
