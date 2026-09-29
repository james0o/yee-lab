"""Voter shares that a voting method needs, fast enough to follow a dragged candidate.

The ranking probabilities of ranking_cells.py and normal.py are a complete profile,
enough for every method, but they need all cells of the bisector arrangement:
O(C^4) edges for C candidates (468 slanted edges for 8). Most methods need far less:

    borda, condorcet_cycle, schulze   pairwise shares d[c, e] = P(c ranked above e),
                                      one half-plane per pair (C (C - 1) / 2 edges)
    fptp                              first-choice shares, the C Voronoi cells
    irv                               the whole profile (first choices among every
                                      remaining set)

(Borda needs only pairwise shares: a ballot gives c one point per candidate ranked
below c, so c's expected score is sum_e d[c, e].)

Every share of a polygon is a sum over its edges: Green's theorem edge integrals
(ranking_cells._edge_integral) with the tabulated CDFs of beta_tables.py for Beta
voters, signed triangles with Owen's T (normal._triangle) for normal voters. An edge
term depends only on the edge, so they are cached by their endpoints: dragging one
candidate moves only its C - 1 bisectors, and only edges on those are new.
For normal voters a half-plane holds Phi(signed distance / sigma) of the voters, so
their pairwise shares need no edges at all.

Everything here is at the interpolation nodes, shape (N, N, ...); regions.py
interpolates it to the points where the methods are evaluated.
"""

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from scipy.special import logit, ndtr

import normal
import ranking_cells
from beta_tables import TabulatedBeta
from distributions import Distribution
from ranking_cells import NODES, QUAD_NODES, Spread, _clip, _edge_integral

PIXELS = 300  # the outermost medians are 1/2 and 1 - 1/2 pixel from the walls
CACHE_BYTES = 64 * 2**20  # edge integrals kept between requests
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

_pool = ThreadPoolExecutor(max_workers=os.cpu_count())


def node_count(distribution: Distribution, deviation: float) -> int:
    """Interpolation nodes per axis. Narrow Beta voters (deviation below 0.1) change
    the shares faster than NODES follow: at 0.05 they would sum to up to 1.025."""
    return 2 * NODES - 1 if distribution == "beta" and deviation < 0.1 else NODES


@dataclass(frozen=True)
class Model:
    """Voters of every pixel: distribution, deviation, and the Beta spread rule
    (None for normal voters). Hashable, so it keys the caches."""

    distribution: Distribution
    deviation: float
    spread: Spread | None = None
    pixels: int = PIXELS

    @cached_property
    def nodes(self) -> int:
        return node_count(self.distribution, self.deviation)

    @cached_property
    def medians(self) -> np.ndarray:
        """Node medians (Beta) or means (normal), per axis."""
        if self.distribution == "beta":
            return ranking_cells.node_params(self.pixels, self.nodes, self.deviation, self.spread)[0]
        return normal.node_medians(self.pixels, self.nodes)

    def transform(self, medians):
        """The variable the nodes are Chebyshev points in (see node_medians)."""
        return logit(medians) if self.distribution == "beta" else medians

    @property
    def sigma(self) -> float:
        return normal.sigma_from_deviation(self.deviation)


@lru_cache(maxsize=4)  # ~11 MB each at 49 nodes, built in ~0.2 s
def _beta_tables(model: Model) -> TabulatedBeta:
    params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
    return TabulatedBeta(params)


_tables_lock = threading.Lock()


def beta_tables(model: Model) -> TabulatedBeta:
    """CDF tables of the model's nodes, built once: the lock keeps the threads that
    ask for them first from each building their own."""
    with _tables_lock:
        return _beta_tables(model)

# ---------------------------------------------------------------- Edge cache


class _EdgeCache:
    """Edge terms (N, N) by (model, start, end), least recently used dropped
    first once they take more than CACHE_BYTES. `misses` counts computed edges."""

    def __init__(self, max_bytes):
        self.max_bytes, self.bytes, self.misses = max_bytes, 0, 0
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key, value):
        with self._lock:
            if key in self._items:
                return
            self._items[key] = value
            self.bytes += value.nbytes
            self.misses += 1
            while self.bytes > self.max_bytes and len(self._items) > 1:
                self.bytes -= self._items.popitem(last=False)[1].nbytes

    def clear(self):
        with self._lock:
            self._items.clear()
            self.bytes = self.misses = 0


edge_cache = _EdgeCache(CACHE_BYTES)
_QUADRATURE = np.polynomial.legendre.leggauss(QUAD_NODES)


def _point_key(p):
    return round(float(p[0]), 9), round(float(p[1]), 9)


def _edge_key(start, end):
    """(canonical key, sign, canonical start, canonical end): each edge is stored
    once, with its lexicographically smaller endpoint first."""
    ks, ke = _point_key(start), _point_key(end)
    if ks <= ke:
        return (ks, ke), 1.0, start, end
    return (ke, ks), -1.0, end, start


def _integral(model, key, start, end):
    """Cached edge term of the canonical edge start -> end."""
    value = edge_cache.get((model, key))
    if value is None:
        if model.distribution == "beta":
            value = _edge_integral(start, end, beta_tables(model), *_QUADRATURE)
        else:
            value = normal._triangle(start, end, model.medians, model.sigma)
        edge_cache.put((model, key), value)
    return value


def edge_integrals(model: Model, edges):
    """Edge terms of the segments `edges` [(start, end), ...], each (N, N): from the
    cache, the missing ones computed in parallel (betainc, owens_t and most numpy
    loops release the GIL)."""
    keyed = [_edge_key(np.asarray(s, dtype=np.float64), np.asarray(e, dtype=np.float64))
             for s, e in edges]
    unique = {key: (start, end) for key, _, start, end in keyed}
    values = dict(zip(unique, _pool.map(lambda key: _integral(model, key, *unique[key]), unique)))
    return [sign * values[key] for key, sign, _, _ in keyed]


def cached_edge_integral(model: Model):
    """callable(start, end) for ranking_cells.compute_ranking_probabilities, which
    runs its own threads."""

    def integral(start, end):
        key, sign, start, end = _edge_key(start, end)
        return sign * _integral(model, key, start, end)

    return integral


def _polygon_shares(model: Model, polygons):
    """Voter share of each convex CCW polygon (None = empty), shape (N, N, P)."""
    edges, owner = [], []
    for p, poly in enumerate(polygons):
        if poly is None:
            continue
        for k in range(len(poly)):
            edges.append((poly[k], poly[(k + 1) % len(poly)]))
            owner.append(p)
    shares = np.zeros((model.nodes, model.nodes, len(polygons)))
    for p, integral in zip(owner, edge_integrals(model, edges)):
        shares[..., p] += integral
    return np.clip(shares, 0.0, 1.0)

# ---------------------------------------------------------------- Shares


def _half_plane(candidates, c, e, box=SQUARE):
    """Part of `box` closer to candidate c than to e (convex CCW, or None)."""
    normal_ = 2 * (candidates[e] - candidates[c])
    offset = candidates[e] @ candidates[e] - candidates[c] @ candidates[c]
    return _clip(box, normal_, offset)


def pairwise_shares(candidates, model: Model) -> np.ndarray:
    """d[i, j, c, e] = share of the voters of node (i, j) ranking c above e;
    d[..., c, c] = 0. Shape (N, N, C, C)."""
    candidates = np.asarray(candidates, dtype=np.float64)
    n = len(candidates)
    pairs = [(c, e) for c in range(n) for e in range(c + 1, n)]
    if model.distribution == "beta":
        above = _polygon_shares(model, [_half_plane(candidates, c, e) for c, e in pairs])
    else:
        # c above e  <=>  (e - c) . p < (|e|^2 - |c|^2) / 2, and (e - c) . p is normal
        m = model.medians
        above = np.empty((len(m), len(m), len(pairs)))
        for k, (c, e) in enumerate(pairs):
            diff = candidates[e] - candidates[c]
            threshold = (candidates[e] @ candidates[e] - candidates[c] @ candidates[c]) / 2
            mean = diff[0] * m[:, None] + diff[1] * m[None, :]
            above[..., k] = ndtr((threshold - mean) / (model.sigma * np.linalg.norm(diff)))
    d = np.zeros((*above.shape[:2], n, n))
    for k, (c, e) in enumerate(pairs):
        d[..., c, e], d[..., e, c] = above[..., k], 1 - above[..., k]
    return d


def voronoi_cells(candidates, box=SQUARE):
    """Voronoi cell of each candidate within `box` (convex CCW, or None)."""
    candidates = np.asarray(candidates, dtype=np.float64)
    cells = []
    for c in range(len(candidates)):
        cell = box
        for e in range(len(candidates)):
            if e != c and cell is not None:
                cell = _half_plane(candidates, c, e, cell)
        cells.append(cell)
    return cells


def first_choice_shares(candidates, model: Model) -> np.ndarray:
    """first[i, j, c] = share of the voters of node (i, j) whose first choice is c,
    shape (N, N, C)."""
    if model.distribution == "beta":
        return _polygon_shares(model, voronoi_cells(candidates))
    # Voters can leave the square: cells in the box of normal.normal_cells.
    margin = normal.BOX * model.sigma
    box = np.array([[-margin, -margin], [1 + margin, -margin],
                    [1 + margin, 1 + margin], [-margin, 1 + margin]])
    return _polygon_shares(model, voronoi_cells(candidates, box))


def ranking_shares(candidates, model: Model):
    """(rankings (R, C), shares (N, N, R)) as ranking_cells / normal
    compute_ranking_probabilities return them, with the edges from the edge cache."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if model.distribution == "beta":
        params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
        return ranking_cells.compute_ranking_probabilities(
            candidates, params, edge_integral=cached_edge_integral(model))
    polygons, rankings = normal.normal_cells(candidates, model.sigma)
    return rankings, _polygon_shares(model, polygons)
