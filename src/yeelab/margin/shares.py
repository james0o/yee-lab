"""Voter shares that a voting method needs, fast enough to follow a dragged candidate.

The ranking probabilities of pixels/beta.py and pixels/normal.py are a complete profile,
enough for every method, but they need all cells of the bisector arrangement:
O(C^4) edges for C candidates (468 slanted edges for 8). Most methods need far less:

    borda, baldwin, nanson, schulze,  pairwise shares d[c, e] = P(c ranked above e),
    condorcet, minimax, black         one half-plane per pair (C (C - 1) / 2 edges)
    fptp                              first-choice shares, the C Voronoi cells
    koth                              both of these: the king by first choices, its
                                      challengers by pairwise shares
    irv                               the whole profile (first choices among every
                                      remaining set)
    king_runoff                       all three: koth against irv, the duel by pairwise
                                      shares
    approval                          approval shares at a threshold: the cells of one
                                      closest and one farthest candidate, each cut by
                                      one line per candidate
    approval_gap                      approval shares at the largest gap: the cells of
                                      the bisector arrangement, each cut where two of
                                      its gaps are equal

(Borda needs only pairwise shares: a ballot gives c one point per candidate ranked
below c, so c's expected score is sum_e d[c, e].)

Every share of a polygon is a sum over its edges: Green's theorem edge integrals
(pixels.beta._edge_integral, compiled on the tabulated CDFs of beta_tables.py) for
Beta voters, signed triangles with Owen's T (normal.triangle_terms) for normal voters. An edge
term depends only on the edge, so they are cached by their endpoints: dragging one
candidate moves only its C - 1 bisectors, and only edges on those are new.
For normal voters a half-plane holds Phi(signed distance / sigma) of the voters, so
their pairwise shares need no edges at all.

Everything here is at the interpolation nodes, shape (N, N, ...); regions.py
interpolates it to the points where the methods are evaluated.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from scipy.special import logit, ndtr

from yeelab import normal, ranking_cells
from yeelab.approval import GAP, Cut, half_plane
from yeelab.distributions import Distribution
from yeelab.margin.beta_tables import TabulatedBeta
from yeelab.ranking_cells import NODES, QUAD_NODES, Spread, _clip

PIXELS = 300  # the outermost medians are 1/2 and 1 - 1/2 pixel from the walls
CACHE_BYTES = 64 * 2**20  # edge integrals kept between requests
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


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


def _edge_terms(model, segments):
    """Edge terms of the segments [(start, end), ...], each (N, N)."""
    if model.distribution == "beta":
        return beta_tables(model).edge_terms(np.reshape(segments, (-1, 4)), *_QUADRATURE)
    return normal.triangle_terms(np.reshape(segments, (-1, 4)), model.medians, model.sigma)


def edge_integrals(model: Model, edges):
    """(sign, term) of the segments `edges` [(start, end), ...]: the term (N, N) of the
    canonical edge from the cache, the missing ones computed and cached; the
    segment's term is sign * term."""
    keyed = [_edge_key(np.asarray(s, dtype=np.float64), np.asarray(e, dtype=np.float64))
             for s, e in edges]
    unique = {key: (start, end) for key, _, start, end in keyed}
    values = {key: edge_cache.get((model, key)) for key in unique}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        for key, value in zip(missing, _edge_terms(model, [unique[key] for key in missing])):
            edge_cache.put((model, key), value)
            values[key] = value
    return [(sign, values[key]) for key, sign, _, _ in keyed]


def _polygon_shares(model: Model, polygons):
    """Voter share of each convex CCW polygon (None = empty), shape (N, N, P)."""
    edges, owner = [], []
    for p, poly in enumerate(polygons):
        if poly is None:
            continue
        for k in range(len(poly)):
            edges.append((poly[k], poly[(k + 1) % len(poly)]))
            owner.append(p)
    shares = np.zeros((len(polygons), model.nodes, model.nodes))  # contiguous per polygon
    for p, (sign, term) in zip(owner, edge_integrals(model, edges)):
        if sign > 0:
            shares[p] += term
        else:
            shares[p] -= term
    return np.clip(np.moveaxis(shares, 0, -1), 0.0, 1.0)

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


def _voter_box(model: Model) -> np.ndarray:
    """Where the voters are: the unit square, or for normal voters, who can leave it,
    the box of normal.normal_cells."""
    if model.distribution == "beta":
        return SQUARE
    margin = normal.BOX * model.sigma
    return np.array([[-margin, -margin], [1 + margin, -margin],
                     [1 + margin, 1 + margin], [-margin, 1 + margin]])


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
    return _polygon_shares(model, voronoi_cells(candidates, _voter_box(model)))


def ranking_shares(candidates, model: Model):
    """(rankings (R, C), shares (N, N, R)) as pixels.beta / pixels.normal
    compute_ranking_probabilities return them: the cells of the bisector arrangement,
    with their edges from the edge cache."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if model.distribution == "beta":
        polygons, rankings = ranking_cells.ranking_cells(candidates)
    else:
        polygons, rankings = normal.normal_cells(candidates, model.sigma)
    return rankings, _polygon_shares(model, polygons)

# ---------------------------------------------------------------- Approval


def _threshold_polygons(candidates, threshold, box):
    """(polygons, approves) of the approval ballots at a threshold (yeelab.approval):
    convex CCW polygons in `box` and, for each, the candidates that every voter in it
    approves. A voter's ballot depends only on its closest candidate a and farthest b,
    so each cell of one a and one b (a Voronoi cell cut by the bisectors of b) approves
    a, and its part on candidate i's side of i's line approves i."""
    n = len(candidates)
    polygons, approves = [], []
    for a, nearest in enumerate(voronoi_cells(candidates, box)):
        for b in range(n):
            cell = nearest if b != a else None
            for e in range(n):
                if e not in (a, b) and cell is not None:
                    cell = _half_plane(candidates, e, b, cell)
            if cell is None:
                continue
            polygons.append(cell)
            approves.append([a])
            for i in range(n):
                if i in (a, b):
                    continue
                weights = np.zeros(n)
                weights[i], weights[a], weights[b] = 1.0, -threshold, threshold - 1.0
                part = _clip(cell, *half_plane(candidates, weights))
                if part is not None:
                    polygons.append(part)
                    approves.append([i])
    return polygons, approves


def _gap_polygons(candidates, cells, rankings):
    """(polygons, approves) of the approval ballots at the largest gap, like
    _threshold_polygons: each cell of the bisector arrangement (`cells`, `rankings` of
    ranking_cells) split into the parts where gap k of its ranking is the largest, which
    approve the first k + 1 candidates of the ranking.

    The gaps are linear within a cell, so one that is no larger than another at every
    vertex is so everywhere: most cells have one largest gap and are not cut at all."""
    n = len(candidates)
    squares = (candidates ** 2).sum(axis=-1)
    polygons, approves = [], []
    for cell, ranking in zip(cells, rankings):
        ranking = ranking.tolist()
        gaps = np.zeros((n - 1, n))  # gap k = u[ranking[k]] - u[ranking[k + 1]], as weights
        gaps[np.arange(n - 1), ranking[:-1]] = 1.0
        gaps[np.arange(n - 1), ranking[1:]] = -1.0
        normals, offsets = -2 * gaps @ candidates, -gaps @ squares  # approval.half_plane of each
        values = offsets[:, None] - normals @ cell.T  # gap k at vertex v
        floor = values.min(axis=1).max()  # the largest gap is at least this everywhere
        for k in range(n - 1):
            if values[k].max() < floor:
                continue
            part = cell
            for other in range(n - 1):
                if other == k or part is None or (values[k] >= values[other]).all():
                    continue
                part = _clip(part, normals[k] - normals[other], offsets[k] - offsets[other])
            if part is not None:
                polygons.append(part)
                approves.append(ranking[:k + 1])
    return polygons, approves


def _summed_shares(model: Model, polygons, approves, n):
    """Share of the voters who approve each of n candidates, (N, N, n): for candidate
    c, the share of the union of the polygons whose `approves` lists c.

    The edge terms are added per candidate, with the sign of each polygon's direction
    along the edge. An edge between two polygons that approve the same candidates
    cancels and takes no integral, which leaves the borders of each candidate's region."""
    counts, ends = {}, {}
    for poly, who in zip(polygons, approves):
        for k in range(len(poly)):
            key, sign, start, end = _edge_key(poly[k], poly[(k + 1) % len(poly)])
            if key not in counts:
                counts[key], ends[key] = np.zeros(n), (start, end)
            counts[key][who] += sign
    keys = [key for key, count in counts.items() if count.any()]
    shares = np.zeros((n, model.nodes, model.nodes))  # contiguous per candidate
    for key, (sign, term) in zip(keys, edge_integrals(model, [ends[key] for key in keys])):
        for c in np.flatnonzero(counts[key]):
            shares[c] += sign * counts[key][c] * term
    return np.clip(np.moveaxis(shares, 0, -1), 0.0, 1.0)


def approval_polygons(candidates, model: Model, cut: Cut):
    """(polygons, approves): where the voters of `model` are, cut into convex CCW
    polygons, and the candidates every voter of each polygon approves at `cut`
    (yeelab.approval). A polygon need not list all of them: polygons overlap."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if cut != GAP:
        return _threshold_polygons(candidates, cut, _voter_box(model))
    if model.distribution == "beta":
        return _gap_polygons(candidates, *ranking_cells.ranking_cells(candidates))
    return _gap_polygons(candidates, *normal.normal_cells(candidates, model.sigma))


def approval_shares(candidates, model: Model, cut: Cut) -> np.ndarray:
    """approved[i, j, c] = share of the voters of node (i, j) who approve c at `cut`
    (a threshold or approval.GAP), shape (N, N, C). They do not sum to 1: a voter
    approves between one candidate and all but one."""
    polygons, approves = approval_polygons(candidates, model, cut)
    return _summed_shares(model, polygons, approves, len(candidates))
