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
    approval, approval_gap            approval shares at their cut: the voters who approve
                                      a candidate have curved borders, so these come
                                      from a grid of voters, not from polygons

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
from scipy.special import betainc, logit, ndtr

from yeelab import normal, ranking_cells
from yeelab.approval import Cut, coverage
from yeelab.distributions import Distribution
from yeelab.margin.beta_tables import TabulatedBeta
from yeelab.ranking_cells import NODES, QUAD_NODES, Spread, _clip

PIXELS = 300  # the outermost medians are 1/2 and 1 - 1/2 pixel from the walls
CACHE_BYTES = 64 * 2**20  # edge integrals kept between requests
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
# The voter grid of the approval shares (_grid_lines): cells per axis across the unit
# square, ballots per axis in a cell that a border crosses, the smallest cell at a wall
# (Beta voters) and the growth of the cells beyond the square (normal voters).
APPROVAL_CELLS = 256
APPROVAL_SUB = 8
WALL_CELL = 1e-6
OUTSIDE_GROWTH = 1.1


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


def _grid_lines(model: Model) -> np.ndarray:
    """Lines of the voter grid along one axis, increasing: APPROVAL_CELLS equal cells
    across the unit square, and smaller or larger ones where the voters call for them.

    Beta voters: the cell at each wall is halved again and again down to WALL_CELL. The
    density of a Beta with a < 1 is infinite at the wall, and the voters of a pixel next
    to a wall sit within a small fraction of an equal cell.
    Normal voters: beyond the square the cells grow by OUTSIDE_GROWTH each, out to the
    box of normal.normal_cells, where the density is smooth and no candidate is near."""
    step = 1.0 / APPROVAL_CELLS
    inside = np.linspace(0.0, 1.0, APPROVAL_CELLS + 1)
    if model.distribution == "beta":
        halves = step / 2.0 ** np.arange(1, 1 + int(np.ceil(np.log2(step / WALL_CELL))))
        return np.concatenate([[0.0], halves[::-1], inside[1:-1], 1 - halves, [1.0]])
    outside, reach = [], normal.BOX * model.sigma
    while not outside or outside[-1] < reach:
        step *= OUTSIDE_GROWTH
        outside.append((outside[-1] if outside else 0.0) + step)
    outside = np.array(outside)
    return np.concatenate([-outside[::-1], inside, 1 + outside])


@lru_cache(maxsize=8)  # < 1 MB each, built in a few ms
def _voter_grid(model: Model):
    """(lines (K + 1,), mass (N, K)) of the voter grid: its lines along one axis
    (_grid_lines) and the share of each node's voters between two neighbouring lines,
    from the exact CDF. Both axes have the same lines and nodes, so a rectangle of the
    grid holds mass[i, k] * mass[j, l] of the voters of node (i, j). Read-only."""
    lines = _grid_lines(model)
    if model.distribution == "beta":
        params = ranking_cells.node_params(model.pixels, model.nodes, model.deviation, model.spread)[1]
        cdf = betainc(params[:, :1], params[:, 1:], lines)
    else:
        cdf = ndtr((lines - model.medians[:, None]) / model.sigma)
    mass = np.diff(cdf, axis=1)
    lines.setflags(write=False)
    mass.setflags(write=False)
    return lines, mass


def approval_shares(candidates, model: Model, cut: Cut) -> np.ndarray:
    """approved[i, j, c] = share of the voters of node (i, j) who approve c at `cut`
    (approval.HALF or GAP), shape (N, N, C). They do not sum to 1: a voter approves
    between one candidate and all but one.

    The voters who approve a candidate have curved borders (yeelab.approval), so their
    share is not a sum of edge terms like the others. The plane is cut into the
    rectangles of the voter grid instead: approval.coverage gives the part of each
    rectangle that approves c, and the rectangles are added up with the exact share of
    the voters each one holds. The approving region is the same for every node, and only
    its border within a rectangle is approximate, so the shares stay smooth in the
    node; they are within about 1e-3 of the exact ones (tests/test_approval.py)."""
    lines, mass = _voter_grid(model)
    cover = coverage(lines, lines, candidates, cut, APPROVAL_SUB)
    return np.moveaxis(mass @ cover.astype(np.float64) @ mass.T, 0, -1)

