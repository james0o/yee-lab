"""Win regions of a Yee diagram as polygons instead of pixels.

A method's winner is decided by comparing shares that are smooth in the median; only
the winner jumps. Every method of MARGINS (the methods built in yeelab.build) returns,
next to the winner, a margin that is continuous and 0 on every border between two
winners. So for each winner c

    psi_c = margin where c wins, -margin elsewhere

is continuous, and the region of c is {psi_c > 0}. Its border is traced by marching
squares on a grid (contourpy), with each crossing placed by linear interpolation of
psi_c along a grid edge. Two neighbouring regions get the same crossing, a / (a + b)
along the edge from the side with margin a, so they meet without gaps. The grid only
has to be fine enough to catch thin slivers; the shares come from the Chebyshev
interpolant (ranking_cells.interpolate_to), which is exact to ~1e-5 at any point. The
Score totals and score comparisons are computed at the grid points themselves. A
nonzero runoff tie weight also needs the voter-grid ballots split by ordinal order.

Polygons follow GeoJSON: an outer ring counter-clockwise, then its holes clockwise.
A region can have several polygons (FPTP flares at the walls) and holes (an island
of another winner).

A grid point need not be drawn where its median is. With pixels at the geometric median
of their voters (geometric.py) the same winners and margins are traced on the grid moved
by g: the point of the median m is drawn at g(m), and a crossing is placed along the
moved edge as before. The moved grid ends short of the walls, and so do the regions.
The Voronoi diagram is then traced on the moved grid too (nearest), so that it is drawn
only where the methods are and the two can be compared. The grid can also be moved only
part of the way, from where the usual diagram draws a point towards g(m) (`shift`): the
web UI steps through these to move the diagram between the two.
"""

from itertools import combinations

import contourpy
import numpy as np

from yeelab.build import (FIRST, METHODS, PAIRWISE, PROFILE, Ballots, Graded, Scored, ScoredPairwise, Share, Voters,
                          Winner)
from yeelab.margin.geometric import PIXEL_MEDIAN, PixelMedian, geometric_medians, rim
from yeelab.margin.shares import (
    Model,
    ScoreGrid,
    first_choice_shares,
    graded_shares,
    pairwise_shares,
    ranking_shares,
    score_comparison_shares,
    unscored_shares,
    voronoi_cells,
)
from yeelab.ranking_cells import interpolate_to

DIGITS = 6  # decimals of the vertices sent to the UI
TINY = 1e-12  # rings with less area are dropped (collapsed onto a grid point)


# method -> its winner and margin (evaluate) from the shares it needs (needs)
MARGINS = METHODS


def grid(size: int, pixels: int):
    """Contour coordinates per axis, both walls and the centres (k + 1/2) / size, and
    the medians they are evaluated at: the walls themselves are outside the model
    (a Beta median of 0 has no voters), so medians are clamped to the outermost
    pixel centres 1/2 / pixels and 1 - 1/2 / pixels, as the interpolation nodes are."""
    coords = np.concatenate([[0.0], (np.arange(size) + 0.5) / size, [1.0]])
    return coords, np.clip(coords, 0.5 / pixels, 1 - 0.5 / pixels)


def voters(needs: frozenset[Share], candidates, model: Model, size: int) -> Voters:
    """The shares in `needs` (yeelab.build.voters) at the points of grid(size),
    computed at the nodes and interpolated (score shares: at the points themselves);
    [i, j] is the point (coordinates[i], coordinates[j])."""
    medians = grid(size, model.pixels)[1]

    def interpolate(values):
        return interpolate_to(values, model.medians, medians, model.transform, model.split)

    shares = {}
    if FIRST in needs:
        shares["first"] = interpolate(first_choice_shares(candidates, model))
    if PAIRWISE in needs:  # d[e, c] = 1 - d[c, e]: interpolate one triangle
        n = len(candidates)
        c, e = np.triu_indices(n, 1)
        above = interpolate(pairwise_shares(candidates, model)[..., c, e])
        # d gathered from [0, above, 1 - above]: much faster than assigning into its last axes
        index = np.zeros((n, n), dtype=np.intp)
        index[c, e], index[e, c] = 1 + np.arange(len(c)), 1 + len(c) + np.arange(len(c))
        zero = np.zeros((*above.shape[:2], 1), dtype=above.dtype)
        values = np.concatenate([zero, above, 1 - above], axis=-1)
        shares["pairwise"] = np.take(values, index.ravel(), axis=-1).reshape(*above.shape[:2], n, n)
    if PROFILE in needs:
        shares["rankings"], node_shares = ranking_shares(candidates, model)
        shares["probs"] = interpolate(node_shares)
    # one ballot table per score ballot: by order where burying or compromising read it too
    ordered = {share.scored for share in needs if isinstance(share, Ballots)}
    scores = [share for share in needs if isinstance(share, Scored)]
    if scores:
        shares["unscored"] = {share: unscored_shares(candidates, model, share.levels, medians, share.rule,
                                                     share.delta, mu=share.mu, kappa=share.kappa,
                                                     power=share.power, votes=share.votes,
                                                     by_order=share in ordered)
                              for share in scores}
    graded = [share for share in needs if isinstance(share, Graded)]
    if graded:  # the median: from the same ballot table as the mean
        shares["graded"] = {share: graded_shares(candidates, model, share.scored.levels, medians, share.scored.rule,
                                                 share.scored.delta, mu=share.scored.mu, kappa=share.scored.kappa,
                                                 power=share.scored.power, votes=share.scored.votes,
                                                 by_order=share.scored in ordered)
                            for share in graded}
    score_pairs = [share for share in needs if isinstance(share, ScoredPairwise)]
    if score_pairs:
        shares["scored_pairwise"] = {
            share: score_comparison_shares(candidates, model, medians, share.scored.levels,
                                           share.scored.rule, share.scored.delta, mu=share.scored.mu,
                                           kappa=share.scored.kappa, power=share.scored.power,
                                           votes=share.scored.votes, by_order=share.scored in ordered,
                                           tie_weight=share.tie_weight)
            for share in score_pairs
        }
    grids = [share for share in needs if isinstance(share, Ballots)]
    if grids:  # the score shares of strategic voters, computed when asked
        shares["ballots"] = {share: ScoreGrid(np.asarray(candidates, dtype=np.float64), model, medians,
                                              share.scored.levels, share.scored.rule, share.scored.delta,
                                              share.scored.mu, share.scored.kappa, share.scored.power,
                                              share.scored.votes)
                             for share in grids}
        shares["cells"] = np.stack(np.meshgrid(np.arange(len(medians)), np.arange(len(medians)), indexing="ij"),
                                   axis=-1)
    return Voters(**shares)


def winners(method: str | Winner, candidates, model: Model, size: int):
    """(coordinates (G,), winner (G, G), margin (G, G)) with G = size + 2 on the grid();
    [i, j] is the point (coordinates[i], coordinates[j]). `method` is a key of MARGINS
    or a built method itself."""
    rule = MARGINS[method] if isinstance(method, str) else method
    coords = grid(size, model.pixels)[0]
    return coords, *rule.evaluate(voters(rule.needs, candidates, model, size))


def _ring(points):
    return np.round(points, DIGITS).ravel().tolist()


def _area(points):
    """Signed (shoelace) area, positive for a counter-clockwise ring."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def contour_regions(coords, winner, margin, points=None):
    """[{"winner": c, "polygons": [[outer, hole, ...], ...]}] per winner, each ring
    flat [x0, y0, x1, y1, ...], from winner and margin on the grid of coords. With
    `points` (G, G, 2) the grid point [i, j] is drawn at points[i, j] instead of
    (coords[i], coords[j]): a grid of quadrilaterals, which must not fold over."""
    x, y = (coords, coords) if points is None else (points[..., 0].T, points[..., 1].T)
    regions = []
    for label in np.unique(winner):
        psi = np.where(winner == label, margin, -margin).astype(np.float64)
        generator = contourpy.contour_generator(x, y, psi.T, fill_type="OuterOffset")
        polygons = []
        for p, o in zip(*generator.filled(0.0, np.inf)):
            rings = [p[o[k]:o[k + 1]] for k in range(len(o) - 1)]
            if abs(_area(rings[0])) > TINY:
                polygons.append([_ring(r) for r in rings if abs(_area(r)) > TINY])
        regions.append({"winner": int(label), "polygons": polygons})
    return regions


def voronoi_regions(candidates):
    """Exact Voronoi regions: straight borders, no grid."""
    return [{"winner": c, "polygons": [[_ring(cell)]]}
            for c, cell in enumerate(voronoi_cells(candidates)) if cell is not None]


def nearest(candidates, points):
    """(winner, margin) of the Voronoi diagram at the `points` (..., 2): the nearest
    candidate, and by how much the square of the distance to the next one is larger. A
    difference of two squared distances is linear in the point, so along a straight grid
    edge the crossing is placed exactly on the bisector."""
    candidates = np.asarray(candidates, dtype=np.float64)
    squared = ((points[..., None, :] - candidates) ** 2).sum(axis=-1)
    closest = np.partition(squared, 1, axis=-1)
    return squared.argmin(axis=-1), closest[..., 1] - closest[..., 0]


def geometric_grid(model: Model, size: int, shift: float = 1.0):
    """(keep (G,), points (K, K, 2)) of the grid(size) with pixels at their geometric
    median: the grid points of the medians marked in `keep` are drawn at `points`. The
    points at and next to a wall can share a median, the outermost one of the model, and
    only one of them is kept: they would be drawn at the same point.

    `shift` below 1 moves each point only that part of the way, on a straight line from
    where the usual diagram draws it (the outermost ones: on the walls) to its geometric
    median. Both ends keep the order of the neighbours along each axis and the
    orientation of every cell, and so does every point in between: the grid does not
    fold over on the way."""
    coords, medians = grid(size, model.pixels)
    keep = np.concatenate([[True], np.diff(medians) > 0])
    points = geometric_medians(model, medians[keep])
    if shift < 1:
        usual = coords[keep]
        usual[[0, -1]] = coords[[0, -1]]  # the outermost medians are drawn up to the walls
        usual = np.stack(np.meshgrid(usual, usual, indexing="ij"), axis=-1)
        points = usual + shift * (points - usual)
    return keep, points


def regions(method: str | Winner, candidates, model: Model | None, size: int,
            pixel_median: PixelMedian = PIXEL_MEDIAN, shift: float = 1.0):
    """Win regions of `method` ("voronoi", a key of MARGINS or a built method); model
    None means every voter at their pixel, which is the Voronoi diagram for every method.
    `pixel_median` "geometric" draws each pixel at the geometric median of its voters
    (geometric.py), which leaves a strip along the walls empty; `shift` below 1 draws it
    only that part of the way there (geometric_grid). It only matters for Beta
    voters: normal ones have both medians at the same point. The Voronoi diagram of a
    model of Beta voters leaves the same strip empty: its borders are the same straight
    lines, drawn only where the model has pixels."""
    moved = pixel_median == "geometric" and model is not None and model.distribution == "beta"
    if method == "voronoi" or model is None:
        if not moved:
            return voronoi_regions(candidates)
        keep, points = geometric_grid(model, size, shift)
        return contour_regions(grid(size, model.pixels)[0][keep], *nearest(candidates, points), points)
    coords, winner, margin = winners(method, candidates, model, size)
    if moved:
        keep, points = geometric_grid(model, size, shift)
        kept = np.ix_(keep, keep)
        return contour_regions(coords[keep], winner[kept], margin[kept], points)
    return contour_regions(coords, winner, margin)


def voronoi_edges(candidates):
    """The exact borders between the Voronoi cells of the candidates within the unit
    square: ([(start, end), ...], [(c, e), ...]), for each pair the part of its bisector
    that is no farther from c and e than from any other candidate."""
    candidates = np.asarray(candidates, dtype=np.float64)
    squares = (candidates ** 2).sum(axis=1)
    segments, pairs = [], []
    for c, e in combinations(range(len(candidates)), 2):
        normal = candidates[e] - candidates[c]
        if not normal.any():
            continue
        middle, along = (candidates[c] + candidates[e]) / 2, np.array([-normal[1], normal[0]])
        # the points middle + t along with n . p <= o: within the square, and for every
        # other candidate f, 2 (f - c) . p <= |f|^2 - |c|^2
        others = [f for f in range(len(candidates)) if f not in (c, e)]
        normals = np.vstack([-np.eye(2), np.eye(2), 2 * (candidates[others] - candidates[c])])
        offsets = np.concatenate([[0, 0, 1, 1], squares[others] - squares[c]])
        slope, room = normals @ along, offsets - normals @ middle
        with np.errstate(divide="ignore", invalid="ignore"):
            bounds = room / slope
        low, high = bounds[slope < 0].max(initial=-np.inf), bounds[slope > 0].min(initial=np.inf)
        if low < high and (room[slope == 0] >= 0).all():
            segments.append((middle + low * along, middle + high * along))
            pairs.append((c, e))
    return segments, pairs


def _inside(ring, points):
    """Whether each of the points (P, 2) is inside the ring (K, 2): a ray to the right
    crosses it an odd number of times."""
    x, y = points[:, :1], points[:, 1:]
    ax, ay = ring[:, 0], ring[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossed = ((ay > y) != (by > y)) & (x < ax + (y - ay) * (bx - ax) / (by - ay))
    return crossed.sum(axis=1) % 2 == 1


def _within(segment, ring):
    """The parts [(start, end), ...] of the segment (start, end) inside the ring (K, 2)."""
    start, end = segment
    step, edges, to = end - start, np.roll(ring, -1, axis=0) - ring, ring - start
    cross = step[0] * edges[:, 1] - step[1] * edges[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):  # start + t step = ring + u edges
        t = (to[:, 0] * edges[:, 1] - to[:, 1] * edges[:, 0]) / cross
        u = (to[:, 0] * step[1] - to[:, 1] * step[0]) / cross
    cuts = np.unique(np.concatenate([[0.0, 1.0], t[(cross != 0) & (u >= 0) & (u < 1) & (t > 0) & (t < 1)]]))
    inside = _inside(ring, start + (cuts[:-1] + cuts[1:])[:, None] / 2 * step)
    return [(start + a * step, start + b * step) for a, b, kept in zip(cuts[:-1], cuts[1:], inside) if kept]


def voronoi_borders(candidates, model: Model | None, size: int, pixel_median: PixelMedian = PIXEL_MEDIAN,
                    shift: float = 1.0):
    """The borders of regions("voronoi", ...) as metrics.borders gives them, but exact:
    voronoi_edges, and with pixels at their geometric median only their parts within the
    moved grid (geometric_grid), where the methods are drawn. The grid edges a border
    crosses are straight, so regions() traces it through the same points, and far slower,
    except within a grid cell of a point where three Voronoi cells meet: there it stops
    short, and leaves the bisector where a grid point's margin is to a third candidate."""
    segments, pairs = voronoi_edges(candidates)
    moved = pixel_median == "geometric" and model is not None and model.distribution == "beta"
    if not moved:
        return segments, pairs
    ring = rim(geometric_grid(model, size, shift)[1])
    parts = [(part, pair) for segment, pair in zip(segments, pairs) for part in _within(segment, ring)]
    return [part for part, _ in parts], [pair for _, pair in parts]


def diagram(method: str | Winner, candidates, model: Model | None, size: int,
            pixel_median: PixelMedian = PIXEL_MEDIAN, shift: float = 1.0):
    """Return the win regions and the winner grid with its represented points."""
    moved = pixel_median == "geometric" and model is not None and model.distribution == "beta"
    coords = (grid(size, model.pixels)[0] if model is not None else
              np.concatenate([[0.0], (np.arange(size) + 0.5) / size, [1.0]]))
    if method == "voronoi" or model is None:
        if not moved:
            points = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1)
            winner, margin = nearest(candidates, points)
            return voronoi_regions(candidates), points, winner, margin
        keep, points = geometric_grid(model, size, shift)
        winner, margin = nearest(candidates, points)
        return contour_regions(coords[keep], winner, margin, points), points, winner, margin
    coords, winner, margin = winners(method, candidates, model, size)
    if moved:
        keep, points = geometric_grid(model, size, shift)
        kept = np.ix_(keep, keep)
        winner, margin = winner[kept], margin[kept]
        return contour_regions(coords[keep], winner, margin, points), points, winner, margin
    points = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1)
    return contour_regions(coords, winner, margin), points, winner, margin
