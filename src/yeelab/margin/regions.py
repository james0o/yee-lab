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
approval and score shares are computed at the grid points themselves
(shares.unapproved_shares, shares.unscored_shares).

Polygons follow GeoJSON: an outer ring counter-clockwise, then its holes clockwise.
A region can have several polygons (FPTP flares at the walls) and holes (an island
of another winner).

A grid point need not be drawn where its median is. With pixels at the geometric median
of their voters (geometric.py) the same winners and margins are traced on the grid moved
by g: the point of the median m is drawn at g(m), and a crossing is placed along the
moved edge as before. The moved grid ends short of the walls, and so do the regions.
"""

import contourpy
import numpy as np

from yeelab.build import FIRST, METHODS, PAIRWISE, PROFILE, Approved, Scored, Share, Voters, Winner
from yeelab.margin.geometric import PIXEL_MEDIAN, PixelMedian, geometric_medians
from yeelab.margin.shares import (
    Model,
    first_choice_shares,
    pairwise_shares,
    ranking_shares,
    unapproved_shares,
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
    computed at the nodes and interpolated (the approval and score shares: at the
    points themselves); [i, j] is the point (coordinates[i], coordinates[j])."""
    medians = grid(size, model.pixels)[1]

    def interpolate(values):
        return interpolate_to(values, model.medians, medians, model.transform)

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
    cuts = [share.cut for share in needs if isinstance(share, Approved)]
    if cuts:
        shares["unapproved"] = {cut: unapproved_shares(candidates, model, cut, medians)
                                for cut in cuts}
    levels = [share.levels for share in needs if isinstance(share, Scored)]
    if levels:
        shares["unscored"] = {n: unscored_shares(candidates, model, n, medians) for n in levels}
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
    """Exact regions of pixels.methods.voronoi: straight borders, no grid."""
    return [{"winner": c, "polygons": [[_ring(cell)]]}
            for c, cell in enumerate(voronoi_cells(candidates)) if cell is not None]


def geometric_grid(model: Model, size: int):
    """(keep (G,), points (K, K, 2)) of the grid(size) with pixels at their geometric
    median: the grid points of the medians marked in `keep` are drawn at `points`. The
    points at and next to a wall can share a median, the outermost one of the model, and
    only one of them is kept: they would be drawn at the same point."""
    medians = grid(size, model.pixels)[1]
    keep = np.concatenate([[True], np.diff(medians) > 0])
    return keep, geometric_medians(model, medians[keep])


def regions(method: str | Winner, candidates, model: Model | None, size: int,
            pixel_median: PixelMedian = PIXEL_MEDIAN):
    """Win regions of `method` ("voronoi", a key of MARGINS or a built method); model
    None means every voter at their pixel, which is the Voronoi diagram for every method.
    `pixel_median` "geometric" draws each pixel at the geometric median of its voters
    (geometric.py), which leaves a strip along the walls empty. It only matters for Beta
    voters: no voters, or normal ones, have both medians at the same point."""
    if method == "voronoi" or model is None:
        return voronoi_regions(candidates)
    coords, winner, margin = winners(method, candidates, model, size)
    if pixel_median == "geometric" and model.distribution == "beta":
        keep, points = geometric_grid(model, size)
        kept = np.ix_(keep, keep)
        return contour_regions(coords[keep], winner[kept], margin[kept], points)
    return contour_regions(coords, winner, margin)
