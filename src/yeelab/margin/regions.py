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
approval shares are computed at the grid points themselves (shares.unapproved_shares).

Polygons follow GeoJSON: an outer ring counter-clockwise, then its holes clockwise.
A region can have several polygons (FPTP flares at the walls) and holes (an island
of another winner).
"""

import contourpy
import numpy as np

from yeelab.build import FIRST, METHODS, PAIRWISE, PROFILE, Approved, Share, Voters
from yeelab.margin.shares import (
    Model,
    first_choice_shares,
    pairwise_shares,
    ranking_shares,
    unapproved_shares,
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
    computed at the nodes and interpolated (the approval shares: at the points
    themselves); [i, j] is the point (coordinates[i], coordinates[j])."""
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
    return Voters(**shares)


def winners(method: str, candidates, model: Model, size: int):
    """(coordinates (G,), winner (G, G), margin (G, G)) with G = size + 2 on the grid();
    [i, j] is the point (coordinates[i], coordinates[j])."""
    rule = MARGINS[method]
    coords = grid(size, model.pixels)[0]
    return coords, *rule.evaluate(voters(rule.needs, candidates, model, size))


def _ring(points):
    return np.round(points, DIGITS).ravel().tolist()


def _area(points):
    """Signed (shoelace) area, positive for a counter-clockwise ring."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def contour_regions(coords, winner, margin):
    """[{"winner": c, "polygons": [[outer, hole, ...], ...]}] per winner, each ring
    flat [x0, y0, x1, y1, ...], from winner and margin on the grid of coords."""
    regions = []
    for label in np.unique(winner):
        psi = np.where(winner == label, margin, -margin).astype(np.float64)
        generator = contourpy.contour_generator(coords, coords, psi.T, fill_type="OuterOffset")
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


def regions(method: str, candidates, model: Model | None, size: int):
    """Win regions of `method` ("voronoi" or a key of MARGINS); model None means
    every voter at their pixel, which is the Voronoi diagram for every method."""
    if method == "voronoi" or model is None:
        return voronoi_regions(candidates)
    return contour_regions(*winners(method, candidates, model, size))
