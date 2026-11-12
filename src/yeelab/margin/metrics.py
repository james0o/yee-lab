"""Agreement scores for a Yee diagram compared with the candidates' Voronoi diagram."""

import numpy as np
from numba import njit
from scipy.spatial import QhullError, Voronoi

from yeelab.voting import CYCLE, drawn

# The metrics average over the focus region (focus_distance): the points within FOCUS_ROUND
# deviations of the hull of the candidates and of the Voronoi vertices that are within
# FOCUS_REACH deviations of that hull.
FOCUS_ROUND = 0.25
FOCUS_REACH = 2.0
OUTLINE_STEPS = 64  # the corners of the outline around a vertex of the hull


def _axis_weights(points):
    """(x weights (X,), y weights (Y,)) of the grid of `points` (X, Y, 2): the area a point
    stands for is the product of its two (the trapezoid rule along each axis)."""
    x, y = points[:, 0, 0], points[0, :, 1]
    x_weights = np.empty(len(x))
    y_weights = np.empty(len(y))
    x_steps, y_steps = np.diff(x), np.diff(y)
    x_weights[0], x_weights[-1] = x_steps[0] / 2, x_steps[-1] / 2
    y_weights[0], y_weights[-1] = y_steps[0] / 2, y_steps[-1] / 2
    x_weights[1:-1] = (x_steps[:-1] + x_steps[1:]) / 2
    y_weights[1:-1] = (y_steps[:-1] + y_steps[1:]) / 2
    return x_weights, y_weights


def _area_weights(points):
    x_weights, y_weights = _axis_weights(points)
    return x_weights[:, None] * y_weights[None, :]


def _losses(candidates, points, assigned, scale):
    """(loss of the assigned candidate, mean loss of all candidates, decided) at each point:
    the loss of a candidate is r / (r + L), r how much farther it is from the point than the
    nearest one, L the median distance between two candidates. A draw (voting.draw) is a
    candidate drawn at random among its own: the mean loss of them. decided is False where
    no candidate is assigned (a Condorcet cycle, Undecided): there the loss is undefined."""
    distances = np.linalg.norm(points[..., None, :] - candidates, axis=-1)
    regrets = distances - distances.min(axis=-1, keepdims=True)
    every = regrets / (regrets + scale)
    assigned = np.where(assigned < len(candidates), assigned, CYCLE)
    chosen = drawn(assigned, len(candidates))
    count = chosen.sum(axis=-1)
    valid = count > 0
    with np.errstate(invalid="ignore"):
        loss = np.where(valid, (every * chosen).sum(axis=-1) / count, np.nan)
    return loss, every.mean(axis=-1), valid


def borders(cells):
    """The borders between the win regions `cells` (regions.regions), without the outer
    edge: ([(start, end), ...], [(winner, winner), ...]), the segments that two regions
    share and the winners on either side of each."""
    edges = {}
    for cell in cells:
        for polygon in cell["polygons"]:
            for ring in polygon:
                points = np.asarray(ring).reshape(-1, 2)
                for start, end in zip(points, np.roll(points, -1, axis=0)):
                    key = tuple(sorted((tuple(start), tuple(end))))
                    edge = edges.setdefault(key, {"start": start, "end": end, "winners": []})
                    edge["winners"].append(cell["winner"])
    shared = [edge for edge in edges.values() if len(edge["winners"]) == 2]
    return ([(edge["start"], edge["end"]) for edge in shared],
            [tuple(edge["winners"]) for edge in shared])


def convex_hull(points):
    """The vertices of the convex hull of `points` (N, 2), counter-clockwise, without repeats:
    one point or two if the hull is a point or a segment."""
    sorted_points = sorted({tuple(point) for point in np.asarray(points, dtype=np.float64)})
    if len(sorted_points) < 3:
        return np.array(sorted_points)

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(values):
        chain = []
        for point in values:
            while len(chain) > 1 and turn(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain[:-1]

    return np.array(half(sorted_points) + half(sorted_points[::-1]))


def hull_distance(points, hull):
    """The distance of the `points` (..., 2) to the convex polygon `hull` (convex_hull),
    0 inside it."""
    starts, ends = hull, np.roll(hull, -1, axis=0)
    edges = ends - starts
    lengths = (edges ** 2).sum(axis=-1)
    relative = points[..., None, :] - starts
    along = np.divide((relative * edges).sum(axis=-1), lengths, out=np.zeros(relative.shape[:-1]),
                      where=lengths > 0)
    nearest = starts + np.clip(along, 0, 1)[..., None] * edges
    distance = np.linalg.norm(points[..., None, :] - nearest, axis=-1).min(axis=-1)
    if len(hull) >= 3:
        side = edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]
        distance = np.where((side >= 0).all(axis=-1), 0.0, distance)
    return distance


def focus_hull(candidates, deviation):
    """The convex hull (convex_hull) of the candidates and of the Voronoi vertices inside
    the unit square that are at most FOCUS_REACH * deviation from the hull of the candidates:
    the points where three cells or more meet are where the electorate is most divided, and a
    vertex of nearly collinear candidates can lie far away."""
    candidates = np.asarray(candidates, dtype=np.float64)
    hull = convex_hull(candidates)
    try:
        vertices = Voronoi(candidates).vertices
    except (QhullError, ValueError):  # fewer than three candidates, or all on one line
        return hull
    vertices = vertices[((vertices >= 0) & (vertices <= 1)).all(axis=1)]
    vertices = vertices[hull_distance(vertices, hull) <= FOCUS_REACH * deviation]
    return convex_hull(np.vstack([candidates, vertices]))


def focus_distance(points, candidates, deviation):
    """How far the `points` (..., 2) are from focus_hull; the focus region is where this is
    at most FOCUS_ROUND * deviation."""
    return hull_distance(np.asarray(points, dtype=np.float64), focus_hull(candidates, deviation))


def focus_outline(candidates, deviation):
    """The border of the focus region as [(x, y), ...], counter-clockwise: the focus hull
    with a disc of radius FOCUS_ROUND * deviation rolled around it, as a polygon of
    OUTLINE_STEPS corners per vertex of the hull (it may leave the unit square)."""
    angles = 2 * np.pi * np.arange(OUTLINE_STEPS) / OUTLINE_STEPS
    circle = FOCUS_ROUND * deviation * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    hull = focus_hull(candidates, deviation)
    return convex_hull((hull[:, None, :] + circle).reshape(-1, 2))


def _scores(loss, random, area):
    """{"error": e, "random": e_random, "relative": sqrt(e / e_random)} in percent: the means of
    `loss` and `random` weighted by `area`, None where the area is 0."""
    if not area.sum() > 0:
        return {"error": None, "random": None, "relative": None}
    error, chance = np.average(loss, weights=area), np.average(random, weights=area)
    return {"error": float(100 * error), "random": float(100 * chance),
            "relative": float(100 * np.sqrt(error / chance)) if chance > 0 else None}


def voronoi_agreement(candidates, points, winner, deviation, cycles="skip"):
    """The disagreement of the winners `winner` at the `points` with the nearest candidate, over
    the whole diagram and over the focus region (focus_distance), `deviation` being the voters'
    mean absolute deviation: {"diagram": scores, "region": scores}, each of the form
    {"error": e, "random": e_random, "relative": sqrt(e / e_random)}, e the mean loss of the
    winners (_losses) and e_random that of a candidate drawn at random at every point, both
    over the decided points only: a cycle (Undecided) is no result, so its points are left out
    of the diagram, as if they were not part of it; a draw is a candidate drawn at random
    among those of the draw (_losses). The scores are None where nothing is
    decided in the area. relative is 0 where the nearest candidate wins everywhere and 1 for
    no better than chance, the square root stretching the small errors; it depends on the
    candidates less than e does, and e_random > 0 for two candidates or more. All three in
    percent.

    With cycles="worst" a cycle is the worst result instead: its points stay in the diagram
    with the loss of the farthest candidate, so that a method cannot leave out the points it
    decides badly (yee-search scores the methods it generates so)."""
    if cycles not in ("skip", "worst"):
        raise ValueError(f'cycles must be "skip" or "worst", got {cycles!r}')
    candidates = np.ascontiguousarray(candidates, dtype=np.float64)
    pairwise = np.linalg.norm(candidates[:, None] - candidates[None, :], axis=-1)
    pairwise[pairwise == 0] = np.nan
    scale = float(np.nanmedian(pairwise))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 3 or min(points.shape[:2]) < 2:  # not a grid: by the arrays
        return _agreement_of_arrays(candidates, points, winner, deviation, scale, cycles)
    sums = _agreement(points, candidates, np.ascontiguousarray(winner, dtype=np.int64), scale,
                      np.ascontiguousarray(focus_hull(candidates, deviation), dtype=np.float64),
                      FOCUS_ROUND * deviation, *_axis_weights(points), cycles == "worst")
    return {name: _scores_of_sums(*sums[k]) for k, name in enumerate(("diagram", "region"))}


def _scores_of_sums(area: float, loss: float, random: float):
    """_scores from the sums over the area of the weights, and of the loss and the loss of a
    random winner times them."""
    if not area > 0:
        return {"error": None, "random": None, "relative": None}
    error, chance = loss / area, random / area
    return {"error": float(100 * error), "random": float(100 * chance),
            "relative": float(100 * np.sqrt(error / chance)) if chance > 0 else None}


def _agreement_of_arrays(candidates, points, winner, deviation, scale, cycles):
    """voronoi_agreement by whole arrays: what _agreement compiles, for points of any shape."""
    loss, random, decided = _losses(candidates, points, winner, scale)
    if cycles == "worst":
        distances = np.linalg.norm(points[..., None, :] - candidates, axis=-1)
        regret = distances.max(axis=-1) - distances.min(axis=-1)
        loss = np.where(decided, loss, regret / (regret + scale))
        decided = np.ones_like(decided)
    area = _area_weights(points)
    focus = focus_distance(points, candidates, deviation) <= FOCUS_ROUND * deviation
    return {name: _scores(loss[decided], random[decided], weights[decided])
            for name, weights in (("diagram", area), ("region", area * focus))}


@njit(cache=True, nogil=True, error_model="numpy")
def _agreement(points, candidates, winner, scale, hull, radius, x_weights, y_weights, worst):
    """The sums voronoi_agreement averages, in one pass over the grid of `points` (X, Y, 2)
    without an array of every candidate at every point: [[area, loss, random] of the whole
    diagram, the same of the focus region], over the decided points, each weighing
    x_weights[i] * y_weights[j]. The loss of a point is that of _losses, with `worst` that
    of the farthest candidate where no one is assigned; the focus region is the points at
    most `radius` from the convex polygon `hull` (hull_distance)."""
    n, corners = candidates.shape[0], hull.shape[0]
    every = np.empty(n)
    sums = np.zeros((2, 3))
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            x, y = points[i, j, 0], points[i, j, 1]
            nearest = np.inf
            for c in range(n):
                every[c] = np.sqrt((x - candidates[c, 0]) ** 2 + (y - candidates[c, 1]) ** 2)
                nearest = min(nearest, every[c])
            random = farthest = 0.0
            for c in range(n):
                regret = every[c] - nearest
                every[c] = regret / (regret + scale)
                random += every[c]
                farthest = max(farthest, every[c])
            random /= n
            assigned = winner[i, j]
            if 0 <= assigned < n:
                loss = every[assigned]
            elif assigned < -1:  # a draw: the mean of its candidates (voting.draw)
                bits, count, loss = -1 - assigned, 0, 0.0
                for c in range(n):
                    if (bits >> c) & 1:
                        loss += every[c]
                        count += 1
                if count == 0:
                    if not worst:
                        continue
                    loss = farthest
                else:
                    loss /= count
            elif worst:
                loss = farthest
            else:
                continue
            weight = x_weights[i] * y_weights[j]
            sums[0, 0] += weight
            sums[0, 1] += weight * loss
            sums[0, 2] += weight * random
            # hull_distance: 0 inside a polygon of three corners or more
            inside, distance = corners >= 3, np.inf
            for k in range(corners):
                sx, sy = hull[k, 0], hull[k, 1]
                ex, ey = hull[(k + 1) % corners, 0] - sx, hull[(k + 1) % corners, 1] - sy
                rx, ry = x - sx, y - sy
                if ex * ry - ey * rx < 0:
                    inside = False
                length = ex * ex + ey * ey
                along = min(max((rx * ex + ry * ey) / length, 0.0), 1.0) if length > 0 else 0.0
                distance = min(distance, np.sqrt((rx - along * ex) ** 2 + (ry - along * ey) ** 2))
            if inside or distance <= radius:
                sums[1, 0] += weight
                sums[1, 1] += weight * loss
                sums[1, 2] += weight * random
    return sums
