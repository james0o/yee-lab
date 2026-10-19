"""Agreement scores for a Yee diagram compared with the candidates' Voronoi diagram."""

import numpy as np
from scipy.spatial import QhullError, Voronoi

# The metrics average over the focus region (focus_distance): the points within FOCUS_ROUND
# deviations of the hull of the candidates and of the Voronoi vertices that are within
# FOCUS_REACH deviations of that hull.
FOCUS_ROUND = 0.25
FOCUS_REACH = 2.0
OUTLINE_STEPS = 64  # the corners of the outline around a vertex of the hull


def _area_weights(points):
    x, y = points[:, 0, 0], points[0, :, 1]
    x_weights = np.empty(len(x))
    y_weights = np.empty(len(y))
    x_steps, y_steps = np.diff(x), np.diff(y)
    x_weights[0], x_weights[-1] = x_steps[0] / 2, x_steps[-1] / 2
    y_weights[0], y_weights[-1] = y_steps[0] / 2, y_steps[-1] / 2
    x_weights[1:-1] = (x_steps[:-1] + x_steps[1:]) / 2
    y_weights[1:-1] = (y_steps[:-1] + y_steps[1:]) / 2
    return x_weights[:, None] * y_weights[None, :]


def _losses(candidates, points, assigned, scale):
    """(loss of the assigned candidate, mean loss of all candidates, decided) at each point:
    the loss of a candidate is r / (r + L), r how much farther it is from the point than the
    nearest one, L the median distance between two candidates. decided is False where no
    candidate is assigned (a Condorcet cycle, Undecided): there the loss is undefined."""
    distances = np.linalg.norm(points[..., None, :] - candidates, axis=-1)
    regrets = distances - distances.min(axis=-1, keepdims=True)
    every = regrets / (regrets + scale)
    valid = (assigned >= 0) & (assigned < len(candidates))
    loss = np.full(assigned.shape, np.nan)
    rows, columns = np.nonzero(valid)
    loss[rows, columns] = every[rows, columns, assigned[rows, columns]]
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


def voronoi_agreement(candidates, points, winner, deviation):
    """The disagreement of the winners `winner` at the `points` with the nearest candidate, over
    the whole diagram and over the focus region (focus_distance), `deviation` being the voters'
    mean absolute deviation: {"diagram": scores, "region": scores}, each of the form
    {"error": e, "random": e_random, "relative": sqrt(e / e_random)}, e the mean loss of the
    winners (_losses) and e_random that of a candidate drawn at random at every point, both
    over the decided points only: a cycle (Undecided) is no result, so its points are left out
    of the diagram, as if they were not part of it. The scores are None where nothing is
    decided in the area. relative is 0 where the nearest candidate wins everywhere and 1 for
    no better than chance, the square root stretching the small errors; it depends on the
    candidates less than e does, and e_random > 0 for two candidates or more. All three in
    percent."""
    candidates = np.asarray(candidates, dtype=np.float64)
    pairwise = np.linalg.norm(candidates[:, None] - candidates[None, :], axis=-1)
    pairwise[pairwise == 0] = np.nan
    scale = float(np.nanmedian(pairwise))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    loss, random, decided = _losses(candidates, points, winner, scale)
    area = _area_weights(points)
    focus = focus_distance(points, candidates, deviation) <= FOCUS_ROUND * deviation
    return {name: _scores(loss[decided], random[decided], weights[decided])
            for name, weights in (("diagram", area), ("region", area * focus))}
