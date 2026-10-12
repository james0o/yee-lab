"""Agreement scores for a Yee diagram compared with the candidates' Voronoi diagram."""

import numpy as np
from scipy.spatial import cKDTree

from yeelab.margin.regions import nearest, regions

_INTERIOR_TOLERANCE = 0.02
_EDGE_TOLERANCE = 0.05
_EDGE_DECAY = 0.1


def _convex_hull(points):
    """Return the counter-clockwise convex hull, or None when it has no area."""
    ordered = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(ordered) < 3:
        return None

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1])
    return hull if len(hull) >= 3 else None


def _inside_hull(points, hull):
    flat = points.reshape(-1, 2)
    inside = np.ones(len(flat), dtype=bool)
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end - start
        relative = flat - start
        inside &= edge[0] * relative[:, 1] - edge[1] * relative[:, 0] >= -1e-12
    return inside.reshape(points.shape[:-1])


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


def _assignment_score(candidates, points, assigned, mask, scale, area_weights):
    distances = np.linalg.norm(points[..., None, :] - candidates, axis=-1)
    nearest_distance = np.partition(distances, 1, axis=-1)[..., 0]
    valid = (assigned >= 0) & (assigned < len(candidates))
    chosen_distance = np.full(assigned.shape, np.inf)
    rows, columns = np.nonzero(valid)
    chosen_distance[rows, columns] = distances[rows, columns, assigned[rows, columns]]
    regret = chosen_distance - nearest_distance
    loss = np.ones(assigned.shape)
    np.divide(regret, regret + scale, out=loss, where=valid)
    weight = area_weights / (1 + (nearest_distance / scale) ** 2)
    return float(100 * (1 - np.average(loss[mask], weights=weight[mask])))


def _boundary_points(winner, margin, points):
    boundaries = []
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        change = winner[first] != winner[second]
        if not change.any():
            continue
        first_margin = margin[first][change]
        second_margin = margin[second][change]
        denominator = first_margin + second_margin
        fraction = np.divide(first_margin, denominator, out=np.full_like(first_margin, 0.5),
                             where=np.isfinite(denominator) & (denominator > 0))
        fraction = np.clip(fraction, 0, 1)
        start, end = points[first][change], points[second][change]
        boundaries.append(start + fraction[:, None] * (end - start))
    return np.concatenate(boundaries) if boundaries else np.empty((0, 2))


def _edge_tolerance(points):
    edge_distance = np.minimum.reduce(
        (points[:, 0], 1 - points[:, 0], points[:, 1], 1 - points[:, 1])
    )
    return _INTERIOR_TOLERANCE + (_EDGE_TOLERANCE - _INTERIOR_TOLERANCE) * np.exp(
        -edge_distance / _EDGE_DECAY
    )


def _boundary_score(actual, reference):
    if not len(actual) or not len(reference):
        return 100.0 if not len(actual) and not len(reference) else 0.0
    actual_distance = cKDTree(reference).query(actual)[0]
    reference_distance = cKDTree(actual).query(reference)[0]
    precision = np.mean(actual_distance <= _edge_tolerance(actual))
    recall = np.mean(reference_distance <= _edge_tolerance(reference))
    return float(200 * precision * recall / (precision + recall)) if precision + recall else 0.0


def voronoi_boundary_segments(candidates, model, size, pixel_median, shift):
    """Return subdivided shared Voronoi boundaries and the local acceptance tolerance."""
    cells = regions("voronoi", candidates, model, size, pixel_median, shift)
    edges = {}
    for cell in cells:
        for polygon in cell["polygons"]:
            for ring in polygon:
                points = np.asarray(ring).reshape(-1, 2)
                for start, end in zip(points, np.roll(points, -1, axis=0)):
                    first, second = tuple(start), tuple(end)
                    key = tuple(sorted((first, second)))
                    edge = edges.get(key)
                    if edge:
                        edge["count"] += 1
                    else:
                        edges[key] = {"start": start, "end": end, "count": 1}

    segments = []
    for edge in edges.values():
        if edge["count"] != 2:
            continue
        start, end = edge["start"], edge["end"]
        length = np.linalg.norm(end - start)
        parts = max(1, int(np.ceil(length / 0.005)))
        points = start + np.linspace(0, 1, parts + 1)[:, None] * (end - start)
        tolerances = _edge_tolerance((points[:-1] + points[1:]) / 2)
        segments.extend([
            [*first.tolist(), *last.tolist(), float(tolerance)]
            for first, last, tolerance in zip(points[:-1], points[1:], tolerances)
        ])
    return segments


def voronoi_agreement(candidates, points, winner, margin):
    """Compute whole-square regret, convex-hull regret, and tolerant boundary F1 (0–100)."""
    candidates = np.asarray(candidates, dtype=np.float64)
    pairwise = np.linalg.norm(candidates[:, None] - candidates[None, :], axis=-1)
    pairwise[pairwise == 0] = np.nan
    scale = float(np.nanmedian(pairwise))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    voronoi_winner, voronoi_margin = nearest(candidates, points)
    area_weights = _area_weights(points)
    inside_score = None
    hull = _convex_hull(candidates)
    if hull is not None:
        inside = _inside_hull(points, hull)
        if inside.any():
            inside_score = _assignment_score(candidates, points, winner, inside, scale, area_weights)

    full = np.ones(winner.shape, dtype=bool)
    return {
        "assignment": _assignment_score(candidates, points, winner, full, scale, area_weights),
        "inside_hull": inside_score,
        "boundaries": _boundary_score(
            _boundary_points(winner, margin, points),
            _boundary_points(voronoi_winner, voronoi_margin, points),
        ),
    }
