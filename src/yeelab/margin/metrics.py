"""Agreement scores for a Yee diagram compared with the candidates' Voronoi diagram."""

import numpy as np


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


def _shared_boundary_segments(cells):
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
    return [(edge["start"], edge["end"]) for edge in edges.values() if edge["count"] == 2]


def _sample_segments(segments, spacing=0.005):
    """Sample a collection uniformly by total arc length."""
    if not segments:
        return np.empty((0, 2)), np.empty(0)
    segment_array = np.asarray(segments, dtype=np.float64)
    lengths = np.linalg.norm(segment_array[:, 1] - segment_array[:, 0], axis=1)
    keep = lengths > 1e-12
    segment_array, lengths = segment_array[keep], lengths[keep]
    if not len(lengths):
        return np.empty((0, 2)), np.empty(0)
    total_length = lengths.sum()
    if total_length <= 0:
        return np.empty((0, 2)), np.empty(0)
    count = max(1, int(np.ceil(total_length / spacing)))
    distances = (np.arange(count) + 0.5) * total_length / count
    cumulative = np.cumsum(lengths)
    indices = np.searchsorted(cumulative, distances, side="right")
    starts = np.concatenate([[0.0], cumulative[:-1]])[indices]
    fractions = (distances - starts) / lengths[indices]
    points = segment_array[indices, 0] + fractions[:, None] * (
        segment_array[indices, 1] - segment_array[indices, 0]
    )
    return points, np.full(count, total_length / count)


def _sample_boundaries(cells):
    return _sample_segments(_shared_boundary_segments(cells))


def _centrality(points):
    radius_squared = ((points - 0.5) ** 2).sum(axis=-1)
    return np.exp(-4 * radius_squared)


def _point_to_segments(points, segments):
    """Exact nearest Euclidean distance to continuous line segments (clamped projection)."""
    segments = np.asarray(segments, dtype=np.float64)
    if not len(points) or not len(segments):
        return np.full(len(points), np.inf)
    starts, directions = segments[:, 0], segments[:, 1] - segments[:, 0]
    squared_lengths = np.sum(directions * directions, axis=1)
    keep = squared_lengths > 1e-24
    starts, directions, squared_lengths = starts[keep], directions[keep], squared_lengths[keep]
    if not len(squared_lengths):
        return np.full(len(points), np.inf)
    result = np.empty(len(points))
    for index, point in enumerate(points):
        fractions = np.clip(
            np.sum((point - starts) * directions, axis=1) / squared_lengths, 0, 1
        )
        projections = starts + fractions[:, None] * directions
        result[index] = np.sqrt(np.min(np.sum((point - projections) ** 2, axis=1)))
    return result


def _symmetric_boundary_distances(actual_segments, reference_segments, actual, actual_lengths,
                                  reference, reference_lengths):
    actual_to_reference = _point_to_segments(actual, reference_segments)
    reference_to_actual = _point_to_segments(reference, actual_segments)
    if not np.isfinite(actual_to_reference).all() or not np.isfinite(reference_to_actual).all():
        return None

    actual_weights = actual_lengths * _centrality(actual)
    reference_weights = reference_lengths * _centrality(reference)
    actual_weights /= actual_weights.sum()
    reference_weights /= reference_weights.sum()
    distances = np.concatenate([actual_to_reference, reference_to_actual])
    weights = np.concatenate([actual_weights / 2, reference_weights / 2])
    return float(np.average(distances, weights=weights))


def boundary_distance(actual_cells, reference_cells):
    """Centre-weighted symmetric mean distance between the boundaries, or None without any."""
    actual, actual_lengths = _sample_boundaries(actual_cells)
    reference, reference_lengths = _sample_boundaries(reference_cells)
    if not len(actual) or not len(reference):
        return None
    return _symmetric_boundary_distances(
        _shared_boundary_segments(actual_cells), _shared_boundary_segments(reference_cells),
        actual, actual_lengths, reference, reference_lengths,
    )


def voronoi_boundary_segments(cells):
    """Return shared Voronoi boundary segments, without the outer square perimeter."""
    segments = []
    for start, end in _shared_boundary_segments(cells):
        length = np.linalg.norm(end - start)
        count = max(1, int(np.ceil(length / 0.005)))
        points = start + np.linspace(0, 1, count + 1)[:, None] * (end - start)
        segments.extend([
            [*first.tolist(), *last.tolist()]
            for first, last in zip(points[:-1], points[1:])
        ])
    return segments


def voronoi_agreement(candidates, points, winner, actual_cells, reference_cells):
    """Compute whole-square regret, convex-hull regret, and the weighted boundary distance."""
    candidates = np.asarray(candidates, dtype=np.float64)
    pairwise = np.linalg.norm(candidates[:, None] - candidates[None, :], axis=-1)
    pairwise[pairwise == 0] = np.nan
    scale = float(np.nanmedian(pairwise))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

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
        "boundary_mean": boundary_distance(actual_cells, reference_cells),
    }
