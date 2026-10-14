"""Agreement scores for a Yee diagram compared with the candidates' Voronoi diagram."""

import math

import numpy as np
from numba import njit


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


def _sample_segments(segments, spacing=0.005, with_indices=False):
    """Sample a collection uniformly by total arc length; with_indices also returns the
    segment of every sample."""
    empty = (np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int))
    if not segments:
        return empty if with_indices else empty[:2]
    segment_array = np.asarray(segments, dtype=np.float64)
    lengths = np.linalg.norm(segment_array[:, 1] - segment_array[:, 0], axis=1)
    kept = np.flatnonzero(lengths > 1e-12)
    segment_array, lengths = segment_array[kept], lengths[kept]
    total_length = lengths.sum()
    if not len(lengths) or total_length <= 0:
        return empty if with_indices else empty[:2]
    count = max(1, int(np.ceil(total_length / spacing)))
    distances = (np.arange(count) + 0.5) * total_length / count
    cumulative = np.cumsum(lengths)
    indices = np.searchsorted(cumulative, distances, side="right")
    starts = np.concatenate([[0.0], cumulative[:-1]])[indices]
    fractions = (distances - starts) / lengths[indices]
    points = segment_array[indices, 0] + fractions[:, None] * (
        segment_array[indices, 1] - segment_array[indices, 0]
    )
    weights = np.full(count, total_length / count)
    return (points, weights, kept[indices]) if with_indices else (points, weights)


def _half_gaps(pairs, candidates, scale):
    """Half the distance between the two winners of each boundary segment; half the median
    candidate distance `scale` where one side is a cycle or both sides have the same winner."""
    gaps = np.full(len(pairs), scale / 2)
    for index, (first, second) in enumerate(pairs):
        if first != second and 0 <= first < len(candidates) and 0 <= second < len(candidates):
            gaps[index] = np.linalg.norm(candidates[first] - candidates[second]) / 2
    return gaps


def _nearness(points, candidates, scale):
    """1 / (1 + (d_min / scale)^2), the weight of the assignment regret."""
    nearest = np.linalg.norm(points[:, None, :] - candidates, axis=-1).min(axis=-1)
    return 1 / (1 + (nearest / scale) ** 2)


@njit(cache=True, nogil=True, error_model="numpy")
def _nearest_on_segments(points, starts, directions, squared_lengths):
    """The distance of each point to the nearest of the segments starts + t directions,
    0 <= t <= 1, with squared_lengths |directions|^2 > 0: one pass over the segments per
    point, about 15 times faster than numpy's passes over temporary arrays."""
    out = np.empty(points.shape[0])
    for i in range(points.shape[0]):
        x, y = points[i, 0], points[i, 1]
        nearest = np.inf
        for k in range(starts.shape[0]):
            ax, ay = starts[k, 0], starts[k, 1]
            dx, dy = directions[k, 0], directions[k, 1]
            t = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / squared_lengths[k]))
            px, py = x - (ax + t * dx), y - (ay + t * dy)
            nearest = min(nearest, px * px + py * py)
        out[i] = math.sqrt(nearest)
    return out


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
    return _nearest_on_segments(np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(starts),
                                np.ascontiguousarray(directions), squared_lengths)


def boundary_distance(actual, reference, candidates, scale):
    """Symmetric mean distance between the borders `actual` and `reference` (borders()),
    each sample's distance divided by half the distance between the two candidates its
    border separates and capped at 1 (the border has moved all the way to a candidate).
    Samples are weighted by length and by nearness to a candidate like the assignment
    regret. None without any border."""
    candidates = np.asarray(candidates, dtype=np.float64)
    sides = []
    for segments, pairs in (actual, reference):
        points, lengths, indices = _sample_segments(segments, with_indices=True)
        if not len(points):
            return None
        sides.append((segments, points, lengths, _half_gaps(pairs, candidates, scale)[indices]))

    means = []
    for (_, points, lengths, gaps), (other, *_) in zip(sides, reversed(sides)):
        distances = _point_to_segments(points, other)
        if not np.isfinite(distances).all():
            return None
        relative = np.minimum(1, distances / np.maximum(gaps, 1e-12))
        means.append(np.average(relative, weights=lengths * _nearness(points, candidates, scale)))
    return float(np.mean(means))


def voronoi_agreement(candidates, points, winner, actual, reference):
    """Compute whole-square regret, convex-hull regret, and the weighted boundary distance
    between the borders `actual` of the diagram and `reference` of the Voronoi diagram
    (borders(), regions.voronoi_borders)."""
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
        "boundary_mean": boundary_distance(actual, reference, candidates, scale),
    }
