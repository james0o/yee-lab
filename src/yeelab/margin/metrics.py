"""Agreement scores for a Yee diagram compared with the candidates' Voronoi diagram."""

import numpy as np


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


def voronoi_agreement(candidates, points, winner):
    """The disagreement of the winners `winner` at the `points` with the nearest candidate:
    {"error": e, "random": e_random, "relative": sqrt(e / e_random)}, e the mean loss of the winners
    over the area (_losses) and e_random that of a candidate drawn at random at every point,
    both over the decided points only: a cycle (Undecided) is no result, so its points are
    left out of the diagram, as if they were not part of it. None where nothing is decided.
    relative is 0 where the nearest candidate wins everywhere and 1 for no better than chance,
    the square root stretching the small errors; it depends on the candidates less than e does, and e_random > 0 for two candidates or more.
    All three in percent."""
    candidates = np.asarray(candidates, dtype=np.float64)
    pairwise = np.linalg.norm(candidates[:, None] - candidates[None, :], axis=-1)
    pairwise[pairwise == 0] = np.nan
    scale = float(np.nanmedian(pairwise))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    loss, random, decided = _losses(candidates, points, winner, scale)
    area = _area_weights(points)[decided]
    if not area.sum() > 0:
        return {"error": None, "random": None, "relative": None}
    error, chance = np.average(loss[decided], weights=area), np.average(random[decided], weights=area)
    return {"error": float(100 * error), "random": float(100 * chance),
            "relative": float(100 * np.sqrt(error / chance)) if chance > 0 else None}
