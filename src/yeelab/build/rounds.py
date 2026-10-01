"""One round of blocks.Eliminate, compiled: which candidates go at every point.

The totals of a round come from the ballot's tally in numpy (blocks.py); deciding who
goes is a loop over a handful of candidates per point, which numpy can only do as
reductions over a short last axis, several times slower than the tally itself.
"""

import numpy as np
from numba import njit

from yeelab import threads

CHUNK = 4096  # points per task of the compiled loops (threads.py)


@njit(cache=True, nogil=True, error_model="numpy")
def _drop_lowest(totals, alive, gap):
    """drop_lowest for totals, alive (P, C) into gap (P,)."""
    for p in range(totals.shape[0]):
        # the two lowest remaining totals; ties go to the first candidate, like argmin
        lowest, second = -1, -1
        for c in range(totals.shape[1]):
            if not alive[p, c]:
                continue
            if lowest < 0 or totals[p, c] < totals[p, lowest]:
                second, lowest = lowest, c
            elif second < 0 or totals[p, c] < totals[p, second]:
                second = c
        if second < 0:  # one left: decided
            gap[p] = np.inf
            continue
        gap[p] = totals[p, second] - totals[p, lowest]
        alive[p, lowest] = False


@njit(cache=True, nogil=True, error_model="numpy")
def _drop_below_mean(totals, alive, gap):
    """drop_below_mean for totals, alive (P, C) into gap (P,)."""
    n = totals.shape[1]
    for p in range(totals.shape[0]):
        count, added, first, lowest = 0, 0.0, -1, -1
        for c in range(n):
            if alive[p, c]:
                count += 1
                added += totals[p, c]
                if first < 0:
                    first = c
                if lowest < 0 or totals[p, c] < totals[p, lowest]:
                    lowest = c
        if count < 2:  # one left: decided
            gap[p] = np.inf
            continue
        mean = added / count
        closest, stay = np.inf, 0
        for c in range(n):
            if alive[p, c]:
                closest = min(closest, abs(totals[p, c] - mean))
                # the lowest goes too, in case rounding puts the mean just below it
                if totals[p, c] > mean and c != lowest:
                    stay += 1
        gap[p] = closest
        for c in range(n):  # if no one would stay (all totals equal), the first does
            if alive[p, c] and (totals[p, c] <= mean or c == lowest) and (stay > 0 or c != first):
                alive[p, c] = False


def _round(kernel, totals, alive):
    """kernel on the points flattened, in chunks in parallel: updates alive (..., C) in
    place and returns the gap of every point, shape (...)."""
    if not alive.flags.c_contiguous:  # reshape would copy it
        raise ValueError("alive must be C-contiguous: it is updated in place")
    flat = alive.reshape(-1, alive.shape[-1])
    points = np.ascontiguousarray(totals).reshape(flat.shape)
    gap = np.empty(len(flat))
    threads.in_chunks(lambda a, b: kernel(points[a:b], flat[a:b], gap[a:b]), len(flat), CHUNK)
    return gap.reshape(alive.shape[:-1])


def drop_lowest(totals: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Removes the lowest remaining total from alive (..., C) at every point (ties: the
    lowest index); returns the gap between the two lowest, inf where one remains."""
    return _round(_drop_lowest, totals, alive)


def drop_below_mean(totals: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Removes every remaining total at most their mean from alive (..., C) at every
    point; if that is all of them (all totals equal), the lowest index stays. Returns
    the smallest distance of a remaining total from the mean, inf where one remains."""
    return _round(_drop_below_mean, totals, alive)
