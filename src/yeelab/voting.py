"""Shared compiled logic for Condorcet cycles and instant-runoff rounds.

The rounds give the IRV winner and its margin together; the margin costs nothing extra,
and IRV methods built from yeelab.build return both. irv_schedules does the same for
every way to split the eliminations into rounds (blocks.Schedules).
"""

import numpy as np
from numba import njit

from yeelab import threads

CYCLE = -1  # winner where there is no Condorcet winner
CHUNK = 4096  # points per task of the compiled loop (threads.py)


@njit(cache=True, nogil=True, error_model="numpy")
def _irv_points(rankings, probs, winner, margin):
    """irv_rounds for probs (P, R) into winner (P,) and margin (P,).

    Every ballot type points to its highest ranked remaining candidate, so an
    elimination only moves the ballots that pointed to the eliminated candidate:
    at most R C steps per point in all rounds together.
    """
    n_ballots, n_candidates = rankings.shape
    tally = np.empty(n_candidates)
    alive = np.empty(n_candidates, dtype=np.bool_)
    top = np.empty(n_ballots, dtype=np.int64)
    for p in range(probs.shape[0]):
        tally[:] = 0.0
        alive[:] = True
        for r in range(n_ballots):
            top[r] = 0
            tally[rankings[r, 0]] += probs[p, r]
        gap = np.inf
        for _ in range(n_candidates - 1):
            # the two lowest tallies; ties go to the first candidate, like argmin
            lowest, second = -1, -1
            for c in range(n_candidates):
                if not alive[c]:
                    continue
                if lowest < 0 or tally[c] < tally[lowest]:
                    second, lowest = lowest, c
                elif second < 0 or tally[c] < tally[second]:
                    second = c
            gap = min(gap, tally[second] - tally[lowest])
            alive[lowest] = False
            for r in range(n_ballots):
                if rankings[r, top[r]] == lowest:
                    k = top[r] + 1
                    while not alive[rankings[r, k]]:
                        k += 1
                    top[r] = k
                    tally[rankings[r, k]] += probs[p, r]
        for c in range(n_candidates):
            if alive[c]:
                winner[p] = c
        margin[p] = gap


@njit(cache=True, nogil=True, error_model="numpy")
def _schedule_points(rankings, probs, chosen, margin):
    """irv_schedules for probs (P, R) into chosen (P, C), all False, and margin (P,).

    A round of a schedule keeps the k highest tallies of the candidates it starts with, so
    what follows depends only on those: every set of remaining candidates (a bit mask)
    that some schedule reaches is tallied once. Its highest tally wins the schedule that
    keeps one there, and the k-th highest minus the next is the gap of keeping k.
    """
    n_ballots, n_candidates = rankings.shape
    full = (1 << n_candidates) - 1
    seen = np.zeros(1 << n_candidates, dtype=np.int64)  # the last point (+ 1) that reached a set
    stack = np.empty(1 << n_candidates, dtype=np.int64)
    tally = np.empty(n_candidates)
    order = np.empty(n_candidates, dtype=np.int64)
    for p in range(probs.shape[0]):
        gap = np.inf
        seen[full] = p + 1
        stack[0] = full
        size = 1
        while size > 0:
            size -= 1
            alive = stack[size]
            tally[:] = 0.0
            for r in range(n_ballots):
                k = 0
                while not (alive >> rankings[r, k]) & 1:
                    k += 1
                tally[rankings[r, k]] += probs[p, r]
            # the remaining candidates, highest tally first (ties: the lowest index)
            left = 0
            for c in range(n_candidates):
                if (alive >> c) & 1:
                    i = left
                    while i > 0 and tally[order[i - 1]] < tally[c]:
                        order[i] = order[i - 1]
                        i -= 1
                    order[i] = c
                    left += 1
            chosen[p, order[0]] = True
            kept = 1 << order[0]  # the k highest
            for k in range(1, left):
                gap = min(gap, tally[order[k - 1]] - tally[order[k]])
                if k > 1 and seen[kept] != p + 1:
                    seen[kept] = p + 1
                    stack[size] = kept
                    size += 1
                kept |= 1 << order[k]
        margin[p] = gap


def irv_schedules(rankings: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The winners of every elimination schedule (blocks.Schedules on first choices):
    each round drops some number of the fewest first choices among the remaining
    candidates, in every way to split the C - 1 eliminations into rounds. Returns
    (chosen (..., C) bool, margin (...)) for rankings (R, C) and their shares probs
    (..., R).

    The margin is the smallest gap of any round of any schedule, the k-th highest tally
    minus the next where the round keeps k: a winner changes only where some round's
    decision flips, and there its gap is 0.
    """
    shape = probs.shape[:-1]
    flat = np.ascontiguousarray(probs.reshape(-1, probs.shape[-1]))
    rankings = np.ascontiguousarray(rankings, dtype=np.int64)
    n = rankings.shape[1]
    chosen = np.zeros((len(flat), n), dtype=np.bool_)
    margin = np.empty(len(flat))
    threads.in_chunks(lambda a, b: _schedule_points(rankings, flat[a:b], chosen[a:b], margin[a:b]),
                      len(flat), CHUNK)
    return chosen.reshape(*shape, n), margin.reshape(shape)


def irv_rounds(rankings: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instant runoff: repeatedly eliminate the candidate with the fewest votes.
    Returns (winner, margin) for rankings (R, C) and their shares probs (..., R).

    Ballots are complete rankings, so a candidate with a majority is never
    eliminated; eliminating until one candidate remains gives the IRV winner.
    The margin is the smallest gap between the two lowest tallies of any round:
    the winner can change only where some elimination flips, and the gap is 0 on
    both sides of such a flip.
    """
    shape = probs.shape[:-1]
    flat = np.ascontiguousarray(probs.reshape(-1, probs.shape[-1]))
    rankings = np.ascontiguousarray(rankings, dtype=np.int64)
    winner = np.empty(len(flat), dtype=np.int64)
    margin = np.empty(len(flat))
    threads.in_chunks(lambda a, b: _irv_points(rankings, flat[a:b], winner[a:b], margin[a:b]),
                      len(flat), CHUNK)
    return winner.reshape(shape), margin.reshape(shape)
