"""What pixels/methods.py and margin/methods.py share: the winner code of a Condorcet
cycle, and the rounds of instant runoff, compiled.

The rounds give the IRV winner and its margin together; the margin costs nothing extra,
margin.methods.irv_margin returns both and pixels.methods.irv only the winner.
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
