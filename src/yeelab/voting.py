"""Shared compiled logic for Condorcet cycles and instant-runoff rounds.

The rounds give the IRV winner and its margin together; the margin costs nothing extra,
and IRV methods built from yeelab.build return both. irv_schedules does the same for
every way to split the eliminations into rounds (blocks.Schedules), and
descending_coalitions for descending acquiescing coalitions (blocks.DescendingCoalitions).
"""

import numpy as np
from numba import njit

from yeelab import threads

CYCLE = -1  # winner where there is no Condorcet winner
CHUNK = 4096  # points per task of the compiled loop (threads.py)


@njit(cache=True, nogil=True, error_model="numpy")
def _irv_points(rankings, probs, bullet, winner, margin):
    """irv_rounds for probs (P, R) and bullet (P, C), or (P, 0) for none, into winner
    (P,) and margin (P,).

    Every ballot type points to its highest ranked remaining candidate, so an
    elimination only moves the ballots that pointed to the eliminated candidate:
    at most R C steps per point in all rounds together. A bullet ballot counts for its
    first choice; once that is out, it counts equally for every remaining candidate,
    all ranked equal on it. That adds the same to every remaining tally, which changes
    no elimination and no gap, so it is left out.
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
        if bullet.shape[1] > 0:  # as a whole: a loop over c here slows the rounds below by a third
            tally += bullet[p]
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
def _schedule_points(rankings, probs, bullet, chosen, margin):
    """irv_schedules for probs (P, R) and bullet (P, C), or (P, 0) for none, into chosen
    (P, C), all False, and margin (P,). A bullet ballot counts for its first choice
    while it remains, and then the same for every remaining candidate, which is left
    out as in _irv_points.

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
            if bullet.shape[1] > 0:  # the others' tallies are not read
                tally += bullet[p]
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


@njit(cache=True, nogil=True, error_model="numpy")
def _coalition_rounds(score, full):
    """The winner and margin of descending_coalitions at one point from the score of
    every set but the full one, (full,).

    A set (a bit mask) cuts the eligible candidates E if it holds some of them but not
    all; only such a set changes E. Every set with a higher score than one that cuts E
    holds all of E or none of it by then, so each round takes the highest score of a
    set that cuts E (ties: the lowest mask) and keeps its part of E. A rival is a set
    that cuts E and holds none of that part: had it come first, the winner would be one
    of the candidates the round drops. The round's gap is the top score minus the
    highest score of a rival; other sets that cut E can come first and leave the winner.
    """
    eligible = full
    gap = np.inf
    while eligible & (eligible - 1):  # two or more are eligible
        top, best = 0, -np.inf
        for s in range(1, full):
            cut = s & eligible
            if cut != 0 and cut != eligible and score[s] > best:
                top, best = s, score[s]
        kept = top & eligible
        rival = -np.inf
        for s in range(1, full):
            cut = s & eligible
            if cut != 0 and cut != eligible and (cut & kept) == 0:
                rival = max(rival, score[s])
        gap = min(gap, best - rival)
        eligible = kept
    c = 0
    while not (eligible >> c) & 1:
        c += 1
    return c, gap


@njit(cache=True, nogil=True, error_model="numpy")
def _coalition_points(prefixes, probs, bullet, winner, margin):
    """descending_coalitions for probs (P, R) and bullet (P, C), or (P, 0) for none, into
    winner (P,) and margin (P,), with prefixes (R, C - 1) the bit masks of the first 1,
    ..., C - 1 candidates of every ranking. A bullet ballot for c, which ranks all
    others equal below c, acquiesces to every set that holds c. The rounds are a
    function of their own (_coalition_rounds): inline, the bullet ballots slowed them
    by a tenth even where there are none."""
    n_ballots, steps = prefixes.shape
    full = (1 << (steps + 1)) - 1
    score = np.empty(full)  # of every set but the full one, which never cuts E
    held = np.zeros(full)  # the bullet ballots of the candidates in every set
    lowest = np.zeros(full, dtype=np.int64)  # the lowest candidate of every set
    for s in range(1, full):
        while not (s >> lowest[s]) & 1:
            lowest[s] += 1
    for p in range(probs.shape[0]):
        score[:] = 0.0
        for r in range(n_ballots):
            for k in range(steps):
                score[prefixes[r, k]] += probs[p, r]
        if bullet.shape[1] > 0:
            for s in range(1, full):  # s without its lowest candidate, and that one
                held[s] = held[s & (s - 1)] + bullet[p, lowest[s]]
                score[s] += held[s]
        winner[p], margin[p] = _coalition_rounds(score, full)


def _bullet(bullet: np.ndarray | None, points: int) -> np.ndarray:
    """The bullet ballots (..., C) of the kernels, flat (P, C), or (P, 0) for none."""
    if bullet is None:
        return np.empty((points, 0))
    return np.ascontiguousarray(bullet, dtype=np.float64).reshape(points, -1)


def descending_coalitions(rankings: np.ndarray, probs: np.ndarray,
                          bullet: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Descending acquiescing coalitions (blocks.DescendingCoalitions): every set of
    candidates scores the share of the voters who rank none of the others above any of
    it, and from the highest score down every set makes the candidates outside it
    ineligible, unless that leaves no one eligible. Returns (winner, margin) for
    rankings (R, C), their shares probs (..., R) and the shares of the bullet ballots
    (..., C) of each first choice, None for none (Voters.bullet).

    A complete ranking acquiesces to exactly the sets of its first k candidates, so a
    set's score is the sum of the shares of the rankings it begins, and a bullet
    ballot to every set that holds its first choice. The margin is the
    smallest gap of any round: the top score of a set that cuts the eligible candidates
    minus the highest score of a rival, a set that would leave the winner among the
    candidates the round drops. Unlike that of IRV it vanishes only where the winner
    changes.
    """
    shape = probs.shape[:-1]
    flat = np.ascontiguousarray(probs.reshape(-1, probs.shape[-1]))
    rankings = np.asarray(rankings, dtype=np.int64)
    prefixes = np.ascontiguousarray(np.bitwise_or.accumulate(1 << rankings, axis=1)[:, :-1])
    ballots = _bullet(bullet, len(flat))
    winner = np.empty(len(flat), dtype=np.int64)
    margin = np.empty(len(flat))
    threads.in_chunks(lambda a, b: _coalition_points(prefixes, flat[a:b], ballots[a:b], winner[a:b],
                                                     margin[a:b]), len(flat), CHUNK)
    return winner.reshape(shape), margin.reshape(shape)


def irv_schedules(rankings: np.ndarray, probs: np.ndarray,
                  bullet: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The winners of every elimination schedule (blocks.Schedules on first choices):
    each round drops some number of the fewest first choices among the remaining
    candidates, in every way to split the C - 1 eliminations into rounds. Returns
    (chosen (..., C) bool, margin (...)) for rankings (R, C), their shares probs
    (..., R) and the bullet ballots (..., C), None for none (as in irv_rounds).

    The margin is the smallest gap of any round of any schedule, the k-th highest tally
    minus the next where the round keeps k: a winner changes only where some round's
    decision flips, and there its gap is 0.
    """
    shape = probs.shape[:-1]
    flat = np.ascontiguousarray(probs.reshape(-1, probs.shape[-1]))
    rankings = np.ascontiguousarray(rankings, dtype=np.int64)
    n = rankings.shape[1]
    ballots = _bullet(bullet, len(flat))
    chosen = np.zeros((len(flat), n), dtype=np.bool_)
    margin = np.empty(len(flat))
    threads.in_chunks(lambda a, b: _schedule_points(rankings, flat[a:b], ballots[a:b], chosen[a:b],
                                                    margin[a:b]), len(flat), CHUNK)
    return chosen.reshape(*shape, n), margin.reshape(shape)


def irv_rounds(rankings: np.ndarray, probs: np.ndarray,
               bullet: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Instant runoff: repeatedly eliminate the candidate with the fewest votes.
    Returns (winner, margin) for rankings (R, C), their shares probs (..., R) and the
    shares of the bullet ballots (..., C) of each first choice, None for none
    (Voters.bullet): once its first choice is out, a bullet ballot counts equally for
    all remaining candidates.

    Ballots are complete, so a candidate with a majority is never
    eliminated; eliminating until one candidate remains gives the IRV winner.
    The margin is the smallest gap between the two lowest tallies of any round:
    the winner can change only where some elimination flips, and the gap is 0 on
    both sides of such a flip.
    """
    shape = probs.shape[:-1]
    flat = np.ascontiguousarray(probs.reshape(-1, probs.shape[-1]))
    rankings = np.ascontiguousarray(rankings, dtype=np.int64)
    ballots = _bullet(bullet, len(flat))
    winner = np.empty(len(flat), dtype=np.int64)
    margin = np.empty(len(flat))
    threads.in_chunks(lambda a, b: _irv_points(rankings, flat[a:b], ballots[a:b], winner[a:b], margin[a:b]),
                      len(flat), CHUNK)
    return winner.reshape(shape), margin.reshape(shape)
