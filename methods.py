import numpy as np
from scipy.spatial.distance import cdist
from const import PIXELS, N_PART, CANDIDATES, N_CANDIDATES
from typing import Callable
import beta_weights
import time, argparse, os
from beta_weights import load_weights

args = argparse.ArgumentParser()
args.add_argument("--n_part", type=int, default=N_PART)
args.add_argument("--pixels", type=int, default=PIXELS)
args = args.parse_args()

weights = load_weights(args.n_part)
n_part = args.n_part

def create_voters2d(voters) -> np.ndarray:
    return np.column_stack((np.repeat(voters, len(voters)), np.tile(voters, len(voters))))

def calculate_distance(voters2d, candidates) -> np.ndarray:
    return cdist(voters2d, candidates, metric='euclidean')

voters = np.concatenate(([0.0], beta_weights.create_voters(n_part), [1.0]))
grid_size = len(voters)
voters2d = create_voters2d(voters)
distance = calculate_distance(voters2d, CANDIDATES)
cardinal_dist = np.argsort(distance, axis=1).argsort().astype(np.uint8)
_rank_prefs = np.argsort(cardinal_dist, axis=1).reshape(
    grid_size, grid_size, N_CANDIDATES).astype(np.uint8)
n_voters = grid_size * grid_size
prefs_flat = _rank_prefs.reshape(n_voters, N_CANDIDATES)
prefs_indicator = np.eye(N_CANDIDATES, dtype=weights.dtype)[prefs_flat]
_i_idx = np.repeat(np.arange(grid_size), grid_size)
_j_idx = np.tile(np.arange(grid_size), grid_size)
_voter_weights = (
    weights[:, _i_idx].T[:, :, None] * weights[:, _j_idx].T[:, None, :]
)

def aggregate(
    voter_scores: np.ndarray,
    x_slice: slice | None = None,
    y_slice: slice | None = None,
) -> np.ndarray:
    """Sum weighted scores over selected pixel blocks.
    voter_scores shape (n_part, n_part, N_CANDIDATES).
    x_slice and y_slice select rows/columns of the pixel grid.
    """
    x_weights = weights[x_slice] if x_slice is not None else weights
    y_weights = weights[y_slice] if y_slice is not None else weights
    inner = np.einsum('ijc,yj->ciy', voter_scores, y_weights, optimize=True)
    return np.einsum('xi,ciy->xyc', x_weights, inner, optimize=True)

def aggregate_pairwise(voter_pref: np.ndarray) -> np.ndarray:
    """Weighted sums of pairwise preferences. voter_pref shape (n_part, n_part, C, C)."""
    inner = np.einsum('ijab,yj->abyi', voter_pref.astype(weights.dtype), weights, optimize=True)
    return np.einsum('xi,abyi->xyab', weights, inner, optimize=True)

def _round_votes(
    prefs: np.ndarray,
    still_standing: np.ndarray,
    voter_weights: np.ndarray,
    x_slice: slice | None = None,
    y_slice: slice | None = None,
) -> np.ndarray:
    """Weighted first-preference counts among still-standing candidates, per pixel."""
    if np.all(still_standing):
        first = prefs[..., 0]
        return aggregate(
            (first[..., np.newaxis] == np.arange(N_CANDIDATES)).astype(weights.dtype),
            x_slice=x_slice,
            y_slice=y_slice,
        )

    counts = np.zeros((*still_standing.shape[:2], N_CANDIDATES), dtype=weights.dtype)
    remaining = np.ones((n_voters, *still_standing.shape[:2]), dtype=bool)

    for r in range(N_CANDIDATES):
        candidates = prefs_flat[:, r]
        if not remaining.any():
            break

        still_this_rank = still_standing[..., candidates]
        still_this_rank = np.transpose(still_this_rank, (2, 0, 1))
        mask = remaining & still_this_rank
        if not mask.any():
            continue

        weighted_mask = voter_weights * mask
        counts += np.einsum('vc,vxy->xyc', prefs_indicator[:, r, :], weighted_mask, optimize=True)
        remaining &= ~mask

    return counts

def first_past_the_post() -> np.ndarray:
    votes = np.argmin(cardinal_dist, axis=1).reshape(grid_size, grid_size)
    voter_scores = votes[..., np.newaxis] == np.arange(N_CANDIDATES)
    return aggregate(voter_scores.astype(weights.dtype)).argmax(axis=2)

def borda_count() -> np.ndarray:
    voter_scores = (N_CANDIDATES - 1 - cardinal_dist).reshape(grid_size, grid_size, N_CANDIDATES)
    return aggregate(voter_scores.astype(weights.dtype)).argmax(axis=2)

def black() -> np.ndarray:
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pref = ranks[..., :, None] < ranks[..., None, :]
    pairwise = aggregate_pairwise(pref)
    beats = pairwise > np.transpose(pairwise, (0, 1, 3, 2))

    is_condorcet = np.zeros((PIXELS, PIXELS, N_CANDIDATES), dtype=bool)
    for c in range(N_CANDIDATES):
        others = [o for o in range(N_CANDIDATES) if o != c]
        is_condorcet[:, :, c] = np.all(beats[:, :, c, others], axis=-1)

    has_condorcet = is_condorcet.any(axis=-1)
    return np.where(has_condorcet, is_condorcet.argmax(axis=-1), borda_count())

def instant_runoff() -> np.ndarray:
    prefs = _rank_prefs
    winners = np.zeros((PIXELS, PIXELS), dtype=int)
    block = 100

    for x0 in range(0, PIXELS, block):
        x1 = min(PIXELS, x0 + block)
        for y0 in range(0, PIXELS, block):
            y1 = min(PIXELS, y0 + block)
            eliminated = np.zeros((x1 - x0, y1 - y0, N_CANDIDATES), dtype=bool)
            active = np.ones((x1 - x0, y1 - y0), dtype=bool)
            voter_weights_block = _voter_weights[:, x0:x1, y0:y1]

            for _ in range(N_CANDIDATES - 1):
                if not active.any():
                    break

                still_standing = ~eliminated
                round_votes = _round_votes(
                    prefs,
                    still_standing,
                    voter_weights_block,
                    x_slice=slice(x0, x1),
                    y_slice=slice(y0, y1),
                )
                has_majority = round_votes > round_votes.sum(axis=2, keepdims=True) / 2
                got_majority = active & has_majority.any(axis=2)
                winners[x0:x1, y0:y1] = np.where(
                    got_majority,
                    has_majority.argmax(axis=2),
                    winners[x0:x1, y0:y1],
                )
                active &= ~got_majority

                if not active.any():
                    break

                vote_for_elim = np.where(still_standing, round_votes, np.inf)
                vote_for_elim = np.where(active[..., np.newaxis], vote_for_elim, -np.inf)
                to_eliminate = vote_for_elim.argmin(axis=2)
                elim_mask = np.zeros((x1 - x0, y1 - y0, N_CANDIDATES), dtype=bool)
                elim_mask[
                    np.arange(x1 - x0)[:, None],
                    np.arange(y1 - y0)[None, :],
                    to_eliminate,
                ] = True
                eliminated |= elim_mask & active[..., np.newaxis]

            if active.any():
                winners[x0:x1, y0:y1] = np.where(
                    active,
                    (~eliminated).argmax(axis=2),
                    winners[x0:x1, y0:y1],
                )

    return winners

def schulze() -> np.ndarray:
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pairwise_pref = ranks[..., :, None] < ranks[..., None, :]
    pairwise = aggregate_pairwise(pairwise_pref)

    path_strength = np.where(
        pairwise > np.transpose(pairwise, (0, 1, 3, 2)),
        pairwise,
        0,
    )

    for k in range(N_CANDIDATES):
        for i in range(N_CANDIDATES):
            if i == k:
                continue
            for j in range(N_CANDIDATES):
                if j == k or j == i:
                    continue
                path_strength[:, :, i, j] = np.maximum(
                    path_strength[:, :, i, j],
                    np.minimum(path_strength[:, :, i, k], path_strength[:, :, k, j]),
                )

    wins = path_strength >= np.transpose(path_strength, (0, 1, 3, 2))
    win_counts = wins.sum(axis=-1)
    return win_counts.argmax(axis=-1)


def baldwin() -> np.ndarray:
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pref = ranks[..., :, None] < ranks[..., None, :]
    pairwise = aggregate_pairwise(pref)

    still_standing = np.ones((PIXELS, PIXELS, N_CANDIDATES), dtype=bool)
    for _ in range(N_CANDIDATES - 1):
        scores = np.einsum('xyab,xyb->xya', pairwise, still_standing, optimize=True)
        remaining_scores = np.where(still_standing, scores, np.inf)
        to_eliminate = remaining_scores.argmin(axis=2)

        elim_mask = np.zeros_like(still_standing)
        elim_mask[
            np.arange(PIXELS)[:, None],
            np.arange(PIXELS)[None, :],
            to_eliminate,
        ] = True
        still_standing &= ~elim_mask

        if still_standing.sum(axis=-1).max() == 1:
            break

    return still_standing.argmax(axis=2)

def approval_naive(threshold: float) -> np.ndarray:
    """Approval voting naive implementation: each voter approves candidates within a fixed threshold."""
    votes = np.argmin(cardinal_dist, axis=1).reshape(grid_size, grid_size)
    voter_scores = votes[..., np.newaxis] == np.arange(N_CANDIDATES)
    voter_scores = voter_scores.astype(weights.dtype)
    approval_mask = voter_scores >= threshold
    voter_scores = np.where(approval_mask, voter_scores, 0)
    winners = aggregate(voter_scores).argmax(axis=2)
    return winners

def _condorcet_winner():
  ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
  pref = ranks[..., :, None] < ranks[..., None, :]
  pairwise = aggregate_pairwise(pref)
  beats = pairwise > np.transpose(pairwise, (0, 1, 3, 2))

  is_condorcet = np.zeros((PIXELS, PIXELS, N_CANDIDATES), dtype=bool)
  for c in range(N_CANDIDATES):
    others = [o for o in range(N_CANDIDATES) if o != c]
    is_condorcet[:, :, c] = np.all(beats[:, :, c, others], axis=-1)

  has_condorcet = is_condorcet.any(axis=-1)
  return has_condorcet, is_condorcet.argmax(axis=-1)

def condorcet_failure() -> np.ndarray:
  """Condorcet winner per pixel; N_CANDIDATES marks pixels with no Condorcet winner."""
  has_condorcet, winner = _condorcet_winner()
  return np.where(has_condorcet, winner, N_CANDIDATES)

def winners_pixels(method: Callable[[], np.ndarray]) -> np.ndarray:
    """Returns the winners of the election, shape (PIXELS, PIXELS)"""
    start_time = time.perf_counter()
    winners = method()
    print(f"Time taken: {time.perf_counter() - start_time} seconds")
    return winners


if __name__ == "__main__":
    if not os.path.exists('winners'):
        os.makedirs('winners')

    np.save('winners/fptp.npy', winners_pixels(first_past_the_post))
    np.save('winners/borda.npy', winners_pixels(borda_count))
    np.save('winners/black.npy', winners_pixels(black))
    np.save('winners/irv.npy', winners_pixels(instant_runoff))
    np.save('winners/schulze.npy', winners_pixels(schulze))
    np.save('winners/baldwin.npy', winners_pixels(baldwin))
    np.save('winners/approval.npy', winners_pixels(lambda: approval_naive(threshold=0.5)))
    np.save('winners/condorcet_failure.npy', winners_pixels(condorcet_failure))