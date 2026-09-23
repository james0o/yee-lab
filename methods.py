import numpy as np
from scipy.spatial.distance import cdist
from cache import candidate_hash, metadata, read_metadata, winners_path
from dataclasses import dataclass, field
from pathlib import Path
from const import PIXELS as DEFAULT_PIXELS, N_PART as DEFAULT_N_PART
from const import CANDIDATES as DEFAULT_CANDIDATES, DISTANCE as DEFAULT_DISTANCE
from typing import Callable
import beta_weights
import time
from beta_weights import load_weights


@dataclass(frozen=True)
class ElectionConfig:
    pixels: int = DEFAULT_PIXELS
    n_part: int = DEFAULT_N_PART
    distance: float = DEFAULT_DISTANCE
    candidates: np.ndarray = field(
        default_factory=lambda: DEFAULT_CANDIDATES.copy()
    )
    cache_root: Path = Path("cache")


PIXELS = DEFAULT_PIXELS
N_PART = DEFAULT_N_PART
CANDIDATES = DEFAULT_CANDIDATES
N_CANDIDATES = CANDIDATES.shape[0]

def create_voters2d(voters) -> np.ndarray:
    return np.column_stack((np.repeat(voters, len(voters)), np.tile(voters, len(voters))))

def calculate_distance(voters2d, candidates) -> np.ndarray:
    return cdist(voters2d, candidates, metric='euclidean')

def configure(config: ElectionConfig) -> None:
    """Configure the module-level voting state for one election model."""
    global CANDIDATES, N_CANDIDATES, PIXELS, N_PART
    global weights, n_part, grid_size, voters2d, distance, target_distance, cache_root
    global cardinal_dist, _rank_prefs, n_voters, prefs_flat, prefs_indicator
    global _i_idx, _j_idx, _voter_weights

    candidates = np.asarray(config.candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must have shape (n_candidates, 2)")
    if config.pixels < 1 or config.n_part < 1:
        raise ValueError("pixels and n_part must be positive")

    PIXELS = config.pixels
    N_PART = config.n_part
    target_distance = float(config.distance)
    cache_root = Path(config.cache_root)
    CANDIDATES = np.ascontiguousarray(candidates)
    N_CANDIDATES = CANDIDATES.shape[0]
    weights = load_weights(
        n_part=N_PART,
        pixels=PIXELS,
        distance=config.distance,
        cache_root=config.cache_root,
    )
    n_part = N_PART
    voters = np.concatenate(([0.0], beta_weights.create_voters(n_part), [1.0]))
    grid_size = len(voters)
    voters2d = create_voters2d(voters)
    distance = calculate_distance(voters2d, CANDIDATES)
    cardinal_dist = np.argsort(distance, axis=1).argsort().astype(np.uint8)
    _rank_prefs = np.argsort(cardinal_dist, axis=1).reshape(
        grid_size, grid_size, N_CANDIDATES
    ).astype(np.uint8)
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

def plurality() -> np.ndarray:
    votes = np.argmin(cardinal_dist, axis=1).reshape(grid_size, grid_size)
    voter_scores = votes[..., np.newaxis] == np.arange(N_CANDIDATES)
    return aggregate(voter_scores.astype(weights.dtype)).argmax(axis=2)

def king_of_the_hill() -> np.ndarray:
    """Return the strongest first-preference challenger to the plurality winner."""
    first_preferences = np.argmin(cardinal_dist, axis=1).reshape(
        grid_size, grid_size
    )
    first_preference_votes = aggregate(
        (first_preferences[..., np.newaxis] == np.arange(N_CANDIDATES)).astype(
            weights.dtype
        )
    )
    plurality_winners = first_preference_votes.argmax(axis=2)

    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pairwise = aggregate_pairwise(
        ranks[..., :, np.newaxis] < ranks[..., np.newaxis, :]
    )
    opponent_votes = pairwise[
        np.arange(PIXELS)[:, np.newaxis],
        np.arange(PIXELS)[np.newaxis, :],
        :,
        plurality_winners,
    ]
    majority = opponent_votes > 0.5
    majority &= np.arange(N_CANDIDATES) != plurality_winners[..., np.newaxis]
    challenger_votes = np.where(majority, first_preference_votes, -np.inf)
    challenger = challenger_votes.argmax(axis=2)
    has_challenger = majority.any(axis=-1)
    return np.where(has_challenger, challenger, plurality_winners)

def chain_runoff() -> np.ndarray:
    """Return the first-preference-ranked candidate not beaten by the next one."""
    first_preferences = np.argmin(cardinal_dist, axis=1).reshape(
        grid_size, grid_size
    )
    first_preference_votes = aggregate(
        (first_preferences[..., np.newaxis] == np.arange(N_CANDIDATES)).astype(
            weights.dtype
        )
    )
    order = np.argsort(-first_preference_votes, axis=2, kind="stable")

    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pairwise = aggregate_pairwise(
        ranks[..., :, np.newaxis] < ranks[..., np.newaxis, :]
    )
    rows = np.arange(PIXELS)[:, np.newaxis]
    columns = np.arange(PIXELS)[np.newaxis, :]

    winner = order[..., -1]
    found = np.zeros((PIXELS, PIXELS), dtype=bool)
    for position in range(N_CANDIDATES - 1):
        current = order[..., position]
        next_candidate = order[..., position + 1]
        next_votes = pairwise[rows, columns, next_candidate, current]
        current_votes = pairwise[rows, columns, current, next_candidate]
        not_beaten = next_votes <= current_votes
        select = ~found & not_beaten
        winner = np.where(select, current, winner)
        found |= select

    return winner

def koth_chain_runoff() -> np.ndarray:
    """Use a head-to-head runoff between King of the Hill and Chain Runoff."""
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    koth_winners = king_of_the_hill()
    chain_winners = chain_runoff()
    pairwise = aggregate_pairwise(
        ranks[..., :, np.newaxis] < ranks[..., np.newaxis, :]
    )
    rows = np.arange(PIXELS)[:, np.newaxis]
    columns = np.arange(PIXELS)[np.newaxis, :]
    koth_votes = pairwise[rows, columns, koth_winners, chain_winners]
    chain_votes = pairwise[rows, columns, chain_winners, koth_winners]
    return np.where(koth_votes >= chain_votes, koth_winners, chain_winners)

def adjusted_condorcet_plurality() -> np.ndarray:
    """Return the adjusted Condorcet Plurality winner for every pixel."""
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    plurality_winners = plurality()
    adjusted_pairwise = np.zeros(
        (PIXELS, PIXELS, N_CANDIDATES, N_CANDIDATES),
        dtype=weights.dtype,
    )

    for first_winner in range(N_CANDIDATES):
        winner_rank = ranks[..., first_winner]
        retained = ranks <= winner_rank[..., np.newaxis]
        adjusted_ranks = np.where(retained, ranks, N_CANDIDATES)
        pairwise = adjusted_ranks[..., :, np.newaxis] < adjusted_ranks[..., np.newaxis, :]
        pairwise = aggregate_pairwise(pairwise)
        selected = plurality_winners == first_winner
        adjusted_pairwise = np.where(
            selected[..., np.newaxis, np.newaxis],
            pairwise,
            adjusted_pairwise,
        )

    beats = adjusted_pairwise > np.transpose(adjusted_pairwise, (0, 1, 3, 2))
    is_condorcet = np.all(
        beats | np.eye(N_CANDIDATES, dtype=bool)[np.newaxis, np.newaxis],
        axis=-1,
    )
    has_condorcet = is_condorcet.any(axis=-1)
    condorcet_winner = is_condorcet.argmax(axis=-1)
    return np.where(has_condorcet, condorcet_winner, plurality_winners)

def anti_plurality() -> np.ndarray:
    last = np.argmax(cardinal_dist, axis=1).reshape(grid_size, grid_size)
    voter_scores = last[..., np.newaxis] == np.arange(N_CANDIDATES)
    last_place_votes = aggregate(voter_scores.astype(weights.dtype))
    return last_place_votes.argmin(axis=2)

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

def _batch_runoff(elimination_sizes: tuple[int, ...]) -> np.ndarray:
    """Run plurality transfers while eliminating the requested batch sizes."""
    if sum(elimination_sizes) != N_CANDIDATES - 1:
        raise ValueError("elimination sizes must sum to N_CANDIDATES - 1")

    winners = np.zeros((PIXELS, PIXELS), dtype=int)
    block = 100
    candidate_ids = np.arange(N_CANDIDATES)

    for x0 in range(0, PIXELS, block):
        x1 = min(PIXELS, x0 + block)
        for y0 in range(0, PIXELS, block):
            y1 = min(PIXELS, y0 + block)
            eliminated = np.zeros((x1 - x0, y1 - y0, N_CANDIDATES), dtype=bool)
            block_weights = _voter_weights[:, x0:x1, y0:y1]

            for elimination_size in elimination_sizes:
                standing = ~eliminated
                round_votes = _round_votes(
                    _rank_prefs,
                    standing,
                    block_weights,
                    x_slice=slice(x0, x1),
                    y_slice=slice(y0, y1),
                )
                active_votes = np.where(standing, round_votes, np.inf)
                to_eliminate = np.argsort(active_votes, axis=2)[..., :elimination_size]
                eliminate = np.zeros_like(standing)
                eliminate |= np.any(
                    candidate_ids == to_eliminate[..., np.newaxis],
                    axis=-2,
                )
                eliminated |= eliminate

            winners[x0:x1, y0:y1] = (~eliminated).argmax(axis=2)

    return winners

def _compositions(total: int) -> tuple[tuple[int, ...], ...]:
    """Return all ordered positive-integer partitions of total."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return ((),)
    if total == 1:
        return ((1,),)

    return tuple(
        (first, *rest)
        for first in range(1, total + 1)
        for rest in _compositions(total - first)
    )

def runoff_one_then_two() -> np.ndarray:
    """Batch runoff with elimination sequence 1+2."""
    return _batch_runoff((1, N_CANDIDATES - 2))

def runoff_two_then_one() -> np.ndarray:
    """Batch runoff with elimination sequence 2+1."""
    return _batch_runoff((N_CANDIDATES - 2, 1))

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

def mmpo() -> np.ndarray:
    """Return the candidate with the smallest greatest pairwise opposition."""
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pref = ranks[..., :, None] < ranks[..., None, :]
    pairwise = aggregate_pairwise(pref)
    opposition = pairwise.max(axis=-2)
    return opposition.argmin(axis=-1)


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

def _approval_ballots(threshold: float = 0.5) -> np.ndarray:
    """Return deterministic approval ballots based on voter-specific distance quantiles."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    quantiles = np.quantile(distance, threshold, axis=1)
    below = distance < quantiles[:, np.newaxis]
    equal = distance == quantiles[:, np.newaxis]
    target_count = threshold * N_CANDIDATES
    below_count = below.sum(axis=1)
    equal_count = equal.sum(axis=1)
    equal_probability = np.divide(
        target_count - below_count,
        equal_count,
        out=np.zeros_like(quantiles),
        where=equal_count > 0,
    )

    rng = np.random.default_rng(0)
    random_ties = rng.random(distance.shape) < equal_probability[:, np.newaxis]
    return (below | (equal & random_ties)).reshape(
        grid_size, grid_size, N_CANDIDATES
    )

def approval(threshold: float = 0.5) -> np.ndarray:
    """Approve candidates within each voter's distance quantile."""
    approved = _approval_ballots(threshold)
    return aggregate(approved.astype(weights.dtype)).argmax(axis=2)

def _score_ballots(categories: int = 5) -> np.ndarray:
    """Return voter scores based on distance quantiles."""
    if not isinstance(categories, (int, np.integer)) or categories < 2:
        raise ValueError("categories must be an integer greater than or equal to 2")

    distance_order = np.argsort(distance, axis=1)
    ranks = np.argsort(distance_order, axis=1)
    score = categories - 1 - np.floor(
        ranks * categories / N_CANDIDATES
    ).astype(int)
    score = np.clip(score, 0, categories - 1)
    return score.reshape(grid_size, grid_size, N_CANDIDATES)

def score_voting(categories: int = 5) -> np.ndarray:
    """Score candidates by distance quantiles using a configurable score scale."""
    voter_scores = _score_ballots(categories)
    return aggregate(voter_scores.astype(weights.dtype)).argmax(axis=2)

def star_voting(categories: int = 5) -> np.ndarray:
    """Score Then Automatic Runoff using the two highest-scoring candidates."""
    voter_scores = _score_ballots(categories)
    total_scores = aggregate(voter_scores.astype(weights.dtype))
    top_two = np.argsort(-total_scores, axis=2, kind="stable")[..., :2]

    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pairwise = aggregate_pairwise(
        ranks[..., :, np.newaxis] < ranks[..., np.newaxis, :]
    )
    rows = np.arange(PIXELS)[:, np.newaxis]
    columns = np.arange(PIXELS)[np.newaxis, :]
    first = top_two[..., 0]
    second = top_two[..., 1]
    first_votes = pairwise[rows, columns, first, second]
    second_votes = pairwise[rows, columns, second, first]
    return np.where(first_votes >= second_votes, first, second)

def sun(categories: int = 5) -> np.ndarray:
    """Run STAR using only winners from ACP, IRV, KOTH, Chain Runoff, or Plurality."""
    candidate_ids = np.arange(N_CANDIDATES)
    possible_winners = np.stack(
        [
            adjusted_condorcet_plurality(),
            instant_runoff(),
            runoff_one_then_two(),
            runoff_two_then_one(),
            king_of_the_hill(),
            chain_runoff(),
            plurality(),
        ],
        axis=-1,
    )
    allowed = (
        possible_winners[..., np.newaxis] == candidate_ids
    ).any(axis=-2)

    voter_scores = _score_ballots(categories)
    total_scores = aggregate(voter_scores.astype(weights.dtype))
    eligible_scores = np.where(allowed, total_scores, -np.inf)
    has_runoff = allowed.sum(axis=-1) >= 2
    top_two = np.argsort(-eligible_scores, axis=2, kind="stable")[..., :2]

    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pairwise = aggregate_pairwise(
        ranks[..., :, np.newaxis] < ranks[..., np.newaxis, :]
    )
    rows = np.arange(PIXELS)[:, np.newaxis]
    columns = np.arange(PIXELS)[np.newaxis, :]
    first = top_two[..., 0]
    second = top_two[..., 1]
    first_votes = pairwise[rows, columns, first, second]
    second_votes = pairwise[rows, columns, second, first]
    runoff_winner = np.where(first_votes >= second_votes, first, second)
    return np.where(has_runoff, runoff_winner, top_two[..., 0])

def sun_small(categories: int = 5) -> np.ndarray:
    """Run STAR using winners from every sequential batch-runoff variant."""
    candidate_ids = np.arange(N_CANDIDATES)
    possible_winners = np.stack(
        [
            _batch_runoff(elimination_sizes)
            for elimination_sizes in _compositions(N_CANDIDATES - 1)
        ],
        axis=-1,
    )
    allowed = (
        possible_winners[..., np.newaxis] == candidate_ids
    ).any(axis=-2)

    voter_scores = _score_ballots(categories)
    total_scores = aggregate(voter_scores.astype(weights.dtype))
    eligible_scores = np.where(allowed, total_scores, -np.inf)
    has_runoff = allowed.sum(axis=-1) >= 2
    top_two = np.argsort(-eligible_scores, axis=2, kind="stable")[..., :2]

    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    pairwise = aggregate_pairwise(
        ranks[..., :, np.newaxis] < ranks[..., np.newaxis, :]
    )
    rows = np.arange(PIXELS)[:, np.newaxis]
    columns = np.arange(PIXELS)[np.newaxis, :]
    first = top_two[..., 0]
    second = top_two[..., 1]
    first_votes = pairwise[rows, columns, first, second]
    second_votes = pairwise[rows, columns, second, first]
    runoff_winner = np.where(first_votes >= second_votes, first, second)
    return np.where(has_runoff, runoff_winner, top_two[..., 0])

def _approval_coalitions(threshold: float, acquiescing: bool) -> np.ndarray:
    """Run descending coalitions on approval ballots with tied 0/1 scores."""
    ballots = _approval_ballots(threshold)
    candidate_ids = np.arange(N_CANDIDATES)
    n_coalitions = 1 << N_CANDIDATES
    coalition_scores = np.zeros(
        (PIXELS, PIXELS, n_coalitions),
        dtype=weights.dtype,
    )

    for coalition in range(1, n_coalitions):
        members = ((coalition >> candidate_ids) & 1).astype(bool)
        inside = candidate_ids[members]
        outside = candidate_ids[~members]
        inside_values = ballots[..., inside]
        outside_values = ballots[..., outside]

        if outside.size == 0:
            committed = np.ones((grid_size, grid_size), dtype=bool)
        elif acquiescing:
            committed = (
                inside_values.min(axis=-1) >= outside_values.max(axis=-1)
            )
        else:
            committed = (
                inside_values.min(axis=-1) > outside_values.max(axis=-1)
            )

        coalition_scores[..., coalition] = aggregate(
            committed[..., np.newaxis].astype(weights.dtype)
        )[..., 0]

    coalition_order = np.argsort(-coalition_scores, axis=-1, kind="stable")
    eligible = np.ones((PIXELS, PIXELS, N_CANDIDATES), dtype=bool)
    for position in range(n_coalitions - 1):
        coalition = coalition_order[..., position]
        coalition_members = (
            (coalition[..., np.newaxis] >> candidate_ids) & 1
        ).astype(bool)
        remaining = eligible & coalition_members
        valid = remaining.any(axis=-1)
        eligible = np.where(valid[..., np.newaxis], remaining, eligible)

        if np.all(eligible.sum(axis=-1) == 1):
            break

    return eligible.argmax(axis=-1)

def approval_dsc(threshold: float = 0.5) -> np.ndarray:
    """DSC applied to quantile-based approval ballots."""
    return _approval_coalitions(threshold, acquiescing=False)

def approval_dac(threshold: float = 0.5) -> np.ndarray:
    """DAC applied to quantile-based approval ballots with tied approvals."""
    return _approval_coalitions(threshold, acquiescing=True)

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

def descending_solid_coalitions() -> np.ndarray:
    """Return the DSC winner for every pixel."""
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    candidate_ids = np.arange(N_CANDIDATES)
    n_coalitions = 1 << N_CANDIDATES
    coalition_scores = np.zeros(
        (PIXELS, PIXELS, n_coalitions),
        dtype=weights.dtype,
    )

    for coalition in range(1, n_coalitions):
        members = ((coalition >> candidate_ids) & 1).astype(bool)
        inside = candidate_ids[members]
        outside = candidate_ids[~members]

        if outside.size == 0:
            solid = np.ones((grid_size, grid_size), dtype=bool)
        else:
            highest_inside = ranks[..., inside].max(axis=-1)
            lowest_outside = ranks[..., outside].min(axis=-1)
            solid = highest_inside < lowest_outside

        coalition_scores[..., coalition] = aggregate(
            solid[..., np.newaxis].astype(weights.dtype)
        )[..., 0]

    coalition_order = np.argsort(-coalition_scores, axis=-1, kind="stable")
    eligible = np.ones((PIXELS, PIXELS, N_CANDIDATES), dtype=bool)

    for position in range(n_coalitions - 1):
        coalition = coalition_order[..., position]
        coalition_members = (
            (coalition[..., np.newaxis] >> candidate_ids) & 1
        ).astype(bool)
        remaining = eligible & coalition_members
        valid = remaining.any(axis=-1)
        eligible = np.where(valid[..., np.newaxis], remaining, eligible)

        if np.all(eligible.sum(axis=-1) == 1):
            break

    return eligible.argmax(axis=-1)

def descending_acquiescing_coalitions() -> np.ndarray:
    """Return the DAC winner for every pixel."""
    ranks = cardinal_dist.reshape(grid_size, grid_size, N_CANDIDATES)
    candidate_ids = np.arange(N_CANDIDATES)
    n_coalitions = 1 << N_CANDIDATES
    coalition_scores = np.zeros(
        (PIXELS, PIXELS, n_coalitions),
        dtype=weights.dtype,
    )

    for coalition in range(1, n_coalitions):
        members = ((coalition >> candidate_ids) & 1).astype(bool)
        inside = candidate_ids[members]
        outside = candidate_ids[~members]

        if outside.size == 0:
            acquiescing = np.ones((grid_size, grid_size), dtype=bool)
        else:
            highest_inside = ranks[..., inside].max(axis=-1)
            lowest_outside = ranks[..., outside].min(axis=-1)
            acquiescing = highest_inside < lowest_outside

        coalition_scores[..., coalition] = aggregate(
            acquiescing[..., np.newaxis].astype(weights.dtype)
        )[..., 0]

    coalition_order = np.argsort(-coalition_scores, axis=-1, kind="stable")
    eligible = np.ones((PIXELS, PIXELS, N_CANDIDATES), dtype=bool)

    for position in range(n_coalitions - 1):
        coalition = coalition_order[..., position]
        coalition_members = (
            (coalition[..., np.newaxis] >> candidate_ids) & 1
        ).astype(bool)
        remaining = eligible & coalition_members
        valid = remaining.any(axis=-1)
        eligible = np.where(valid[..., np.newaxis], remaining, eligible)

        if np.all(eligible.sum(axis=-1) == 1):
            break

    return eligible.argmax(axis=-1)

def winners_pixels(method: Callable[[], np.ndarray]) -> np.ndarray:
    """Returns the winners of the election, shape (PIXELS, PIXELS)"""
    start_time = time.perf_counter()
    winners = method()
    print(f"Time taken: {time.perf_counter() - start_time} seconds")
    return winners


def save_winner(method_name: str, winners: np.ndarray) -> Path:
    """Save a winner array under a cache identity including its candidates."""
    path = winners_path(
        pixels=PIXELS,
        n_part=N_PART,
        distance=target_distance,
        candidates=CANDIDATES,
        method=method_name,
        cache_root=cache_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        winners=winners,
        candidates=CANDIDATES,
        metadata=np.array(
            metadata(
                "winners",
                pixels=int(PIXELS),
                n_part=int(N_PART),
                distance=target_distance,
                candidate_hash=candidate_hash(CANDIDATES),
                method=method_name,
            )
        ),
    )
    return path

def _method_registry() -> dict[str, Callable[[], np.ndarray]]:
    return {
        "plurality": plurality,
        "king_of_the_hill": king_of_the_hill,
        "chain_runoff": chain_runoff,
        "koth_chain_runoff": koth_chain_runoff,
        "acp": adjusted_condorcet_plurality,
        "anti_plurality": anti_plurality,
        "borda": borda_count,
        "black": black,
        "irv": instant_runoff,
        "schulze": schulze,
        "mmpo": mmpo,
        "baldwin": baldwin,
        "approval": lambda: approval(threshold=0.5),
        "score_voting": lambda: score_voting(categories=5),
        "star_voting": lambda: star_voting(categories=5),
        "sun": lambda: sun(categories=5),
        "sun_small": lambda: sun_small(categories=5),
        "approval_dsc": lambda: approval_dsc(threshold=0.5),
        "approval_dac": lambda: approval_dac(threshold=0.5),
        "approval_naive": lambda: approval_naive(threshold=0.5),
        "condorcet_failure": condorcet_failure,
        "dsc": descending_solid_coalitions,
        "dac": descending_acquiescing_coalitions,
    }


def _load_cached_winner(method_name: str) -> np.ndarray | None:
    path = winners_path(
        PIXELS,
        N_PART,
        target_distance,
        CANDIDATES,
        method_name,
        cache_root,
    )
    if not path.exists():
        return None

    expected = {
        "cache_version": 1,
        "kind": "winners",
        "pixels": int(PIXELS),
        "n_part": int(N_PART),
        "distance": target_distance,
        "candidate_hash": candidate_hash(CANDIDATES),
        "method": method_name,
    }
    try:
        with np.load(path) as archive:
            if read_metadata(archive) != expected:
                return None
            if not np.array_equal(archive["candidates"], CANDIDATES):
                return None
            return archive["winners"]
    except (OSError, KeyError, ValueError):
        return None


def run_all(config: ElectionConfig) -> dict[str, np.ndarray]:
    """Compute only missing winner caches for the requested configuration."""
    configure(config)
    results: dict[str, np.ndarray] = {}
    for method_name, method in _method_registry().items():
        winners = _load_cached_winner(method_name)
        if winners is None:
            winners = winners_pixels(method)
            save_winner(method_name, winners)
        else:
            print(f"Using cached {method_name}")
        results[method_name] = winners
    return results
