import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from scipy.spatial.distance import cdist

import beta_weights
from cache import DEFAULT_CACHE_ROOT

CACHE_ROOT = DEFAULT_CACHE_ROOT
Ballot = Literal["ordinal", "cardinal", "both"]


def create_voters2d(voters: np.ndarray) -> np.ndarray:
    return np.column_stack((np.repeat(voters, len(voters)), np.tile(voters, len(voters))))


def _shape(*exp_shape: int | None):
    def check(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != len(exp_shape) or any(
            e is not None and e != s for e, s in zip(exp_shape, arr.shape)):
            raise ValueError(f"expected ndim {len(exp_shape)}, got {arr.ndim}")
        return arr
    return AfterValidator(check)

MatrixCandidates = Annotated[npt.NDArray[np.float64], _shape(None, 2)]

class ElectionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    pixels: int
    n_part: int
    deviation: float
    candidates: MatrixCandidates
    
    @model_validator(mode="after")
    def init_data(self):
        self.n_candidates = self.candidates.shape[0]
        self.weights = beta_weights.load_weights(
            self.n_part,
            self.pixels,
            distance=self.deviation,
            cache_root=CACHE_ROOT,
        )
        self.voters = np.concatenate(([0.0], beta_weights.create_voters(self.n_part), [1.0]))
        self.grid_size = len(self.voters)
        self.voters2d = create_voters2d(self.voters)
        self.distance = cdist(self.voters2d, self.candidates, metric='euclidean')
        self.ordinal_dist = np.argsort(self.distance, axis=1).argsort().astype(np.uint8)
        self._rank_prefs = np.argsort(self.ordinal_dist,axis=1,).reshape(self.grid_size,
                                                                        self.grid_size,
                                                                        self.n_candidates,).astype(np.uint8)
        self._n_voters = self.grid_size * self.grid_size
        self._prefs_flat = self._rank_prefs.reshape(
            self._n_voters,
            self.n_candidates,
        )
        self._prefs_indicator = np.eye(
            self.n_candidates,
            dtype=self.weights.dtype,
        )[self._prefs_flat]

        self._i_idx = np.repeat(
            np.arange(self.grid_size),
            self.grid_size,
        )
        self._j_idx = np.tile(
            np.arange(self.grid_size),
            self.grid_size,
        )

        self._voter_weights = (
            self.weights[:, self._i_idx].T[:, :, None]
            * self.weights[:, self._j_idx].T[:, None, :]
        )

        return self

    def aggregate(
        self,
        voter_scores: np.ndarray,
        x_slice: slice | None = None,
        y_slice: slice | None = None,
    ) -> np.ndarray:
        x_weights = self.weights[x_slice] if x_slice is not None else self.weights
        y_weights = self.weights[y_slice] if y_slice is not None else self.weights
        inner = np.einsum("ijc,yj->ciy", voter_scores, y_weights, optimize=True)
        return np.einsum("xi,ciy->xyc", x_weights, inner, optimize=True)

    def aggregate_pairwise(self, voter_pref: np.ndarray) -> np.ndarray:
        inner = np.einsum(
            "ijab,yj->abyi",
            voter_pref.astype(self.weights.dtype),
            self.weights,
            optimize=True,
        )
        return np.einsum("xi,abyi->xyab", self.weights, inner, optimize=True)

    def ranks(self) -> np.ndarray:
        return self.ordinal_dist.reshape(
            self.grid_size, self.grid_size, self.n_candidates
        )

    def pairwise(self) -> np.ndarray:
        ranks = self.ranks()
        pref = ranks[..., :, None] < ranks[..., None, :]
        return self.aggregate_pairwise(pref)

    def condorcet_winners(self) -> tuple[np.ndarray, np.ndarray]:
        pairwise = self.pairwise()
        beats = pairwise > np.transpose(pairwise, (0, 1, 3, 2))
        is_condorcet = np.all(
            beats | np.eye(self.n_candidates, dtype=bool)[np.newaxis, np.newaxis],
            axis=-1,
        )
        return is_condorcet.any(axis=-1), is_condorcet.argmax(axis=-1)

    def first_preference_votes(self) -> np.ndarray:
        first = np.argmin(self.ordinal_dist, axis=1).reshape(
            self.grid_size, self.grid_size
        )
        return self.aggregate(
            (first[..., np.newaxis] == np.arange(self.n_candidates)).astype(
                self.weights.dtype
            )
        )

class ElectionMethod(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str
    ballot: Ballot
    condorcet: bool
    monotonic: bool

    @abstractmethod
    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        """Evaluate the election method and return the winners as an array of indices."""
        pass

class CondorcetMethod(ElectionMethod):
    ballot: Ballot = "ordinal"
    condorcet: bool = True
    monotonic: bool = True

class SchulzeMethod(CondorcetMethod):
    name: str = "Schulze"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        pairwise = config.pairwise()
        n_candidates = config.n_candidates
        path_strength = np.where(
            pairwise > np.transpose(pairwise, (0, 1, 3, 2)),
            pairwise,
            0,
        )

        for k in range(n_candidates):
            for i in range(n_candidates):
                if i == k:
                    continue
                for j in range(n_candidates):
                    if j == k or j == i:
                        continue
                    path_strength[:, :, i, j] = np.maximum(
                        path_strength[:, :, i, j],
                        np.minimum(
                            path_strength[:, :, i, k],
                            path_strength[:, :, k, j],
                        ),
                    )

        wins = path_strength >= np.transpose(path_strength, (0, 1, 3, 2))
        return wins.sum(axis=-1).argmax(axis=-1).astype(np.uint8)

class MMPOMethod(CondorcetMethod):
    name: str = "MMPO"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        opposition = config.pairwise().max(axis=-2)
        return opposition.argmin(axis=-1).astype(np.uint8)

class BaldwinMethod(CondorcetMethod):
    name: str = "Baldwin"
    monotonic: bool = False

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        pairwise = config.pairwise()
        pixels = config.pixels
        n_candidates = config.n_candidates
        still_standing = np.ones((pixels, pixels, n_candidates), dtype=bool)
        for _ in range(n_candidates - 1):
            scores = np.einsum(
                "xyab,xyb->xya", pairwise, still_standing, optimize=True
            )
            remaining_scores = np.where(still_standing, scores, np.inf)
            to_eliminate = remaining_scores.argmin(axis=2)
            elim_mask = np.zeros_like(still_standing)
            elim_mask[
                np.arange(pixels)[:, None],
                np.arange(pixels)[None, :],
                to_eliminate,
            ] = True
            still_standing &= ~elim_mask
            if still_standing.sum(axis=-1).max() == 1:
                break
        return still_standing.argmax(axis=2).astype(np.uint8)

class BlackMethod(CondorcetMethod):
    name: str = "Black"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        has_condorcet, winner = config.condorcet_winners()
        voter_scores = (config.n_candidates - 1 - config.ordinal_dist).reshape(
            config.grid_size, config.grid_size, config.n_candidates
        )
        borda = config.aggregate(
            voter_scores.astype(config.weights.dtype)
        ).argmax(axis=2)
        return np.where(has_condorcet, winner, borda).astype(np.uint8)

class RunoffMethod(ElectionMethod):
    ballot: Ballot = "ordinal"
    condorcet: bool = False
    monotonic: bool = False

class PluralityMethod(RunoffMethod):
    name: str = "Plurality"
    monotonic: bool = True

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        return config.first_preference_votes().argmax(axis=2).astype(np.uint8)

class KingOfTheHillMethod(RunoffMethod):
    name: str = "King of the Hill"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        first_preference_votes = config.first_preference_votes()
        plurality_winners = first_preference_votes.argmax(axis=2)
        pairwise = config.pairwise()
        pixels = config.pixels
        opponent_votes = pairwise[
            np.arange(pixels)[:, np.newaxis],
            np.arange(pixels)[np.newaxis, :],
            :,
            plurality_winners,
        ]
        majority = opponent_votes > 0.5
        majority &= np.arange(config.n_candidates) != plurality_winners[..., np.newaxis]
        challenger_votes = np.where(majority, first_preference_votes, -np.inf)
        challenger = challenger_votes.argmax(axis=2)
        has_challenger = majority.any(axis=-1)
        return np.where(has_challenger, challenger, plurality_winners).astype(np.uint8)

class ChainRunoffMethod(RunoffMethod):
    name: str = "Chain Runoff"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        first_preference_votes = config.first_preference_votes()
        order = np.argsort(-first_preference_votes, axis=2, kind="stable")
        pairwise = config.pairwise()
        pixels = config.pixels
        rows = np.arange(pixels)[:, np.newaxis]
        columns = np.arange(pixels)[np.newaxis, :]
        winner = order[..., -1]
        found = np.zeros((pixels, pixels), dtype=bool)
        for position in range(config.n_candidates - 1):
            current = order[..., position]
            next_candidate = order[..., position + 1]
            next_votes = pairwise[rows, columns, next_candidate, current]
            current_votes = pairwise[rows, columns, current, next_candidate]
            not_beaten = next_votes <= current_votes
            select = ~found & not_beaten
            winner = np.where(select, current, winner)
            found |= select
        return winner.astype(np.uint8)

class KOTHChainRunoffMethod(RunoffMethod):
    name: str = "KOTH Chain Runoff"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        koth_winners = KingOfTheHillMethod().run(config)
        chain_winners = ChainRunoffMethod().run(config)
        pairwise = config.pairwise()
        pixels = config.pixels
        rows = np.arange(pixels)[:, np.newaxis]
        columns = np.arange(pixels)[np.newaxis, :]
        koth_votes = pairwise[rows, columns, koth_winners, chain_winners]
        chain_votes = pairwise[rows, columns, chain_winners, koth_winners]
        return np.where(
            koth_votes >= chain_votes, koth_winners, chain_winners
        ).astype(np.uint8)

class ACPMethod(RunoffMethod):
    name: str = "ACP"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        ranks = config.ranks()
        plurality_winners = PluralityMethod().run(config)
        n_candidates = config.n_candidates
        pixels = config.pixels
        adjusted_pairwise = np.zeros(
            (pixels, pixels, n_candidates, n_candidates),
            dtype=config.weights.dtype,
        )
        for first_winner in range(n_candidates):
            winner_rank = ranks[..., first_winner]
            retained = ranks <= winner_rank[..., np.newaxis]
            adjusted_ranks = np.where(retained, ranks, n_candidates)
            pairwise = adjusted_ranks[..., :, np.newaxis] < adjusted_ranks[..., np.newaxis, :]
            pairwise = config.aggregate_pairwise(pairwise)
            selected = plurality_winners == first_winner
            adjusted_pairwise = np.where(
                selected[..., np.newaxis, np.newaxis],
                pairwise,
                adjusted_pairwise,
            )
        beats = adjusted_pairwise > np.transpose(adjusted_pairwise, (0, 1, 3, 2))
        is_condorcet = np.all(
            beats | np.eye(n_candidates, dtype=bool)[np.newaxis, np.newaxis],
            axis=-1,
        )
        has_condorcet = is_condorcet.any(axis=-1)
        condorcet_winner = is_condorcet.argmax(axis=-1)
        return np.where(
            has_condorcet, condorcet_winner, plurality_winners
        ).astype(np.uint8)

class RankingMethod(ElectionMethod):
    ballot: Ballot = "ordinal"
    condorcet: bool = False
    monotonic: bool = True

    def descending_coalitions(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        ranks = config.ranks()
        candidate_ids = np.arange(config.n_candidates)
        n_coalitions = 1 << config.n_candidates
        coalition_scores = np.zeros(
            (config.pixels, config.pixels, n_coalitions),
            dtype=config.weights.dtype,
        )
        for coalition in range(1, n_coalitions):
            members = ((coalition >> candidate_ids) & 1).astype(bool)
            inside = candidate_ids[members]
            outside = candidate_ids[~members]
            if outside.size == 0:
                supporting = np.ones((config.grid_size, config.grid_size), dtype=bool)
            else:
                highest_inside = ranks[..., inside].max(axis=-1)
                lowest_outside = ranks[..., outside].min(axis=-1)
                supporting = highest_inside < lowest_outside
            coalition_scores[..., coalition] = config.aggregate(
                supporting[..., np.newaxis].astype(config.weights.dtype)
            )[..., 0]
        coalition_order = np.argsort(-coalition_scores, axis=-1, kind="stable")
        eligible = np.ones(
            (config.pixels, config.pixels, config.n_candidates), dtype=bool
        )
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
        return eligible.argmax(axis=-1).astype(np.uint8)

class AntiPluralityMethod(RankingMethod):
    name: str = "Anti-plurality"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        last = np.argmax(config.ordinal_dist, axis=1).reshape(
            config.grid_size, config.grid_size
        )
        voter_scores = last[..., np.newaxis] == np.arange(config.n_candidates)
        last_place_votes = config.aggregate(voter_scores.astype(config.weights.dtype))
        return last_place_votes.argmin(axis=2).astype(np.uint8)

class BordaMethod(RankingMethod):
    name: str = "Borda"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        voter_scores = (config.n_candidates - 1 - config.ordinal_dist).reshape(
            config.grid_size, config.grid_size, config.n_candidates
        )
        return config.aggregate(
            voter_scores.astype(config.weights.dtype)
        ).argmax(axis=2).astype(np.uint8)

class DSCMethod(RankingMethod):
    name: str = "DSC"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        return self.descending_coalitions(config)

class DACMethod(RankingMethod):
    name: str = "DAC"

    def run(self, config: ElectionConfig) -> npt.NDArray[np.uint8]:
        return self.descending_coalitions(config)
