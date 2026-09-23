import numpy as np
import numpy.typing as npt
from typing import Annotated
from pydantic import AfterValidator
from abc import ABC, abstractmethod

from scipy.spatial.distance import cdist
from collections.abc import Callable
from pydantic import BaseModel, Field, model_validator
import beta_weights

def _shape(*exp_shape: int | None):
    def check(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != len(exp_shape) or any(
            e is not None and e != s for e, s in zip(exp_shape, arr.shape)):
            raise ValueError(f"expected ndim {len(exp_shape)}, got {arr.ndim}")
        return arr
    return AfterValidator(check)

MatrixCandiates = Annotated[npt.NDArray[np.float64], _shape(None, 2)]

class ElectionConfig(BaseModel):
    pixels: int
    n_part: int
    deviation: float
    candidates: MatrixCandiates
    
    @model_validator(mode="after")
    def init_data(self):
        self.n_candidates = self.candidates.shape[0]
        self.weights = beta_weights.load_weights(self.n_part, self.pixels, self.deviation)
        self.voters = np.concatenate(([0.0], beta_weights.create_voters(self.n_part), [1.0]))
        self.grid_size = len(self.voters)
        self.voters2d = beta_weights.create_voters2d(self.voters)
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

class ElectionMethod(BaseModel, ABC):
    name: str
    ordinal: bool
    cardinal: int | None = Field(default=None, ge=2)
    condorcet: bool
    monotonic: bool
    config: ElectionConfig

    @model_validator(mode="after")
    def validate_cardinal(self) -> "ElectionMethod":
        if not self.ordinal and self.cardinal is None:
            raise ValueError(
                "cardinal must be provided when ordinal is False"
            )
        return self

    @abstractmethod
    def run(self) -> npt.NDArray[np.uint8]:
        """Evaluate the election method and return the winners as an array of indices."""
        pass

class CondorcetMethod(ElectionMethod):
    ordinal: bool = True
    condorcet: bool = True
    monotonic: bool = True

class ShulzeMethod(CondorcetMethod):
    name: str = "Schulze"

    def run(self) -> npt.NDArray[np.uint8]:
        pass

