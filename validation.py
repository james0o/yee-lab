import numpy as np
import numpy.typing as npt
from typing import Annotated
from pydantic import AfterValidator

from pathlib import Path
from const import PIXELS as DEFAULT_PIXELS, N_PART as DEFAULT_N_PART
from const import CANDIDATES as DEFAULT_CANDIDATES, DISTANCE as DEFAULT_DISTANCE

from collections.abc import Callable
from pydantic import BaseModel, Field, model_validator

def _shape(*exp_shape: int | None):
    def check(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != len(exp_shape) or any(
            e is not None and e != s for e, s in zip(exp_shape, arr.shape)):
            raise ValueError(f"expected ndim {len(exp_shape)}, got {arr.ndim}")
        return arr
    return AfterValidator(check)

MatrixCandiates = Annotated[npt.NDArray[np.float64], _shape(None, 2)]

class ElectionConfig(BaseModel):
    pixels: int = DEFAULT_PIXELS
    n_part: int = DEFAULT_N_PART
    distance: float = DEFAULT_DISTANCE
    candidates: MatrixCandiates = DEFAULT_CANDIDATES.copy()
    cache_root: Path = Path("cache")

class ElectionMethod(BaseModel):
    name: str
    ordinal: bool
    cardinal: int | None = Field(default=None, ge=2)
    function: Callable[[], np.ndarray]
    condorcet: bool
    monotonic: bool
    election_config: ElectionConfig

    @model_validator(mode="after")
    def validate_cardinal(self) -> "ElectionMethod":
        if not self.ordinal and self.cardinal is None:
            raise ValueError(
                "cardinal must be provided when ordinal is False"
            )
        return self

class CondorcetMethod(ElectionMethod):
    ordinal: bool = True
    condorcet: bool = True
    monotonic: bool = True

class ShulzeMethod(CondorcetMethod):
    name: str = "Schulze"
    function: Callable[[], np.ndarray] = Field(default=schulze)

