import numpy as np

from const import CANDIDATES, DEVIATION, PIXELS
from ranking_cells import load_ranking_probabilities


def fptp(
    candidates: np.ndarray = CANDIDATES,
    pixels: int = PIXELS,
    deviation: float = DEVIATION,
) -> np.ndarray:
    """First past the post winner per pixel, shape (pixels, pixels)."""
    rankings, probs = load_ranking_probabilities(candidates, pixels, deviation)
    first = np.eye(len(candidates))[rankings[:, 0]]
    votes = probs @ first
    return votes.argmax(axis=-1)
