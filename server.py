"""Web UI for Yee diagrams.

Run `uv run fastapi dev server.py` and open http://127.0.0.1:8000.
The page itself is ui/index.html; this file only answers its requests.
"""

from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from const import CANDIDATES, DEVIATION
from methods import METHODS, ideal
from ranking_cells import (
    NODES,
    beta_params_at,
    compute_ranking_probabilities,
    interpolate_to_pixels,
    node_medians,
)

PIXELS = 300
MAX_CANDIDATES = 8  # probabilities take ~0.5 s for 5 candidates, ~4 s for 8

Coordinate = Annotated[float, Field(ge=0, le=1)]

app = FastAPI()


class DiagramRequest(BaseModel):
    candidates: list[tuple[Coordinate, Coordinate]] = Field(
        min_length=2, max_length=MAX_CANDIDATES
    )
    method: Annotated[str, Field(pattern=f"^({'|'.join(METHODS)})$")]


@lru_cache(maxsize=1)
def _beta_params():
    medians = node_medians(PIXELS, NODES)
    return medians, beta_params_at(medians, DEVIATION)


@lru_cache(maxsize=4)
def _ranking_probabilities(candidates: tuple[tuple[float, float], ...]):
    """Kept in memory, not in cache/: nearly every dragged position is new.
    The lru_cache makes switching methods without moving a candidate instant."""
    medians, params = _beta_params()
    rankings, probs = compute_ranking_probabilities(np.array(candidates), params)
    return rankings, interpolate_to_pixels(probs, medians, PIXELS)


@app.get("/api/config")
def config():
    return {
        "methods": list(METHODS),
        "candidates": CANDIDATES.tolist(),
        "max_candidates": MAX_CANDIDATES,
    }


@app.post("/api/diagram")
def diagram(request: DiagramRequest):
    candidates = tuple(request.candidates)
    rankings, probs = _ranking_probabilities(candidates)
    method = METHODS[request.method]
    if method is ideal:  # the only method that needs positions, not just rankings
        method = partial(ideal, candidates=np.array(candidates))
    winners = method(rankings, probs)
    # winners[i, j] is pixel x = i, y = j; image rows go top to bottom, so row 0 is y = 1.
    rows = winners.T[::-1]
    return {"pixels": PIXELS, "winners": rows.ravel().tolist()}


# Last, so that /api/... is matched first; "/" serves ui/index.html.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui", html=True))
