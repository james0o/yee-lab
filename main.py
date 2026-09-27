"""Web UI for Yee diagrams.

Run `uv run fastapi dev main.py` and open http://127.0.0.1:8000.
The page itself is ui/index.html; this file only answers its requests.
"""

from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated, Literal, get_args

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import normal
from const import CANDIDATES, DEVIATION
from methods import METHODS, ideal
from ranking_cells import (
    NODES,
    SPREAD,
    SPREADS,
    TAPER,
    Spread,
    beta_params_at,
    compute_ranking_probabilities,
    interpolate_to_pixels,
    node_medians,
)

PIXELS = 300
MAX_CANDIDATES = 8  # probabilities take ~0.5 s for 5 candidates, ~4 s for 8

Coordinate = Annotated[float, Field(ge=0, le=1)]
Distribution = Literal["beta", "normal"]
# Each Beta spread rule: plain label, LaTeX label (typeset by KaTeX in the UI) and
# tooltip. All rules agree at the centre pixel.
SPREAD_INFO = {
    "mean_abs": {
        "label": "mean |X − m| (legacy)",
        "tex": r"\operatorname{E}|X-m|\ \text{(legacy)}",
        "description": "The original rule: every pixel has the same mean distance of its voters "
        "from the median. Near a wall this pushes the voters on the far side of the median "
        "away, which bends borders and makes round edges.",
    },
    "rms": {
        "label": "RMS of X − m",
        "tex": r"\sqrt{\operatorname{E}(X-m)^2}",
        "description": "Every pixel has the same root mean square distance of its voters from "
        "the median. A few distant voters are enough near a wall, so the rest stay put.",
    },
    "tapered": {
        "label": f"(a + b) × (4m(1−m))^{TAPER:g}",
        "tex": rf"(a+b)\,\bigl(4m(1-m)\bigr)^{{{TAPER:g}}}",
        "description": f"a + b shrinks towards the walls like (4m(1 − m))^{TAPER:g} times its "
        "centre value. The exponent is the result of an optimisation: it gave the straightest "
        "Condorcet borders at equal numbers of cycles. It behaves very much like the RMS rule.",
    },
}
assert set(SPREAD_INFO) == set(SPREADS)

app = FastAPI()


class DiagramRequest(BaseModel):
    candidates: list[tuple[Coordinate, Coordinate]] = Field(
        min_length=2, max_length=MAX_CANDIDATES
    )
    method: Annotated[str, Field(pattern=f"^({'|'.join(METHODS)})$")]
    distribution: Distribution = "beta"
    spread: Spread = SPREAD  # Beta only


@lru_cache(maxsize=None)
def _beta_params(spread: Spread):
    medians = node_medians(PIXELS, NODES)
    return medians, beta_params_at(medians, DEVIATION, spread)


@lru_cache(maxsize=8)
def _ranking_probabilities(
    candidates: tuple[tuple[float, float], ...],
    distribution: Distribution,
    spread: Spread | None,
):
    """Kept in memory, not in cache/: nearly every dragged position is new.
    The lru_cache makes switching methods or voter models without moving a
    candidate instant. `spread` is None for normal voters."""
    if distribution == "normal":
        return normal.ranking_probabilities(np.array(candidates), PIXELS, DEVIATION, NODES)
    medians, params = _beta_params(spread)
    rankings, probs = compute_ranking_probabilities(np.array(candidates), params)
    return rankings, interpolate_to_pixels(probs, medians, PIXELS)


@app.get("/api/config")
def config():
    return {
        "methods": list(METHODS),
        "distributions": list(get_args(Distribution)),
        "spreads": [{"name": name, **info} for name, info in SPREAD_INFO.items()],
        "candidates": CANDIDATES.tolist(),
        "max_candidates": MAX_CANDIDATES,
    }


@app.post("/api/diagram")
def diagram(request: DiagramRequest):
    candidates = tuple(request.candidates)
    spread = request.spread if request.distribution == "beta" else None
    rankings, probs = _ranking_probabilities(candidates, request.distribution, spread)
    method = METHODS[request.method]
    if method is ideal:  # the only method that needs positions, not just rankings
        method = partial(ideal, candidates=np.array(candidates))
    winners = method(rankings, probs)
    # winners[i, j] is pixel x = i, y = j; image rows go top to bottom, so row 0 is y = 1.
    rows = winners.T[::-1]
    return {"pixels": PIXELS, "winners": rows.ravel().tolist()}


# Last, so that /api/... is matched first; "/" serves ui/index.html.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui", html=True))
