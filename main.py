"""Web UI for Yee diagrams.

Run `uv run fastapi dev main.py` and open http://127.0.0.1:8000.
The page itself is ui/index.html; this file only answers its requests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from const import CANDIDATES, DEVIATION
from distributions import DISTRIBUTIONS, Distribution, model
from methods import CYCLE, METHODS, voronoi
from ranking_cells import NODES, SPREAD, SPREADS, TAPER, Spread

PIXELS = 300  # per axis; lower it if dragging feels slow
MAX_CANDIDATES = 8  # probabilities take ~0.5 s for 5 candidates, ~4 s for 8

# The UI offers the Voronoi diagram (no voters) next to the voting methods.
DIAGRAMS = ["voronoi", *METHODS]
CYCLE_BYTE = 255  # CYCLE (-1) in the uint8 response

Coordinate = Annotated[float, Field(ge=0, le=1)]
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
    method: Annotated[str, Field(pattern=f"^({'|'.join(DIAGRAMS)})$")]
    distribution: Distribution = "beta"
    spread: Spread = SPREAD  # Beta only


@lru_cache(maxsize=8)
def _ranking_probabilities(
    candidates: tuple[tuple[float, float], ...],
    distribution: Distribution,
    spread: Spread | None,
):
    """Kept in memory, not in cache/: nearly every dragged position is new.
    The lru_cache makes switching methods or voter models without moving a
    candidate instant. `spread` is None for normal voters."""
    module, options = model(distribution, spread)
    return module.ranking_probabilities(
        np.array(candidates), PIXELS, DEVIATION, NODES, **options
    )


@app.get("/api/config")
def config():
    return {
        "methods": DIAGRAMS,
        "distributions": list(DISTRIBUTIONS),
        "spreads": [{"name": name, **info} for name, info in SPREAD_INFO.items()],
        "candidates": CANDIDATES.tolist(),
        "max_candidates": MAX_CANDIDATES,
        "pixels": PIXELS,
    }


@app.post("/api/diagram")
def diagram(request: DiagramRequest):
    """Winner of every pixel, one byte each (CYCLE_BYTE for a Condorcet cycle),
    in image order: rows top to bottom, PIXELS x PIXELS."""
    if request.method == "voronoi":
        winners = voronoi(np.array(request.candidates), PIXELS)
    else:
        spread = request.spread if request.distribution == "beta" else None
        rankings, probs = _ranking_probabilities(
            tuple(request.candidates), request.distribution, spread
        )
        winners = METHODS[request.method](rankings, probs)
    # winners[i, j] is pixel x = i, y = j; image rows go top to bottom, so row 0 is y = 1.
    rows = winners.T[::-1]
    rows = np.where(rows == CYCLE, CYCLE_BYTE, rows).astype(np.uint8)
    return Response(rows.tobytes(), media_type="application/octet-stream")


# Last, so that /api/... is matched first; "/" serves ui/index.html.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui", html=True))
