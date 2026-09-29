"""Web UI for Yee diagrams.

Run `uv run fastapi dev` and open http://127.0.0.1:8000.
The page itself is ui/index.html; this file only answers its requests.
"""

import json
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from yeelab.distributions import DISTRIBUTIONS, Distribution
from yeelab.normal import sigma_from_deviation
from yeelab.ranking_cells import SPREAD, SPREADS, TAPER, Spread, beta_params
from yeelab.regions import MARGINS, regions
from yeelab.shares import PIXELS, Model

# Methods follow a drag within ~5-30 ms; IRV needs every ranking cell and takes up to
# ~0.1 s for 8 candidates (see docs/math.typ).
MAX_CANDIDATES = 8
# The candidates the page starts with.
CANDIDATES = [[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]]
DEVIATION = 0.2  # default of the deviation slider

# The UI offers the Voronoi diagram (no voters) next to the voting methods.
DIAGRAMS = ["voronoi", *MARGINS]
# Mean absolute deviation of the voters from their pixel, as offered by the UI; DEVIATION
# is the default. At 0 every voter is at the pixel, so every method draws the Voronoi diagram.
DEVIATIONS = [round(0.05 * k, 2) for k in range(9)]
assert DEVIATION in DEVIATIONS
# Win regions are traced on a grid of this many points per axis: coarser while a
# candidate is dragged, finer once it is dropped (see regions.py).
DRAG_GRID = 160
FINAL_GRID = 320

Coordinate = Annotated[float, Field(ge=0, le=1)]
# Each voter distribution along one axis, per pixel: plain label, LaTeX label (typeset
# by KaTeX in the UI) and tooltip.
DISTRIBUTION_INFO = {
    "beta": {
        "label": "X ~ Beta(α, β)",
        "tex": r"X \sim \operatorname{Beta}(\alpha, \beta)",
        "description": "Beta voters, with the pixel as median; they stay inside the square.",
    },
    "normal": {
        "label": "X ~ N(μ, σ²)",
        "tex": r"X \sim \mathcal{N}(\mu, \sigma^2)",
        "description": "Normal voters, with the pixel as mean; they can leave the square.",
    },
}
assert set(DISTRIBUTION_INFO) == set(DISTRIBUTIONS)
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
    deviation: float = DEVIATION
    grid: int = Field(FINAL_GRID, ge=32, le=512)

    @field_validator("deviation")
    @classmethod
    def offered(cls, deviation: float) -> float:
        if deviation not in DEVIATIONS:
            raise ValueError(f"deviation must be one of {DEVIATIONS}")
        return deviation


def _warm_up():
    """Build the default voters' CDF tables (~0.2 s) and load the compiled kernels
    (compiled on the very first run, then from numba's cache) before the first
    request needs them."""
    for distribution in DISTRIBUTIONS:
        model = Model(distribution, DEVIATION, SPREAD if distribution == "beta" else None)
        for method in DIAGRAMS:
            regions(method, CANDIDATES, model, 32)


threading.Thread(target=_warm_up, daemon=True).start()


@lru_cache(maxsize=1)
def _voters():
    """Per deviation above 0: sigma of the normal voters and (a, b) of the Beta voters
    along one axis for every pixel, shape (PIXELS, 2), per spread rule. Independent of
    the candidates, so computed once."""
    return [
        {
            "deviation": deviation,
            "sigma": sigma_from_deviation(deviation),
            "beta_params": {
                spread: beta_params(PIXELS, deviation, spread).tolist() for spread in SPREADS
            },
        }
        for deviation in DEVIATIONS
        if deviation > 0
    ]


@app.get("/api/config")
def config():
    return {
        "methods": DIAGRAMS,
        "distributions": [{"name": name, **info} for name, info in DISTRIBUTION_INFO.items()],
        "spreads": [{"name": name, **info} for name, info in SPREAD_INFO.items()],
        "spread": SPREAD,
        "candidates": CANDIDATES,
        "max_candidates": MAX_CANDIDATES,
        "pixels": PIXELS,
        "drag_grid": DRAG_GRID,
        "final_grid": FINAL_GRID,
        "deviations": DEVIATIONS,
        "deviation": DEVIATION,
        # for the voter distribution plots of the hovered pixel
        "voters": _voters(),
    }


@app.post("/api/regions")
def diagram_regions(request: DiagramRequest):
    """Win region of every winner as polygons in the unit square (y up):
    {"regions": [{"winner": c, "polygons": [[outer, hole, ...], ...]}], "ms": ...},
    rings flat [x0, y0, x1, y1, ...], outer rings counter-clockwise and holes
    clockwise. winner is -1 (methods.CYCLE) for a Condorcet cycle."""
    start = time.perf_counter()
    model = None  # deviation 0: every voter at their pixel, the Voronoi diagram
    if request.deviation > 0:
        spread = request.spread if request.distribution == "beta" else None
        model = Model(request.distribution, request.deviation, spread)
    shapes = regions(request.method, request.candidates, model, request.grid)
    payload = {"regions": shapes, "ms": round(1000 * (time.perf_counter() - start), 1)}
    # json.dumps directly: FastAPI's encoder is slow on thousands of vertices
    return Response(json.dumps(payload, separators=(",", ":")), media_type="application/json")


# Last, so that /api/... is matched first; "/" serves ui/index.html.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui", html=True))
