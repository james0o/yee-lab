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
from pydantic import AfterValidator, BaseModel, Field, field_validator

from yeelab.build import Highest, Score, Tally, Winner, mixed_approval
from yeelab.distributions import DISTRIBUTIONS, Distribution
from yeelab.margin.geometric import PIXEL_MEDIAN, PIXEL_MEDIANS, PixelMedian, pixels_at
from yeelab.margin.regions import MARGINS, regions
from yeelab.margin.shares import PIXELS, Model
from yeelab.normal import sigma_from_deviation
from yeelab.ranking_cells import SPREAD, beta_params

# Methods follow a drag within ~5-30 ms; IRV needs every ranking cell and takes up to
# ~0.1 s for 8 candidates (see docs/math.typ).
MAX_CANDIDATES = 8
# The candidates the page starts with.
CANDIDATES = [[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]]
DEVIATION = 0.2  # default of the deviation slider

# The UI offers the Voronoi diagram (no voters) as the ideal scenario, below the voting methods.
IDEALS = ["voronoi"]
DIAGRAMS = [*MARGINS, *IDEALS]
# Mean absolute deviation of the voters from their pixel, as offered by the UI; DEVIATION
# is the default.
DEVIATIONS = [round(0.05 * k, 2) for k in range(1, 9)]
assert DEVIATION in DEVIATIONS
# The mix slider of "approval_mix": the share of its voters who approve half of the
# candidates, from 0 (all approve down to their largest gap, "approval_gap") to 1 (all
# approve half, "approval"). HALF_SHARE is its default. The categories slider of "score":
# the number of scores on its ballot, from 2 to MAX_LEVELS (scores 0 to 10). SCORE_LEVELS
# is its default. SLIDERS builds the two methods from a request's sliders; they are
# listed with the defaults.
APPROVAL_MIX = "approval_mix"
HALF_SHARE = 0.5
SCORE = "score"
SCORE_LEVELS = 6
MAX_LEVELS = 11
SLIDERS = {
    APPROVAL_MIX: lambda request: mixed_approval(request.half),
    SCORE: lambda request: Highest(Tally(Score(request.levels))),
}
assert MARGINS[APPROVAL_MIX] == mixed_approval(HALF_SHARE)
assert MARGINS[SCORE] == Highest(Tally(Score(SCORE_LEVELS)))
# Win regions are traced on a grid of this many points per axis: coarser while a
# candidate is dragged, finer once it is dropped (see margin/regions.py).
DRAG_GRID = 160
FINAL_GRID = 320

Coordinate = Annotated[float, Field(ge=0, le=1)]
# Each diagram: label (abbreviations in capitals, names capitalized) and tooltip; the
# methods built from blocks (yeelab.build) add the expression that builds them.
METHOD_INFO = {
    "fptp": {"label": "FPTP", "description": "First past the post: the most first choices wins."},
    "irv": {"label": "IRV", "description": "Instant runoff: the candidate with the fewest first "
            "choices among the remaining ones is eliminated, round by round, until one is left."},
    "borda": {"label": "Borda", "description": "Borda count: a voter gives C − 1 points to their "
              "first choice, C − 2 to the second, …, 0 to the last; the most points win."},
    "baldwin": {"label": "Baldwin", "description": "Baldwin: the candidate with the lowest Borda "
                "score among the remaining ones is eliminated, round by round, until one is left."},
    "nanson": {"label": "Nanson", "description": "Nanson: every candidate with a Borda score at "
               "most the mean of the remaining ones is eliminated, round by round, until one is left."},
    "schulze": {"label": "Schulze", "description": "Schulze: the candidate no one beats along the "
                "widest paths of head-to-head margins."},
    "condorcet": {"label": "Condorcet", "description": "Condorcet: the candidate who beats every "
                  "other candidate head to head; black where there is none (a cycle)."},
    "minimax": {"label": "Minimax", "description": "Minimax: the candidate whose worst head-to-head "
                "defeat is the smallest, measured by the margin of votes."},
    "black": {"label": "Black", "description": "Black: the Condorcet winner if there is one, "
              "otherwise the Borda winner."},
    "koth": {"label": "King of the hill", "description": "King of the hill: the candidate with "
             "the most first choices, unless someone beats them head to head; then the one with "
             "the most first choices among those who do."},
    "king_runoff": {"label": "King runoff", "description": "King runoff: the King of the hill "
                    "winner against the IRV winner, head to head; the one more voters rank above "
                    "the other wins."},
    "approval": {"label": "Approval", "description": "Approval: a voter approves the closest "
                 "half of the candidates; with an odd number the middle one too, if it is closer "
                 "to the candidate before it than to the one after it. The candidate approved by "
                 "the most voters wins."},
    "approval_gap": {"label": "Approval (gap)", "description": "Approval (gap): a voter puts "
                     "the candidates in order of distance and approves those above the largest "
                     "gap. The candidate approved by the most voters wins."},
    "approval_mix": {"label": "Approval (mix)", "description": "Approval (mix): both kinds of "
                     "voters. The mix slider below sets the share who approve half of the "
                     "candidates, as in Approval; the others approve those above their largest "
                     "gap, as in Approval (gap). The candidate approved by the most voters wins. "
                     "The expression is the slider's default."},
    "score": {"label": "Score", "description": "Score: a voter gives the closest candidate the "
              "top score, the farthest 0, and every other one the score in proportion to where "
              "its distance is between those two, rounded to a whole score. The highest mean "
              "score wins. The slider below sets the number of categories (scores); with two, a "
              "voter approves the candidates closer than halfway between the closest and the "
              "farthest. The expression is the slider's default."},
    "voronoi": {"label": "Voronoi", "description": "The nearest candidate to the pixel, without "
                "voters: what the ranked methods draw when all voters are at their pixel. With "
                "pixels at geometric medians it is drawn only where the voters above have one, "
                "to compare it with the methods."},
}
assert set(METHOD_INFO) == set(DIAGRAMS)
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
# The UI's Beta voters all follow one spread rule, the default one: what it keeps the same
# for every pixel, as a plain label, a LaTeX label (typeset by KaTeX in the UI) and a tooltip.
SPREAD_INFO = {
    "label": "RMS of X − m",
    "tex": r"\sqrt{\operatorname{E}(X-m)^2}",
    "description": "Every pixel has the same root mean square distance of its voters from "
    "the median along each axis.",
}
assert SPREAD == "rms"
# What a pixel is, for Beta voters: label and tooltip, in the order the UI lists them.
PIXEL_MEDIAN_INFO = {
    "marginal": {
        "label": "the median along each axis",
        "description": "A pixel is the median of its voters along x and along y. It is a median "
        "only in the directions of the axes: a slanted line through the pixel does not split "
        "its voters in half, which bends slanted borders.",
    },
    "geometric": {
        "label": "the geometric median",
        "description": "A pixel is the geometric median of its voters, the point with the "
        "smallest mean distance to them, which does not depend on the axes. The voters are the "
        "same; every election is only drawn somewhere else, closer to the centre. No voters "
        "have their geometric median next to a wall, so a strip along the walls stays empty, "
        "in the Voronoi diagram too.",
    },
}
assert set(PIXEL_MEDIAN_INFO) == set(PIXEL_MEDIANS)

app = FastAPI()


class DiagramRequest(BaseModel):
    candidates: list[tuple[Coordinate, Coordinate]] = Field(
        min_length=2, max_length=MAX_CANDIDATES
    )
    method: Annotated[str, Field(pattern=f"^({'|'.join(DIAGRAMS)})$")]
    distribution: Distribution = "beta"
    pixel_median: PixelMedian = PIXEL_MEDIAN  # Beta only
    deviation: float = DEVIATION
    half: float = Field(HALF_SHARE, ge=0, le=1)  # approval_mix only
    levels: int = Field(SCORE_LEVELS, ge=2, le=MAX_LEVELS)  # score only
    grid: int = Field(FINAL_GRID, ge=32, le=512)

    @field_validator("deviation")
    @classmethod
    def offered(cls, deviation: float) -> float:
        return _offered(deviation)


def _offered(deviation: float) -> float:
    if deviation not in DEVIATIONS:
        raise ValueError(f"deviation must be one of {DEVIATIONS}")
    return deviation


def _warm_up():
    """Build the default voters' CDF tables (~0.2 s) and their geometric medians (~0.2 s),
    and load the compiled kernels (compiled on the very first run, then from numba's
    cache) before the first request needs them."""
    for distribution in DISTRIBUTIONS:
        model = Model(distribution, DEVIATION, SPREAD if distribution == "beta" else None)
        for method in DIAGRAMS:
            regions(method, CANDIDATES, model, 32)
    regions(DIAGRAMS[0], CANDIDATES, Model("beta", DEVIATION, SPREAD), 32, "geometric")


threading.Thread(target=_warm_up, daemon=True).start()


@lru_cache(maxsize=1)
def _voters():
    """Per deviation: sigma of the normal voters and (a, b) of the Beta voters along
    one axis for every pixel, shape (PIXELS, 2). Independent of the candidates, so
    computed once."""
    return [
        {
            "deviation": deviation,
            "sigma": sigma_from_deviation(deviation),
            "beta_params": beta_params(PIXELS, deviation, SPREAD).tolist(),
        }
        for deviation in DEVIATIONS
    ]


@app.get("/api/config")
def config():
    return {
        "methods": [
            {"name": name, "label": METHOD_INFO[name]["label"],
             "description": METHOD_INFO[name]["description"]
             + (f"\n{method!r}" if isinstance(method, Winner) else "")}
            for name, method in MARGINS.items()
        ],
        "ideals": [{"name": name, **METHOD_INFO[name]} for name in IDEALS],
        "distributions": [{"name": name, **info} for name, info in DISTRIBUTION_INFO.items()],
        "spread": SPREAD_INFO,
        "pixel_medians": [{"name": name, **info} for name, info in PIXEL_MEDIAN_INFO.items()],
        "pixel_median": PIXEL_MEDIAN,
        "candidates": CANDIDATES,
        "max_candidates": MAX_CANDIDATES,
        "pixels": PIXELS,
        "drag_grid": DRAG_GRID,
        "final_grid": FINAL_GRID,
        "deviations": DEVIATIONS,
        "deviation": DEVIATION,
        # the methods of the mix slider and of the categories slider, and their defaults
        "approval_mix": APPROVAL_MIX,
        "half": HALF_SHARE,
        "score": SCORE,
        "levels": SCORE_LEVELS,
        "max_levels": MAX_LEVELS,
        # for the voter distribution plots of the hovered pixel
        "voters": _voters(),
    }


@app.post("/api/regions")
def diagram_regions(request: DiagramRequest):
    """Win region of every winner as polygons in the unit square (y up):
    {"regions": [{"winner": c, "polygons": [[outer, hole, ...], ...]}], "ms": ...},
    rings flat [x0, y0, x1, y1, ...], outer rings counter-clockwise and holes
    clockwise. winner is -1 (voting.CYCLE) for a Condorcet cycle."""
    start = time.perf_counter()
    spread = SPREAD if request.distribution == "beta" else None
    model = Model(request.distribution, request.deviation, spread)
    build = SLIDERS.get(request.method)  # built from the sliders
    method = build(request) if build else request.method
    shapes = regions(method, request.candidates, model, request.grid, request.pixel_median)
    payload = {"regions": shapes, "ms": round(1000 * (time.perf_counter() - start), 1)}
    # json.dumps directly: FastAPI's encoder is slow on thousands of vertices
    return Response(json.dumps(payload, separators=(",", ":")), media_type="application/json")


@lru_cache(maxsize=len(DEVIATIONS))  # ~0.1 MB each, built in ~0.1 s
def _pixels_at(deviation: float) -> str:
    at = pixels_at(Model("beta", deviation, SPREAD))
    return json.dumps({"pixels": at[:PIXELS // 2, :PIXELS // 2].ravel().tolist()}, separators=(",", ":"))


assert PIXELS % 2 == 0  # so a quarter of the pixels is a quarter of the square


@app.get("/api/geometric")
def geometric_pixels(deviation: Annotated[float, AfterValidator(_offered)]):
    """For the voters shown while hovering a diagram of geometric medians (Beta voters):
    {"pixels": [...]}, for the point (i + 1/2, j + 1/2) / PIXELS at [i * PIXELS / 2 + j]
    the pixel (k, l), as k * PIXELS + l, whose voters have their geometric median closest
    to the point, or -1 if no voters have it there. Only the quarter of the square next
    to the origin, i, j < PIXELS / 2: the rest are its mirror images."""
    return Response(_pixels_at(deviation), media_type="application/json")


# Last, so that /api/... is matched first; "/" serves ui/index.html.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui", html=True))
