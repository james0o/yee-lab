"""Web UI for Yee diagrams.

Run `uv run fastapi dev` and open http://127.0.0.1:8000.
The page itself is ui/index.html; this file only answers its requests.
"""

import json
import threading
import time
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from yeelab.build import (D21, Bullet, Burial, Coarse, Compromise, Drawn, Highest, Score, ScoreComparisons, Tally,
                           Winner)
from yeelab.build.methods import FOUND, SCORE_POWER, TRIM
from yeelab.distributions import DISTRIBUTIONS, Distribution
from yeelab.margin.geometric import PIXEL_MEDIAN, PIXEL_MEDIANS, PixelMedian, outline, pixels_at
from yeelab.margin.metrics import borders, focus_outline, voronoi_agreement
from yeelab.margin.regions import MARGINS, diagram, regions, voronoi_borders
from yeelab.margin.regions import voters as voters_at
from yeelab.margin.shares import PIXELS, Model
from yeelab.normal import sigma_from_deviation
from yeelab.ranking_cells import SPREAD, beta_params
from yeelab.score import FINE

# Methods follow a drag within ~5-30 ms; IRV needs every ranking cell and takes up to
# ~0.1 s for 8 candidates (see docs/math.typ).
MAX_CANDIDATES = 8
# The candidates the page starts with.
CANDIDATES = [[0.6, 0.35], [0.25, 0.4], [0.35, 0.3], [0.5, 0.5], [0.3, 0.7]]
# The candidates the page offers to start from again, the first the one it starts with; 2 and 3
# are the two the search weighs most (search/src/yeesearch/configs.py, focus_configs).
SETUPS = {
    "Setup 1": CANDIDATES,
    "Setup 2": [[0.271, 0.713], [0.41, 0.555], [0.494, 0.476], [0.632, 0.344]],
    "Setup 3": [[0.233, 0.523], [0.639, 0.688], [0.494, 0.476], [0.57, 0.275]],
}
DEVIATION = 0.2  # default of the deviation slider
SPREAD_POWER = 2.0
# The powers p of the Beta spread rule that the UI offers: only the default, rms (math.typ,
# "Large powers"). The model takes any p from 1 to 100; listing more here, such as
# (1.0, 2.0, 4.0), offers them, and the UI shows its slider again once there are two.
SPREAD_POWERS = (SPREAD_POWER,)
assert SPREAD_POWER in SPREAD_POWERS

# The UI offers the Voronoi diagram (no voters) as the ideal scenario, below the voting methods.
IDEALS = ["voronoi"]
DIAGRAMS = [*MARGINS, *IDEALS]
# Mean absolute deviation of the voters from their pixel, as offered by the UI; DEVIATION
# is the default.
DEVIATIONS = [round(0.05 * k, 2) for k in range(1, 9)]
assert DEVIATION in DEVIATIONS
# The two sliders of the score methods: the number of scores on their ballot
# (categories), from 2 to MAX_LEVELS (scores 0 to 10), and the power p of the part of the
# way, from MIN_POWER (in proportion to the distance) to MAX_POWER by POWER_STEP.
# SCORE_LEVELS and SCORE_POWER are their defaults, the ones the methods are listed with.
SCORE_LEVELS = 6
MAX_LEVELS = 11
MIN_POWER, MAX_POWER, POWER_STEP = 1.0, 2.0, 0.05
TIE_WEIGHT = 0.0
TIE_WEIGHT_STEP = 0.05
# The trim slider of the trimmed means: the percent of the voters taken off each end,
# from 0 (the mean, Score) to MAX_TRIM, where it is majority judgment (Tally's median).
MAX_TRIM, TRIM_STEP = 50, 1


def _score_ballots(block) -> list[Score]:
    """Find Score, Coarse and D21 ballots nested in a built method: the ballots with a
    power (an ordinal D21 has none)."""
    if type(block) in (Score, Coarse) or (type(block) is D21 and not block.ordinal):
        return [block]
    if not is_dataclass(block):
        return []
    return [ballot for field in fields(block)
            for ballot in _score_ballots(getattr(block, field.name))]


def _score_ballot_settings(block, levels: int, power: float):
    """Replace adjustable Score settings while preserving fixed levels and block structure."""
    if type(block) is Score:
        return Score(block.levels if block.fixed else levels, power=power, fixed=block.fixed)
    if type(block) is Coarse:
        return Coarse(Score(FINE, power=power, fixed=True), levels=block.levels)
    if type(block) is D21 and not block.ordinal:  # fixed levels
        return replace(block, power=power)
    if not is_dataclass(block):
        return block
    updated_block = copy(block)
    changed = False
    for field in fields(block):
        value = getattr(block, field.name)
        updated = _score_ballot_settings(value, levels, power)
        if updated is not value:
            object.__setattr__(updated_block, field.name, updated)
            changed = True
    return updated_block if changed else block


def _score_comparisons(block) -> list[ScoreComparisons]:
    """Find ScoreComparisons nested in a built method."""
    if isinstance(block, ScoreComparisons):
        return [block]
    if not is_dataclass(block):
        return []
    return [comparison for field in fields(block)
            for comparison in _score_comparisons(getattr(block, field.name))]


def _score_comparison_settings(block, tie_weight: float):
    """Replace tie weights while preserving the built method's structure."""
    if isinstance(block, ScoreComparisons):
        return ScoreComparisons(block.ballot, tie_weight=tie_weight)
    if not is_dataclass(block):
        return block
    updated_block = copy(block)
    changed = False
    for field in fields(block):
        value = getattr(block, field.name)
        updated = _score_comparison_settings(value, tie_weight)
        if updated is not value:
            object.__setattr__(updated_block, field.name, updated)
            changed = True
    return updated_block if changed else block


def _trims(block) -> list[Tally]:
    """Find the trimmed means (Tally with a trim) nested in a built method."""
    if isinstance(block, Tally) and block.trim:
        return [block]
    if not is_dataclass(block):
        return []
    return [tally for field in fields(block) for tally in _trims(getattr(block, field.name))]


def _trim_settings(block, trim: float):
    """Replace the trim of the trimmed means while preserving the built method's
    structure; MAX_TRIM is the median, majority judgment."""
    if isinstance(block, Tally) and block.trim:
        if trim >= MAX_TRIM:
            return Tally(block.ballot, statistic="median")
        return Tally(block.ballot, trim=trim)
    if not is_dataclass(block):
        return block
    updated_block = copy(block)
    changed = False
    for field in fields(block):
        value = getattr(block, field.name)
        updated = _trim_settings(value, trim)
        if updated is not value:
            object.__setattr__(updated_block, field.name, updated)
            changed = True
    return updated_block if changed else block


def _drawn(method: Winner) -> Winner:
    """The method with the exact ties of its trimmed means a draw (Drawn), where it is the
    highest trimmed mean itself; anything else as it is, such as majority judgment at
    MAX_TRIM, which has a tie-break of its own."""
    if (type(method) is Highest and method.n == 1 and method.among is None
            and isinstance(method.totals, Tally) and method.totals.statistic == "mean"):
        return Drawn(method.totals)
    return method


# Whether the UI sliders apply is derived from each method's built blocks.
SCORE_METHODS = {name: _score_ballots(method) for name, method in MARGINS.items()}
SCORE_METHODS = {name: ballots for name, ballots in SCORE_METHODS.items() if ballots}
SCORE_COMPARISON_METHODS = {name: _score_comparisons(method) for name, method in MARGINS.items()}
SCORE_COMPARISON_METHODS = {name: comparisons for name, comparisons in SCORE_COMPARISON_METHODS.items()
                            if comparisons}
TRIM_METHODS = [name for name, method in MARGINS.items() if _trims(method)]
# the methods that can show the draws of their trimmed means (_drawn): those that are the
# highest trimmed mean itself, not those that take one in a condition
DRAW_METHODS = [name for name in TRIM_METHODS if _drawn(MARGINS[name]) is not MARGINS[name]]
SCORE_LEVEL_METHODS = [name for name, ballots in SCORE_METHODS.items()
                       if any(not ballot.fixed for ballot in ballots)]
SCORE_POWER_METHODS = list(SCORE_METHODS)
FIXED_SCORE_LEVELS = {
    name: sorted({ballot.levels for ballot in ballots if ballot.fixed})
    for name, ballots in SCORE_METHODS.items()
    if all(ballot.fixed for ballot in ballots)
}
assert MIN_POWER <= SCORE_POWER <= MAX_POWER
# How the voters vote: a slider for the share of the voters of each strategy, from 0 to 1
# by STRATEGIC_STEP and at 0 at first (every voter sincere), in the order the UI lists them,
# with its label, tooltip, what the status line says of it and whose voters it leaves out
# ("of": the status line says "of the others" where some vote by that one). The strategies combine
# (strategic), so every slider runs the whole way whatever the others are: burying and
# compromising are shares of the voters who do not bullet vote, each whatever the other
# is, so some voters do both. A voter who bullet votes would gain nothing by either: their
# ballot ranks the frontrunner they like less among the last already, and puts their
# first choice alone above all.
STRATEGIES = {
    "bullet": {"label": "Bullet voters", "phrase": "bullet voting", "description": "Bullet voting: this "
               "share of the voters rank their first choice first and all other candidates equal below "
               "it, and on a score ballot give their first choice the top score and all others 0. Equal "
               "candidates share the mean points of their positions (Borda), abstain head to head, and "
               "once the first choice is eliminated, the ballot counts equally for all remaining "
               "candidates."},
    "burial": {"label": "Burying voters", "phrase": "burying", "of": "bullet", "description": "Burial: "
               "the two candidates with the most first choices of the pixel are the frontrunners. This "
               "share of the voters who do not bullet vote rank the frontrunner they like less last, and "
               "on a score ballot give it 0; the rest of their ballot stays."},
    "compromise": {"label": "Compromising voters", "phrase": "compromising", "of": "bullet", "description":
                   "Compromise: of the voters who do not bullet vote, this share, whatever the burying "
                   "ones, rank the frontrunner they like more first if neither frontrunner is their first "
                   "choice, and on a score ballot give it the top score; the rest of their ballot stays. "
                   "A voter who also buries ranks one frontrunner first and the other last."},
}
BLOCKS = {"bullet": Bullet, "burial": Burial, "compromise": Compromise}  # the block of each strategy
assert list(BLOCKS) == list(STRATEGIES)
STRATEGIC_STEP = 0.05
# The tilt of every strategy (build.strategic): always to the left, the voters left of the
# centre voting strategically more and those right of it less; a slider sets its strength l,
# tilt=(-l, 0), from 0 (as many everywhere) by TILT_STEP. The odds of the left wall are e^l
# times those of the right one: at MAX_TILT about 150 times.
TILT = {"label": "More strategic to the left", "description": "Tilt of every strategy: its share is "
        "that of the voters at the centre of the square, and more of the voters left of it and fewer right "
        "of it vote by it, sigma(logit share - l (x - 1/2)). The odds of the left wall are e^l times those "
        "of the right one; 0 is the same share everywhere."}
TILT_STEP = 0.25
MAX_TILT = 5.0


def strategic(method: Winner, shares: dict[str, float], tilt: float = 0.0) -> Winner:
    """The method where shares[name] of the voters vote by each strategy of STRATEGIES,
    the first outermost: Bullet(Burial(Compromise(method, c), q), b). b of the voters
    bullet vote, and of the others q bury and c compromise, each whatever the other does.
    Compromise is innermost so that every poll reads the sincere first choices. With
    `tilt` l, every strategy is tilted to the left by it, tilt=(-l, 0) (TILT)."""
    for name in reversed(BLOCKS):
        if shares.get(name, 0) > 0:
            method = BLOCKS[name](method, shares[name], tilt=(-tilt, 0.0))
    return method


def _can(block, method: Winner) -> bool:
    try:
        block(method, 0.5)
    except ValueError:
        return False
    return True


# the methods each strategy cannot do: none so far, but a block may refuse a method
UNSUPPORTED = {name: [method for method, built in MARGINS.items() if not _can(block, built)]
               for name, block in BLOCKS.items()}
# Win regions are traced on a grid of this many points per axis: coarser while a
# candidate is dragged, finer once it is dropped (see margin/regions.py).
DRAG_GRID = 160
FINAL_GRID = 320

Coordinate = Annotated[float, Field(ge=0, le=1)]
# Each diagram: label (abbreviations in capitals, names capitalized) and tooltip; the
# methods built from blocks (yeelab.build) add the expression that builds them.
METHOD_INFO = {
    "fptp": {"label": "FPTP", "description": "First past the post: the most first choices wins."},
    "two_round": {"label": "Two-round", "description": "Two-round system: the two candidates "
                  "with the most first choices go to a second round, head to head; the one more "
                  "voters rank above the other wins."},
    "irv": {"label": "IRV", "description": "Instant runoff: the candidate with the fewest first "
            "choices among the remaining ones is eliminated, round by round, until one is left."},
    "dac": {"label": "DAC", "description": "Descending acquiescing coalitions: every set of "
            "candidates scores the share of voters who rank all of it above all other "
            "candidates. From the highest score down, each set makes the candidates outside it "
            "ineligible, unless that would leave no one eligible; the last one eligible wins."},
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
    "score": {"label": "Score", "description": "Score: a voter gives the closest candidate the "
              "top score, the farthest 0, and every other one a score by where its distance is "
              "between those two: its part of the way from the farthest to the closest, to the "
              "power p, rounded to a whole score. The highest mean score wins. The sliders below "
              "set the number of categories (scores) and p: 1 is in proportion to the distance, "
              "above 1 the top scores stay with the candidates near the closest, as on real "
              "ballots. With two categories a voter approves the candidates beyond 2^(-1/p) of "
              "the way: halfway for p = 1, 0.63 for the default 1.5. The expression is the "
              "sliders' defaults."},
    "star": {"label": "STAR", "description": "STAR: the two candidates with the highest total "
             "score advance to a runoff. Each voter gives the runoff vote to the finalist they "
             "scored higher; voters who gave both finalists the same score abstain by default. "
             "The runoff tie-weight slider lets those voters use their ordinal preference, from "
             "0 (STAR) to 1 (ordinary pairwise runoff). The score categories and power use the "
             "same sliders as Score."},
    "d21": {"label": "D21", "description": "D21 (Janeček): a voter gives at most 2 plus votes, to the "
            "closest candidates, and at most 1 minus vote, to the farthest, but a minus vote only "
            "after 2 plus votes. The voter starts from the Score ballot of 3 categories: a plus vote "
            "for the top score, a minus vote for 0. If that leaves them without a minus vote and the "
            "approval ballot of 2 categories approves more candidates, they vote that instead: plus "
            "votes for the candidates it approves, a minus vote for the farthest. The most plus votes "
            "less minus votes win. The power uses the same slider as Score; the categories are fixed."},
    "mj": {"label": "Majority judgment", "description": "Majority judgment: a voter grades every "
           "candidate on the Score ballot, the closest the top grade and the farthest 0. The "
           "highest median grade wins (where the voters split evenly, the lower one). Candidates "
           "with the same median go by Balinski and Laraki's tie-break: the median grades are "
           "taken out and the medians of the rest compared, here a share of the voters at a "
           "time, since the voters of a pixel are a distribution. With two categories it is "
           "approval voting. The score categories and power use the same sliders as Score."},
    "trimmed": {"label": "Trimmed mean", "description": "Trimmed mean: a voter scores every "
                "candidate on the Score ballot. Of the voters of a candidate, in the order of the "
                "score they give it, the trim slider's percent are taken off each end, and the "
                "highest mean score of the rest wins: 0 is Score, 25 the interquartile mean, and "
                "50 majority judgment, the median with its tie-break. The score categories and "
                "power use the same sliders as Score."},
    "coarse_3": {"label": "Score 6 → 3", "description": "The trimmed mean counted in three groups: "
                 "a voter marks the Score ballot of 6 categories (scores 0 to 5), and the scores 0-1, "
                 "2-3 and 4-5 count as 0, 1 and 2. The highest trimmed mean of the groups wins, as "
                 "for Trimmed mean, by the same trim slider: 0 is the mean. The thresholds are those "
                 "of the six scores, 0.3 and 0.7 of the way to the power p, not those of Score with "
                 "3 categories (0.25 and 0.75). The power uses the same slider as Score; the "
                 "categories are fixed."},
    "coarse_2": {"label": "Score 6 → 2", "description": "The trimmed mean counted in two groups: a "
                 "voter marks the Score ballot of 6 categories (scores 0 to 5), and the scores 3 to 5 "
                 "count as an approval, which is Score's approval ballot of 2 categories. The highest "
                 "trimmed mean of the approvals wins, by the same trim slider as Trimmed mean: at 0 "
                 "it is approval voting. The power uses the same slider as Score; the categories are "
                 "fixed."},
    "voronoi": {"label": "Voronoi", "description": "The nearest candidate to the pixel, without "
                "voters: what the ranked methods draw when all voters are at their pixel. With "
                "pixels at geometric medians it is drawn only where the voters above have one, "
                "to compare it with the methods."},
}
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
# What the spread rule of the UI's Beta voters keeps the same for every pixel, as a plain
# label, a LaTeX label (typeset by KaTeX in the UI) and a tooltip: the RMS distance while only
# p = 2 is offered, the p-th power mean once there are more (the UI puts the power for P).
RMS_INFO = {
    "label": "RMS distance from median",
    "tex": r"\sqrt{\operatorname{E}(X-m)^2}",
    "description": "Every pixel has the same root mean square distance of its voters from "
    "the median along each axis.",
}
POWER_MEAN_INFO = {
    "label": "Lᵖ distance from median",
    "tex": r"\left(\operatorname{E}|X-m|^{P}\right)^{1/P}",
    "description": "Every pixel has the same p-th power mean distance of its voters from "
    "the median along each axis. At p = 1 this is mean absolute deviation; at p = 2 it is RMS.",
}
SPREAD_INFO = RMS_INFO if SPREAD_POWERS == (2.0,) else POWER_MEAN_INFO
assert SPREAD == "rms"


def _spread_rule(power: float):
    return SPREAD if power == 2 else power


# What a pixel is, for Beta voters: plain label, LaTeX label (typeset by KaTeX in the UI)
# and tooltip, in the order the UI lists them. V = (V_x, V_y) is a voter of the pixel p.
PIXEL_MEDIAN_INFO = {
    "geometric": {
        "label": "the geometric median",
        "tex": r"p = \operatorname*{arg\,min}_q \operatorname{E}\lVert V - q \rVert",
        "description": "A pixel is the geometric median of its voters, the point with the "
        "smallest mean distance to them, which does not depend on the axes. The voters are the "
        "same; every election is only drawn somewhere else, closer to the centre. No voters "
        "have their geometric median next to a wall, so a strip along the walls stays empty, "
        "in the Voronoi diagram too. The candidates are kept within the coloured region.\n"
        "p = argmin_q E‖V − q‖",
    },
    "marginal": {
        "label": "the median along each axis",
        "tex": r"p = (\operatorname{med} V_x, \operatorname{med} V_y)",
        "description": "A pixel is the median of its voters along x and along y. It is a median "
        "only in the directions of the axes: a slanted line through the pixel does not split "
        "its voters in half, which bends slanted borders.\n"
        "p = (med V_x, med V_y), so P(V_x ≤ p_x) = P(V_y ≤ p_y) = 1/2",
    },
}
assert set(PIXEL_MEDIAN_INFO) == set(PIXEL_MEDIANS)
UI_PIXEL_MEDIAN: PixelMedian = "geometric"  # what the UI starts with; the API's is the library's

app = FastAPI()


def _offered_spread_power(spread_power: float) -> float:
    if spread_power not in SPREAD_POWERS:
        raise ValueError(f"spread_power must be one of {SPREAD_POWERS}")
    return spread_power


SpreadPower = Annotated[float, AfterValidator(_offered_spread_power)]


class DiagramRequest(BaseModel):
    candidates: list[tuple[Coordinate, Coordinate]] = Field(
        min_length=2, max_length=MAX_CANDIDATES
    )
    method: Annotated[str, Field(pattern=f"^({'|'.join(DIAGRAMS)})$")]
    distribution: Distribution = "beta"
    pixel_median: PixelMedian = PIXEL_MEDIAN  # Beta only
    # "geometric" only: how far the pixels are drawn from their medians along the axes
    # towards their geometric median. The UI moves the diagram between the two in steps.
    shift: float = Field(1.0, ge=0, le=1)
    deviation: float = DEVIATION
    spread_power: SpreadPower = SPREAD_POWER
    levels: int = Field(SCORE_LEVELS, ge=2, le=MAX_LEVELS)  # SCORES only
    power: float = Field(SCORE_POWER, ge=MIN_POWER, le=MAX_POWER)  # SCORES only
    tie_weight: float = Field(TIE_WEIGHT, ge=0, le=1)  # ScoreComparisons only
    trim: float = Field(TRIM, ge=0, le=MAX_TRIM)  # trimmed means only
    draws: bool = False  # DRAW_METHODS only: an exact tie of the trimmed means is a draw (_drawn)
    # the share of the voters of each strategy (STRATEGIES), of those not voting by the ones before
    bullet: float = Field(0.0, ge=0, le=1)
    burial: float = Field(0.0, ge=0, le=1)
    compromise: float = Field(0.0, ge=0, le=1)
    tilt: float = Field(0.0, ge=0, le=MAX_TILT)  # of every strategy, to the left (TILT)
    grid: int = Field(FINAL_GRID, ge=32, le=512)
    # The comparison with the Voronoi diagram: the metrics, and its borders to draw over the
    # diagram (overlay). The UI asks for neither while a candidate is dragged (the borders only
    # while they are shown): they take longer than most methods themselves.
    metrics: bool = True
    overlay: bool = True

    @field_validator("deviation")
    @classmethod
    def offered(cls, deviation: float) -> float:
        return _offered(deviation)

    @model_validator(mode="after")
    def supported(self):
        for name in STRATEGIES:
            if getattr(self, name) > 0 and self.method in UNSUPPORTED[name]:
                raise ValueError(f"{STRATEGIES[name]['label']} cannot vote in {self.method}")
        return self


def _offered(deviation: float) -> float:
    if deviation not in DEVIATIONS:
        raise ValueError(f"deviation must be one of {DEVIATIONS}")
    return deviation


# The methods the warm-up runs: one for every compiled kernel and kind of share. The others
# are built from the same blocks, and running all of them took the start a minute longer.
WARM = [name for name in ("fptp", "irv", "dac", "nanson", "baldwin", "schulze", "koth", "star", "mj", "trimmed",
                          "system321", "d21", "d20", "coarse_3", "voronoi") if name in DIAGRAMS]


def _warm_up():
    """Build the default voters' CDF tables (~0.2 s) and their geometric medians (~0.2 s),
    and load the compiled kernels (compiled on the very first run, then from numba's
    cache) before the first request needs them."""
    for distribution in DISTRIBUTIONS:
        model = Model(distribution, DEVIATION, SPREAD if distribution == "beta" else None)
        for method in WARM:
            regions(method, CANDIDATES, model, 32)
    regions(DIAGRAMS[0], CANDIDATES, Model("beta", DEVIATION, SPREAD), 32, "geometric")
    # the kernels of the score shares of strategic voters: STAR reads them all
    regions(Burial(Compromise(MARGINS["star"], 0.5), 0.5), CANDIDATES, Model("beta", DEVIATION, SPREAD), 32)
    # the kernel of the comparison with the Voronoi diagram
    shapes, points, winner, _ = diagram(DIAGRAMS[0], CANDIDATES, Model("beta", DEVIATION, SPREAD), 32)
    voronoi_agreement(CANDIDATES, points, winner, DEVIATION)


WARM_UP = threading.Thread(target=_warm_up, daemon=True)  # tests wait for it (tests/conftest.py)
WARM_UP.start()


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


def _method_option(name: str, method: Winner) -> dict[str, str]:
    """The UI label and tooltip, with useful defaults for methods without overrides."""
    info = METHOD_INFO.get(name, {})
    label = info.get("label", name.replace("_", " ").title())
    expression = repr(method)
    description = info.get("description")
    if name in SCORE_COMPARISON_METHODS and name != "star":
        tie_description = ("The runoff tie-weight slider counts ordinal preferences when both "
                           "candidates received the same score: 0 abstains (STAR), 1 is ordinary "
                           "pairwise runoff.")
        description = f"{description}\n{tie_description}" if description else tie_description
    return {
        "name": name,
        "label": label,
        "description": f"{description}\n{expression}" if description else expression,
    }


@app.get("/api/config")
def config():
    return {
        "methods": [_method_option(name, method) for name, method in MARGINS.items() if name not in FOUND],
        "found": [_method_option(name, MARGINS[name]) for name in FOUND],  # by the search, a row of their own
        "ideals": [{"name": name, **METHOD_INFO[name]} for name in IDEALS],
        "distributions": [{"name": name, **info} for name, info in DISTRIBUTION_INFO.items()],
        "spread": SPREAD_INFO,
        "pixel_medians": [{"name": name, **info} for name, info in PIXEL_MEDIAN_INFO.items()],
        "pixel_median": UI_PIXEL_MEDIAN,
        "candidates": CANDIDATES,
        "setups": [{"name": name, "candidates": candidates} for name, candidates in SETUPS.items()],
        "max_candidates": MAX_CANDIDATES,
        "pixels": PIXELS,
        "drag_grid": DRAG_GRID,
        "final_grid": FINAL_GRID,
        "deviations": DEVIATIONS,
        "deviation": DEVIATION,
        "spread_power": SPREAD_POWER,
        "spread_powers": SPREAD_POWERS,
        # the methods of the categories and power sliders, their defaults and ranges
        "scores": SCORE_POWER_METHODS,
        "score_levels": SCORE_LEVEL_METHODS,
        "score_comparisons": list(SCORE_COMPARISON_METHODS),
        "fixed_score_levels": FIXED_SCORE_LEVELS,
        "levels": SCORE_LEVELS,
        "max_levels": MAX_LEVELS,
        "power": SCORE_POWER,
        "min_power": MIN_POWER,
        "max_power": MAX_POWER,
        "power_step": POWER_STEP,
        "tie_weight": TIE_WEIGHT,
        "tie_weight_step": TIE_WEIGHT_STEP,
        "trims": TRIM_METHODS,
        "trim": TRIM,
        "max_trim": MAX_TRIM,
        "trim_step": TRIM_STEP,
        "draws": DRAW_METHODS,
        # a slider for the share of the voters of each strategy, and the methods it cannot do
        "strategies": [{"name": name, **info} for name, info in STRATEGIES.items()],
        "unsupported": UNSUPPORTED,
        "strategic_step": STRATEGIC_STEP,
        "tilt": {**TILT, "max": MAX_TILT, "step": TILT_STEP},
        # for the voter distribution plots of the hovered pixel
        "voters": _voters(),
    }


@app.post("/api/regions")
def diagram_regions(request: DiagramRequest):
    """Win region of every winner as polygons in the unit square (y up):
    {"regions": [{"winner": c, "polygons": [[outer, hole, ...], ...]}], "metrics": {"diagram": ..., "region": ...},
    "overlays": {"focus": ..., "voronoi_boundaries": ...}, "ms": ...}, rings flat [x0, y0, x1, y1, ...],
    outer rings counter-clockwise and holes clockwise. winner is -1 (voting.CYCLE) for a
    Condorcet cycle, and below it for a draw (voting.draw: the bits of -1 - winner are its
    candidates), which a method of DRAW_METHODS has with `draws`. `bullet` of the voters bullet vote, and of the others `burial` bury and
    `compromise` compromise (strategic), more of them to the left by `tilt`. metrics is null unless `metrics`, and the
    overlays have no voronoi_boundaries unless `overlay`. focus is always there: the border
    of the region that the metrics of "region" average over, [x0, y0, x1, y1, ...] counter-clockwise."""
    start = time.perf_counter()
    spread = _spread_rule(request.spread_power) if request.distribution == "beta" else None
    model = Model(request.distribution, request.deviation, spread)
    if request.method in SCORE_METHODS or request.method in SCORE_COMPARISON_METHODS:
        method = _trim_settings(MARGINS[request.method], request.trim)
        method = _score_ballot_settings(method, request.levels, request.power)
        method = _score_comparison_settings(method, request.tie_weight)
        if request.draws and request.method in DRAW_METHODS:
            method = _drawn(method)
    else:
        method = request.method
    shares = {name: getattr(request, name) for name in STRATEGIES}
    if method != "voronoi" and any(shares.values()):
        method = strategic(MARGINS[method] if isinstance(method, str) else method, shares, request.tilt)
    shapes, points, winner, _ = diagram(
        method, request.candidates, model, request.grid, request.pixel_median, request.shift
    )
    focus = focus_outline(request.candidates, request.deviation)
    payload = {"regions": shapes, "metrics": None,
               "overlays": {"focus": np.round(focus, 5).ravel().tolist()}}
    if request.metrics:
        payload["metrics"] = voronoi_agreement(request.candidates, points, winner, request.deviation)
    if request.overlay:  # [x0, y0, x1, y1] per segment
        # the Voronoi diagram is drawn with its own borders; the others with the exact ones,
        # which its traced borders only miss next to a point where three meet
        reference = borders(shapes) if request.method == "voronoi" else voronoi_borders(
            request.candidates, model, request.grid, request.pixel_median, request.shift
        )
        payload["overlays"]["voronoi_boundaries"] = np.round(np.reshape(reference[0], (-1, 4)), 6).tolist()
    payload["ms"] = round(1000 * (time.perf_counter() - start), 1)
    # json.dumps directly: FastAPI's encoder is slow on thousands of vertices
    return Response(json.dumps(payload, separators=(",", ":")), media_type="application/json")


@lru_cache(maxsize=32)
def _pixels_at(deviation: float, spread_power: float) -> str:
    model = Model("beta", deviation, _spread_rule(spread_power))
    at = pixels_at(model)
    payload = {"pixels": at[:PIXELS // 2, :PIXELS // 2].ravel().tolist(),
               "outline": np.round(outline(model), 6).ravel().tolist()}
    return json.dumps(payload, separators=(",", ":"))


assert PIXELS % 2 == 0  # so a quarter of the pixels is a quarter of the square


@app.get("/api/geometric")
def geometric_pixels(
    deviation: Annotated[float, AfterValidator(_offered)],
    spread_power: SpreadPower = SPREAD_POWER,
):
    """For a diagram of geometric medians (Beta voters): {"pixels": [...], "outline": [...]}.
    pixels, for the voters shown while hovering: for the point (i + 1/2, j + 1/2) / PIXELS
    at [i * PIXELS / 2 + j] the pixel (k, l), as k * PIXELS + l, whose voters have their
    geometric median closest to the point, or -1 if no voters have it there. Only the
    quarter of the square next to the origin, i, j < PIXELS / 2: the rest are its mirror
    images. outline, which the candidates are kept within: the border of the coloured
    region, [x0, y0, x1, y1, ...] counter-clockwise (geometric.outline)."""
    return Response(_pixels_at(deviation, spread_power), media_type="application/json")


@lru_cache(maxsize=32)
def _voters_at(deviation: float, spread_power: float) -> str:
    return json.dumps(
        {
            "deviation": deviation,
            "sigma": sigma_from_deviation(deviation),
            "beta_params": beta_params(PIXELS, deviation, _spread_rule(spread_power)).tolist(),
        },
        separators=(",", ":"),
    )


@app.get("/api/voters")
def voter_parameters(
    deviation: Annotated[float, AfterValidator(_offered)],
    spread_power: SpreadPower = SPREAD_POWER,
):
    """Beta parameters for the pixel medians at the selected deviation and Lp power."""
    return Response(_voters_at(deviation, spread_power), media_type="application/json")


# The coarse-trim chart of the hovered pixel: for each row, a method of the trim slider
# with the categories it shows (the trimmed mean of Score always at 6), its winner at every
# trim from 0 to MAX_TRIM by TRIM_STEP.
TRIM_ROWS = (("coarse_2", 2), ("coarse_3", 3), ("trimmed", 6))
assert all(name in DRAW_METHODS for name, _ in TRIM_ROWS)
TRIMS = tuple(range(0, MAX_TRIM + 1, TRIM_STEP))


@dataclass(frozen=True, eq=False)
class _Side(Winner):
    """The winners of several methods side by side, (..., M), on the same voters: the
    strategies around it move the voters once for all of them. Its margin is inf, so
    theirs is the poll's."""

    methods: tuple[Winner, ...]

    @property
    def needs(self):
        return frozenset().union(*(method.needs for method in self.methods))

    def evaluate(self, voters):
        winners = np.stack([method.evaluate(voters)[0] for method in self.methods], axis=-1)
        return winners, np.full(voters.shape, np.inf)


class TrimRequest(BaseModel):
    candidates: list[tuple[Coordinate, Coordinate]] = Field(min_length=2, max_length=MAX_CANDIDATES)
    median: tuple[Coordinate, Coordinate]  # of the voters of the hovered pixel
    distribution: Distribution = "beta"
    deviation: float = DEVIATION
    spread_power: SpreadPower = SPREAD_POWER
    power: float = Field(SCORE_POWER, ge=MIN_POWER, le=MAX_POWER)
    bullet: float = Field(0.0, ge=0, le=1)
    burial: float = Field(0.0, ge=0, le=1)
    compromise: float = Field(0.0, ge=0, le=1)
    tilt: float = Field(0.0, ge=0, le=MAX_TILT)
    draws: bool = False  # an exact tie of the trimmed means is a draw (_drawn)

    @field_validator("deviation")
    @classmethod
    def offered(cls, deviation: float) -> float:
        return _offered(deviation)


@app.post("/api/trims")
def trim_winners(request: TrimRequest):
    """The winner of each row of TRIM_ROWS at every trim of TRIMS for the voters with
    the median `median`, sincere or strategic as the sliders say: {"trims": [...], "rows":
    [{"name": ..., "levels": ..., "winners": [...]}], "ms": ...}. At MAX_TRIM a row is the
    median, majority judgment on its ballot. With `draws` a winner below -1 is a draw
    (voting.draw) of the trimmed means, but for majority judgment."""
    start = time.perf_counter()
    spread = _spread_rule(request.spread_power) if request.distribution == "beta" else None
    model = Model(request.distribution, request.deviation, spread)
    methods = [_score_ballot_settings(_trim_settings(MARGINS[name], trim), levels, request.power)
               for name, levels in TRIM_ROWS for trim in TRIMS]
    if request.draws:
        methods = [_drawn(method) for method in methods]
    shares = {name: getattr(request, name) for name in STRATEGIES}
    method = strategic(_Side(tuple(methods)), shares, request.tilt)
    x, y = request.median
    winners = method.evaluate(voters_at(method.needs, request.candidates, model, medians=[x, y]))[0][0, 1]
    rows = [{"name": name, "levels": levels, "winners": winners[k * len(TRIMS):(k + 1) * len(TRIMS)].tolist()}
            for k, (name, levels) in enumerate(TRIM_ROWS)]
    return {"trims": list(TRIMS), "rows": rows, "ms": round(1000 * (time.perf_counter() - start), 1)}


# Last, so that /api/... is matched first; "/" serves ui/index.html.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui", html=True))
