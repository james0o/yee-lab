"""Voter distributions around each pixel.

ranking_cells (beta) and normal have the same functions ranking_probabilities,
read_cached_ranking_probabilities and generate_ranking_probabilities, taking
(candidates, pixels, deviation, nodes, ...); only beta takes a spread rule.
"""

from typing import Literal, get_args

import normal
import ranking_cells
from ranking_cells import SPREAD, Spread

Distribution = Literal["beta", "normal"]
DISTRIBUTIONS = get_args(Distribution)


def model(distribution: Distribution, spread: Spread = SPREAD):
    """(module, keyword options) for a voter distribution, used as
    module.ranking_probabilities(candidates, pixels, deviation, nodes, **options)."""
    if distribution == "beta":
        return ranking_cells, {"spread": spread}
    return normal, {}
