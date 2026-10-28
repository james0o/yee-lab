import math

import numpy as np
import pytest

from yeelab.build import Burial
from yeelab.build.methods import METHODS
from yeelab.margin import regions
from yeelab.margin.metrics import voronoi_agreement
from yeelab.voting import CYCLE

from yeesearch.configs import SETS, Config, random_configs
from yeesearch.evaluate import (BURIED, NEEDS, NODE_COST, SINCERE, Prepared, Scorer, errors, objective, restricted,
                                scenarios)

CONFIG = Config("five", ((0.6, 0.35), (0.25, 0.4), (0.35, 0.3), (0.5, 0.5), (0.3, 0.7)), 0.2)


@pytest.fixture(scope="module")
def prepared():
    return Prepared.of(CONFIG, 12)


def test_relative_error_is_the_metric_with_cycles_worst():
    scorer = Scorer(CONFIG, 20, "marginal")
    coords = regions.grid(20, CONFIG.model.pixels)[0]
    points = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1)
    winner = regions.nearest(CONFIG.array, points)[0]
    winner = (winner + 1) % 5
    winner[:6] = CYCLE
    region = voronoi_agreement(CONFIG.array, points, winner, CONFIG.deviation, cycles="worst")["region"]
    assert scorer.relative(winner) == pytest.approx(region["error"] / region["random"])
    assert scorer.relative(regions.nearest(CONFIG.array, points)[0]) == pytest.approx(0)


@pytest.mark.parametrize("name", ["fptp", "irv", "star", "mj", "koth", "smj_two"])
def test_cached_voters_give_the_winners_of_their_own(prepared, name):
    """A method on the cached voters, cut down to the shares it reads, elects what it
    elects on voters computed for it alone, sincerely and buried, also the second time,
    from what the score ballots remember (MemoGrid)."""
    for variant in scenarios(METHODS[name]).values():
        assert variant.needs <= NEEDS
        own = regions.winners(variant, CONFIG.array, CONFIG.model, 12)[1]
        np.testing.assert_array_equal(prepared.winner(variant), own)
        np.testing.assert_array_equal(prepared.winner(variant), own)


def test_methods_beyond_the_cache_are_evaluated_on_their_own(prepared):
    method = METHODS["d21"]
    assert not method.needs <= NEEDS
    np.testing.assert_array_equal(prepared.winner(method), regions.winners(method, CONFIG.array, CONFIG.model, 12)[1])


def test_restricted_voters_keep_only_what_is_needed(prepared):
    voters = restricted(prepared.voters, METHODS["fptp"].needs)
    assert voters.first is not None and voters.pairwise is None and voters.probs is None
    assert voters.unscored is None and voters.ballots is None and voters.cells is None


def test_objective_weighs_sincere_and_burial(prepared):
    """J = mean(SINCERE sincere + BURIED burial against the default poll's frontrunners) + NODE_COST
    size; burial does not change FPTP, whose first choices stay."""
    found = errors(METHODS["fptp"], [prepared, prepared])
    assert found["sincere"] == found["poll"]
    result = objective(METHODS["fptp"], {"sincere": [0.1, 0.3], "poll": [0.2, 0.4]})
    assert result["burial"] == pytest.approx(0.3) and result["sincere"] == pytest.approx(0.2)
    assert result["J"] == pytest.approx(np.mean([SINCERE * 0.1 + BURIED * 0.2, SINCERE * 0.3 + BURIED * 0.4]) + 3 * NODE_COST)
    assert result["size"] == 3
    assert scenarios(METHODS["fptp"]) == {"sincere": METHODS["fptp"], "poll": Burial(METHODS["fptp"], 0.3)}


def test_sets_do_not_share_configurations():
    names = {name: {(c.candidates, c.deviation) for c in make()} for name, make in SETS.items()}
    assert not names["train"] & names["validation"] and not names["train"] & names["test"]
    assert not names["validation"] & names["test"]
    assert names["train"] <= names["wide"] and not names["wide"] & (names["validation"] | names["test"])
    assert len(SETS["train"]()) == 18 and len(SETS["wide"]()) == 58  # those with a candidate near a wall left out


def test_random_candidates_are_apart():
    for config in random_configs(7, 30):
        gaps = np.linalg.norm(config.array[:, None] - config.array[None], axis=-1)
        assert gaps[np.triu_indices(config.n_candidates, 1)].min() >= 0.08
        assert math.isclose(config.deviation, 0.1) or config.deviation in (0.2, 0.3)
