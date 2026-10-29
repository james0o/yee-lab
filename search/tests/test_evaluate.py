import math

import numpy as np
import pytest

from yeelab.build import Burial
from yeelab.build.methods import METHODS
from yeelab.margin import regions
from yeelab.margin.metrics import voronoi_agreement
from yeelab.voting import CYCLE

from yeesearch.configs import SETS, Config, focus_configs, random_configs
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
    focus = {(c.candidates, c.deviation) for c in focus_configs()}
    assert names["train"] - focus <= names["wide"] and not names["wide"] & (names["validation"] | names["test"] | focus)
    assert len(SETS["train"]()) == 20 and len(SETS["wide"]()) == 58  # those with a candidate near a wall left out


def test_random_candidates_are_apart():
    for config in random_configs(7, 30):
        gaps = np.linalg.norm(config.array[:, None] - config.array[None], axis=-1)
        assert gaps[np.triu_indices(config.n_candidates, 1)].min() >= 0.08
        assert math.isclose(config.deviation, 0.1) or config.deviation in (0.2, 0.3)


def test_weighted_configurations_count_more_and_are_never_trimmed():
    from yeelab.build.methods import METHODS

    from yeesearch.evaluate import objective

    plain = {"sincere": [0.1] * 20 + [0.5, 0.6], "poll": [0.1] * 20 + [0.5, 0.6]}
    assert abs(objective(METHODS["fptp"], plain)["sincere"] - 0.12) < 1e-12  # 0.6 trimmed
    weighted = plain | {"weights": [1] * 20 + [3, 3]}
    assert abs(objective(METHODS["fptp"], weighted)["sincere"] - (18 * 0.1 + 3 * 0.5 + 3 * 0.6) / 24) < 1e-12


def test_the_focus_configurations_are_in_training_only_on_their_own_grid():
    from yeesearch.configs import FOCUS_GRID, SETS, focus_configs

    names = {config.name for config in focus_configs()}
    assert names <= {config.name for config in SETS["train"]()}
    assert all(config.grid == FOCUS_GRID for config in focus_configs())
    for other in ("validation", "test", "wide"):
        assert not names & {config.name for config in SETS[other]()}


def test_a_method_behind_its_rival_ranks_below_every_one_that_is_not():
    from yeelab.build.methods import METHODS

    from yeesearch.evaluate import RIVAL_HARD, objective

    errors = {"sincere": [0.1] * 22, "poll": [0.1] * 20 + [0.3, 0.05], "weights": [1] * 20 + [3, 3],
              "rivals": [None] * 20 + [0.2, 0.1]}
    result = objective(METHODS["fptp"], errors)
    assert abs(result["behind"] - 0.5) < 1e-12  # 0.3 is 50 % above its rival, 0.05 below its
    plain = objective(METHODS["fptp"], {key: value for key, value in errors.items() if key != "rivals"})
    assert abs(result["J"] - (plain["J"] + RIVAL_HARD + 0.5)) < 1e-12
    level = objective(METHODS["fptp"], errors | {"poll": [0.1] * 20 + [0.2, 0.1]})  # as good as the rivals
    assert level["behind"] == 0
    assert level["J"] < 1 < result["J"]
