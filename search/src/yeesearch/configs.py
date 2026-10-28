"""The candidate configurations the methods are scored on: random ones and hand-made hard
cases, in three sets that do not share any (TRAIN, VALIDATION, TEST).

A hand-made family is a function of its parameters, so the validation and test sets hold
other instances of it than the training set: rotated about the centre of the square,
spread wider or narrower, with other voters."""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from yeelab.margin.shares import Model
from yeelab.ranking_cells import SPREAD

DEVIATIONS = (0.1, 0.2, 0.3)  # not 0.05: narrow Beta voters take twice the nodes (shares.node_count)
SIZES = (3, 4, 5, 6)  # the number of candidates of a random configuration
LOW, HIGH = 0.1, 0.9  # random candidates are inside this square
SEPARATION = 0.08  # and at least this far from each other
EDGE = 0.04  # hand-made candidates are kept this far from the walls


@dataclass(frozen=True)
class Config:
    """Candidates (C, 2) and the deviation of the Beta voters around every point."""

    name: str
    candidates: tuple[tuple[float, float], ...]
    deviation: float

    @cached_property
    def array(self) -> np.ndarray:
        return np.array(self.candidates, dtype=np.float64)

    @property
    def model(self) -> Model:
        return Model("beta", self.deviation, SPREAD)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def _config(name: str, points, deviation: float) -> Config:
    points = np.clip(np.asarray(points, dtype=np.float64), EDGE, 1 - EDGE)
    return Config(name, tuple((round(float(x), 4), round(float(y), 4)) for x, y in points), deviation)


def random_configs(seed: int, count: int, sizes=SIZES, deviations=DEVIATIONS, prefix: str = "random") -> list[Config]:
    """`count` configurations of `sizes` candidates drawn in [LOW, HIGH]^2 at least
    SEPARATION apart, the voters of a random deviation of `deviations`."""
    rng = np.random.default_rng(seed)
    configs = []
    while len(configs) < count:
        n = int(rng.choice(sizes))
        points = rng.uniform(LOW, HIGH, (n, 2))
        gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)[np.triu_indices(n, 1)]
        if gaps.min() >= SEPARATION:
            configs.append(_config(f"{prefix}{seed}-{len(configs)}", points, float(rng.choice(deviations))))
    return configs


def _placed(points, angle: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """`points` relative to the centre of the square, scaled and rotated by `angle`
    degrees about it."""
    a = math.radians(angle)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return 0.5 + scale * np.asarray(points, dtype=np.float64) @ rotation.T


def dark_horse(angle=0.0, scale=1.0, gap=0.08, far=0.36):
    """Two frontrunners close to the centre and a weak third far away: where the voters of
    each frontrunner bury the other one, the dark horse can win."""
    return _placed([[-gap, 0.0], [gap, 0.0], [0.0, far]], angle, scale)


def dark_horse_four(angle=0.0, scale=1.0, gap=0.1, far=0.35):
    """dark_horse with a fourth candidate on the other side, nearer."""
    return _placed([[-gap, 0.02], [gap, -0.02], [0.03, far], [-0.05, -0.2]], angle, scale)


def center_squeeze(angle=0.0, scale=1.0, radius=0.2):
    """A candidate at the centre among three around it: the Condorcet winner with few
    first choices."""
    around = [[radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a))] for a in (90, 210, 330)]
    return _placed([[0.0, 0.0], *around], angle, scale)


def clones(angle=0.0, scale=1.0, gap=0.02):
    """Two nearly equal candidates and two others: who wins should not depend on the clone."""
    return _placed([[-0.15, 0.0], [-0.15 + gap, 0.0], [0.15, 0.0], [0.0, 0.25]], angle, scale)


def collinear(angle=0.0, scale=1.0, n=4, length=0.6):
    """n candidates evenly spaced on a line through the centre."""
    return _placed([[t, 0.0] for t in np.linspace(-length / 2, length / 2, n)], angle, scale)


def near_walls(angle=0.0, scale=1.0):
    """Candidates near the walls and corners, where the Beta voters are cut off, and one inside."""
    return _placed([[-0.42, -0.42], [0.42, -0.38], [0.0, 0.43], [0.05, 0.0]], angle, scale)


def triangle(angle=0.0, scale=1.0, radius=0.2):
    """An equilateral triangle about the centre: burial easily makes a cycle there."""
    return _placed([[radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a))]
                    for a in (90, 210, 330)], angle, scale)


def outlier(angle=0.0, scale=1.0, spread=0.06):
    """Three candidates close together and one far away."""
    return _placed([[-0.1, -0.05], [-0.1 + spread, -0.05], [-0.1, -0.05 + spread], [0.33, 0.3]], angle, scale)


FAMILIES = {"dark_horse": dark_horse, "dark_horse_four": dark_horse_four, "center_squeeze": center_squeeze,
            "clones": clones, "collinear": collinear, "near_walls": near_walls, "triangle": triangle,
            "outlier": outlier}


def hand_made(angle: float, scale: float, deviation: float, prefix: str, **options) -> list[Config]:
    """One instance of every family, `options` {family: {parameter: value}} for some."""
    return [_config(f"{prefix}{name}", family(angle, scale, **options.get(name, {})), deviation)
            for name, family in FAMILIES.items()]


def train_set() -> list[Config]:
    """12 random configurations (seed 1) and every hand-made family as drawn."""
    return random_configs(1, 12) + hand_made(0.0, 1.0, 0.2, "train-")


def validation_set() -> list[Config]:
    """20 random configurations (seed 2) and the families turned a quarter, other voters."""
    return (random_configs(2, 20)
            + hand_made(90.0, 0.9, 0.15, "validation-", collinear={"n": 3}, dark_horse={"gap": 0.06}))


def test_set() -> list[Config]:
    """200 random configurations (seed 3), the families at three other angles, scales and
    deviations, and out of the training distribution: 7 and 8 candidates, and voters
    narrower and wider than in training."""
    configs = random_configs(3, 200)
    for k, (angle, scale, deviation) in enumerate(((30.0, 1.1, 0.1), (135.0, 0.8, 0.25), (250.0, 1.0, 0.3))):
        configs += hand_made(angle, scale, deviation, f"test{k}-", collinear={"n": 5}, triangle={"radius": 0.15},
                             dark_horse={"gap": 0.1, "far": 0.3}, clones={"gap": 0.03})
    configs += random_configs(4, 20, sizes=(7, 8), prefix="ood-many")
    configs += random_configs(5, 20, deviations=(0.05, 0.4), prefix="ood-deviation")
    return configs


def wide_set() -> list[Config]:
    """The training set, 36 more random configurations (seed 6) and the families turned
    otherwise: the training of the search of Conditionals, which scores millions of
    methods and so needs more configurations not to fit chance."""
    return (train_set() + random_configs(6, 36)
            + hand_made(200.0, 1.05, 0.12, "wide-", collinear={"n": 3}, dark_horse={"gap": 0.07, "far": 0.32}))


SETS = {"train": train_set, "validation": validation_set, "test": test_set, "wide": wide_set}
