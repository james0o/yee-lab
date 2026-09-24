import numpy as np
import scipy.stats as stats
from scipy.optimize import root
from pathlib import Path

from cache import (
    DEFAULT_CACHE_ROOT,
    metadata,
    params_path,
    read_metadata,
    weights_path,
)
from const import DISTANCE, N_PART, PIXELS

def create_voters(n_part):
    voters = np.arange(1 / n_part / 2, 1, 1 / n_part)
    return voters

def cdf_interval(voters, a, b):
    lower = stats.beta.cdf(x=voters - 1 / len(voters) / 2, a=a, b=b)
    upper = stats.beta.cdf(x=voters + 1 / len(voters) / 2, a=a, b=b)
    return upper - lower

def cdf_for_beta(n_part, beta_params):
    a, b = beta_params
    voters = create_voters(n_part)
    half_interval = 1 / (2 * n_part)
    return np.concatenate(
        [
            [stats.beta.cdf(half_interval, a, b)],
            cdf_interval(voters, a, b),
            [1 - stats.beta.cdf(1 - half_interval, a, b)],
        ]
    )

def create_medians(n):
    return np.arange(0.5+1/n/2, 1, 1/n)

def create_weights(median, D_target, X0):
    assert median > 0.5, "Median must be greater than 0.5"

    def system_of_equations(x):
        a, b = x[0], x[1]

        if a < 0 or b < 0:
            return [1e10, 1e10]

        cdf_ab = stats.beta.cdf(x=median, a=a, b=b)
        cdf_a1b = stats.beta.cdf(x=median, a=a + 1, b=b)
       
        eq1 = cdf_ab - 0.5
        ex = a / (a + b)
        eq2 = ex * (1 - 2 * cdf_a1b) - D_target
           
        return [eq1, eq2]

    sol = root(system_of_equations, X0, tol=1e-14, method='lm')
    a_sol, b_sol = sol.x[0], sol.x[1]
    return (a_sol, b_sol)

def generate_params(
    pixels: int,
    distance: float = DISTANCE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> np.ndarray:
    medians = create_medians(pixels)
    N_medians = len(medians)
    X0 = [1.0, 1.0]
    params = np.empty((N_medians, 2), dtype=np.float64)
    for i, m in enumerate(medians):
        (a, b) = create_weights(m, distance, X0)
        X0 = [a, b]
        params[i] = (a, b)

    mirrored_params = np.flip(params, axis=0)[:, ::-1]
    params = np.concatenate((mirrored_params, params), axis=0)
    path = params_path(pixels, distance, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        params=params,
        metadata=np.array(
            metadata("params", pixels=int(pixels), distance=float(distance))
        ),
    )
    return params


def load_params(
    pixels: int,
    distance: float = DISTANCE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> np.ndarray:
    path = params_path(pixels, distance, cache_root)
    if not path.exists():
        return generate_params(pixels, distance, cache_root)
    with np.load(path) as archive:
        saved_metadata = read_metadata(archive)
        if saved_metadata != {
            "cache_version": 1,
            "kind": "params",
            "pixels": int(pixels),
            "distance": float(distance),
        }:
            return generate_params(pixels, distance, cache_root)
        return archive["params"]


def generate_weights(
    pixels: int,
    n_part: int,
    distance: float = DISTANCE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> np.ndarray:
    params = load_params(pixels, distance, cache_root)
    medians = create_medians(pixels)
    weights = np.empty((len(medians), n_part + 2), dtype=np.float64)
    for i, _ in enumerate(medians):
        weights[i] = cdf_for_beta(n_part, params[pixels // 2 + i])

    weights = np.concatenate((np.flip(weights), weights), axis=0)
    path = weights_path(pixels, n_part, distance, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=weights,
        metadata=np.array(
            metadata(
                "weights",
                pixels=int(pixels),
                n_part=int(n_part),
                distance=float(distance),
            )
        ),
    )
    return weights


def load_weights(
    n_part: int,
    pixels: int = PIXELS,
    distance: float = DISTANCE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> np.ndarray:
    path = weights_path(pixels, n_part, distance, cache_root)
    if not path.exists():
        return generate_weights(pixels, n_part, distance, cache_root)
    with np.load(path) as archive:
        saved_metadata = read_metadata(archive)
        if saved_metadata != {
            "cache_version": 1,
            "kind": "weights",
            "pixels": int(pixels),
            "n_part": int(n_part),
            "distance": float(distance),
        }:
            return generate_weights(pixels, n_part, distance, cache_root)
        return archive["weights"]

if __name__ == "__main__":
    generate_weights(pixels=PIXELS, n_part=N_PART, distance=DISTANCE)

