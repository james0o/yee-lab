from pathlib import Path
import re

import numpy as np
import pytest
from scipy import stats

from const import DISTANCE
from beta_weights import activation_function


PARAM_FILES = sorted(Path("params").glob("P*_N*.npz"))


def _archive_metadata(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"P(\d+)_N(\d+)", path.stem)
    assert match is not None
    return int(match.group(1)), int(match.group(2))


@pytest.mark.parametrize("path", PARAM_FILES, ids=lambda path: path.stem)
def test_params_match_theoretical_median_and_deviation(path: Path) -> None:
    pixels, _ = _archive_metadata(path)
    params = np.load(path)["params"]

    assert params.shape == (pixels, 2)
    assert np.issubdtype(params.dtype, np.floating)
    assert np.all(np.isfinite(params))
    assert np.all(params > 0)

    medians = (np.arange(pixels) + 0.5) / pixels
    upper_medians = np.maximum(medians, 1 - medians)
    point_weights = activation_function(upper_medians)

    alpha, beta = params.T
    beta_cdf = stats.beta.cdf(medians, alpha, beta)
    lower_half = medians < 0.5
    mixture_cdf = (1 - point_weights) * beta_cdf
    mixture_cdf[lower_half] += point_weights[lower_half]

    beta_shifted_cdf = stats.beta.cdf(medians, alpha + 1, beta)
    beta_mean = alpha / (alpha + beta)
    beta_deviation = (
        medians * (2 * beta_cdf - 1)
        + beta_mean * (1 - 2 * beta_shifted_cdf)
    )
    point_deviation = np.where(lower_half, medians, 1 - medians)
    deviations = point_weights * point_deviation + (1 - point_weights) * beta_deviation

    np.testing.assert_allclose(mixture_cdf, 0.5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(deviations, DISTANCE, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.mean(deviations), DISTANCE, rtol=0, atol=1e-12)