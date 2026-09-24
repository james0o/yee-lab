import numpy as np
import pytest
from scipy import stats

from const import DISTANCE
from beta_weights import generate_params


PIXEL_COUNTS = [2, 16, 32, 64, 120, 200, 250, 500]


@pytest.mark.parametrize("pixels", PIXEL_COUNTS)
def test_params_match_theoretical_median_and_deviation(pixels: int, tmp_path) -> None:
    params = generate_params(pixels, DISTANCE, cache_root=tmp_path)

    assert params.shape == (pixels, 2)
    assert np.issubdtype(params.dtype, np.floating)
    assert np.all(np.isfinite(params))
    assert np.all(params > 0)

    medians = (np.arange(pixels) + 0.5) / pixels
    alpha, beta = params.T
    beta_cdf = stats.beta.cdf(medians, alpha, beta)
    beta_shifted_cdf = stats.beta.cdf(medians, alpha + 1, beta)
    beta_mean = alpha / (alpha + beta)
    deviations = (
        medians * (2 * beta_cdf - 1)
        + beta_mean * (1 - 2 * beta_shifted_cdf)
    )

    np.testing.assert_allclose(beta_cdf, 0.5, rtol=0, atol=1e-10)
    np.testing.assert_allclose(deviations, DISTANCE, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.mean(deviations), DISTANCE, rtol=0, atol=1e-12)