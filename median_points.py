

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
from const import DISTANCE, PIXELS

def conditional_beta_median(left, right, a, b):
    """
    Medián beta distribuce podmíněné intervalem [left, right].
    """

    cdf_left = stats.beta.cdf(left, a, b)
    cdf_right = stats.beta.cdf(right, a, b)

    probability = 0.5 * (cdf_left + cdf_right)

    return stats.beta.ppf(probability, a, b)

def create_adaptive_points(
    a,
    b,
    min_split_distance=0.1,
    max_depth=None,
):
    """
    Vytvoří adaptivní dělení intervalu [0, 1].

    Výsledek obsahuje vždy minimálně:
        [0, global_median, 1]
    """

    global_median = stats.beta.ppf(0.5, a, b)

    points = {0.0, float(global_median), 1.0}

    # Intervaly, které se ještě pokusíme rozdělit
    intervals = [
        (0.0, global_median, 0),
        (global_median, 1.0, 0),
    ]

    while intervals:
        left, right, depth = intervals.pop()

        if max_depth is not None and depth >= max_depth:
            continue

        new_median = conditional_beta_median(
            left,
            right,
            a,
            b,
        )

        # Nový bod musí být dostatečně daleko od obou hranic
        left_distance = new_median - left
        right_distance = right - new_median

        if left_distance < min_split_distance:
            continue

        if right_distance < min_split_distance:
            continue

        points.add(float(new_median))

        intervals.append((left, new_median, depth + 1))
        intervals.append((new_median, right, depth + 1))

    return np.array(sorted(points))

points = create_adaptive_points(
    a=2.0,
    b=5.0,
    min_split_distance=0.1,
)

print(points)
print("Počet bodů:", len(points))

