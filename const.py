import numpy as np

PIXELS = 500
DEVIATION = 0.3
CANDIDATES = np.array([
    [0.6,0.35],
    [0.25,0.4],
    [0.35,0.3],
    [0.5, 0.5],
    [0.3, 0.7]], dtype=np.float64)
N_CANDIDATES = CANDIDATES.shape[0]