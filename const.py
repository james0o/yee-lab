import numpy as np

PIXELS = 500
DISTANCE = 0.2
N_PART = 16
CANDIDATES = np.array([
    [0.5,0.5],
    [0.4,0.4],
    [0.3,0.5]], dtype=np.float64)
N_CANDIDATES = CANDIDATES.shape[0]