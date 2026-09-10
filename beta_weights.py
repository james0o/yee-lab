import numpy as np
import scipy.stats as stats
from scipy.optimize import root
import os
from const import PIXELS, DISTANCE, N_PART

def create_voters(n_part):
    voters = np.arange(1 / n_part / 2, 1, 1 / n_part)
    return voters

def cdf_interval(voters, a, b):
    lower = stats.beta.cdf(x=voters - 1 / len(voters) / 2, a=a, b=b)
    upper = stats.beta.cdf(x=voters + 1 / len(voters) / 2, a=a, b=b)
    return upper - lower

def cdf_with_points(n_part, beta_params, w1):
    a, b = beta_params
    voters = create_voters(n_part)
    dist = np.concatenate([[0], cdf_interval(voters, a, b), [w1]])
    rescale = dist/dist.sum()
    return rescale

def create_medians(n):
    return np.arange(0.5+1/n/2, 1, 1/n)

def activation_function(median, soft_threshold = 5, sharpness = 5):
    m_trans = (median - 0.5) / 0.5
    return 0.5 / (1 + np.exp(-sharpness * (m_trans - soft_threshold)))

def create_weights_with_mass_points(median, w1, D_target, X0):
    assert median > 0.5, "Median must be greater than 0.5"

    w0 = 0
    wd = 1 - w1
   
    def system_of_equations(x):
        a, b = x[0], x[1]

        if a < 0 or b < 0:
            return [1e10, 1e10]

        cdf_ab = stats.beta.cdf(x=median, a=a, b=b)
        cdf_a1b = stats.beta.cdf(x=median, a=a + 1, b=b)
       
        eq1 = w0 + wd * cdf_ab - 0.5
        ex = a / (a + b)
        eq2 = (w0 * median + w1 * (1 - median) +
               wd * (median * (2*cdf_ab-1) + ex * (1 - 2 * cdf_a1b)) - D_target)
           
        return [eq1, eq2]

    sol = root(system_of_equations, X0, tol=1e-14, method='lm')
    a_sol, b_sol = sol.x[0], sol.x[1]
    return (a_sol, b_sol)

def generate_weights(pixels: int, n_part: int) -> None:
    medians = create_medians(pixels)
    N_medians = len(medians)
    X0 = [1.0, 1.0]
    weights = np.empty((N_medians, n_part + 2), dtype=np.float64)
    params = np.empty((N_medians, 2), dtype=np.float64)
    for i, m in enumerate(medians):
        w1 = activation_function(m)
        (a, b) = create_weights_with_mass_points(m, w1, DISTANCE, X0)
        X0 = [a, b]
        params[i] = (a, b)
        weights[i] = cdf_with_points(n_part, (a, b), w1)
   
    params = np.concatenate((np.flip(params), params), axis=0)
    weights = np.concatenate((np.flip(weights), weights), axis=0)

    if not os.path.exists('params'):
        os.makedirs('params')
    if not os.path.exists('weights'):
        os.makedirs('weights')

    np.savez(f'params/P{pixels}_N{n_part}', params=params)
    np.savez(f'weights/P{pixels}_N{n_part}', weights=weights)

def load_weights(n_part: int, pixels: int = PIXELS) -> np.ndarray:
    if not os.path.exists(f'weights/P{pixels}_N{n_part}.npz'):
        generate_weights(pixels, n_part)
    return np.load(f'weights/P{pixels}_N{n_part}.npz')["weights"]

if __name__ == "__main__":
    generate_weights(pixels=PIXELS, n_part=N_PART)

