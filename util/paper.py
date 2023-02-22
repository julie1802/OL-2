import numpy as np


def derived_calculation(delta_p: float, p_min: float, m: int) -> float:
    q = 2.0 * (m - 1) / (1 - p_min)
    return min(2 / (delta_p * delta_p) * np.log(q), 1 + (1 / (delta_p * delta_p) - 1) * q)
