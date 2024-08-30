import numpy as np


def normalize(x, epsilon=1e-2):
    scale = np.ptp(x)
    if scale > epsilon:
        return (x - x.min()) / scale
    else:
        return np.zeros_like(x)
