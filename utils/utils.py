import numpy as np


def normalize(x):
    scale = np.ptp(x)
    if scale > 0:
        return (x - x.min()) / scale
    else:
        return np.zeros_like(x)