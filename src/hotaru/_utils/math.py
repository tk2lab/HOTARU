import numpy as np


def calc_sn(spikes):
    return np.array([si.max() / (1.4826 * np.median(si[si > 0])) for si in spikes])


def robust_zscore(x):
    medx = np.median(x)
    diff = x - medx
    stdx = 1.4826 * np.median(np.abs(x - medx))
    return diff / stdx


def calc_mah(x, y):
    dx = np.stack([x - x.mean(), y - y.mean()], axis=1)
    tau = np.linalg.inv((dx.T @ dx) / dx.shape[0])
    return np.sqrt(np.sum((dx @ tau) * dx, axis=1))
