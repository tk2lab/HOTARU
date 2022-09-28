import numpy as np


def calc_area(x, threshold):
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    x = (x - xmin) / (xmax - xmin)
    return np.count_nonzero(x > threshold, axis=1)


def calc_overwrap(x):
    x = x.astype(np.float32)
    c = (x @ x.T) / x.sum(axis=1)
    np.fill_diagonal(c, 0.0)
    return c.max(axis=1)


def calc_denseness(x):
    def sp(x):
        n, b = np.histogram(x[x>0], bins=np.linspace(0, 1, 51))
        return b[np.argmax(n)]
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    x = (x - xmin) / (xmax - xmin)
    return np.array([sp(xi) for xi in x])
