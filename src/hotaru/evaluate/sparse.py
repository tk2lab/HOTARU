import numpy as np


def calc_sparseness(x):
    n, b = np.histogram(x[x>0], bins=50)
    return b[np.argmax(n)] / x.max()
