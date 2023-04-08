from collections import namedtuple

import numpy as np


def save(path, obj):
    np.savez(path, **obj._asdict())


def load(path):
    with np.load(path) as npz:
        return napedtuple("LoadedData", npz.files)(**dict(npz.items()))
