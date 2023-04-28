from collections import namedtuple

import numpy as np
import pandas as pd


def save(path, obj):
    if path.suffix == ".npz":
        np.savez(path, **obj._asdict())
    elif path.suffix == ".npy":
        np.save(path, obj)
    elif path.suffix == ".cvs":
        obj.to_csv(path)


def load(path):
    if path.suffix == ".npz":
        with np.load(path) as npz:
            return namedtuple("LoadedData", npz.files)(**dict(npz.items()))
    elif path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
