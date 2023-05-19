from collections import namedtuple

import numpy as np
import pandas as pd


def save(path, obj):
    if isinstance(path, (list, tuple)):
        for p, o in zip(path, obj):
            save(p, o)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            np.savez(path, **obj._asdict())
        elif path.suffix == ".npy":
            np.save(path, obj)
        elif path.suffix == ".csv":
            obj.to_csv(path)


def load(path):
    if isinstance(path, (list, tuple)):
        return tuple(load(p) for p in path)
    else:
        suffix = path.suffix
        if suffix == ".npz":
            with np.load(path) as npz:
                return namedtuple("LoadedData", npz.files)(**dict(npz.items()))
        elif suffix == ".npy":
            return np.load(path)
        elif suffix == ".csv":
            return pd.read_csv(path, index_col=0)
