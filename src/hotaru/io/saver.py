from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd


def save(path, obj):
    if isinstance(path, (list, tuple)):
        for p, o in zip(path, obj):
            save(p, o)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        match path.suffix:
            case ".npz":
                np.savez(path, **obj._asdict())
            case ".npy":
                np.save(path, obj)
            case ".csv":
                obj.to_csv(path)


def try_load(path):
    if isinstance(path, (list, tuple)):
        return [try_load(p) for p in path]
    else:
        path = Path(path)
        try:
            match path.suffix:
                case ".npz":
                    with np.load(path) as npz:
                        return namedtuple("LoadedData", npz.files)(**dict(npz.items()))
                case ".npy":
                    return np.load(path)
                case ".csv":
                    return pd.read_csv(path, index_col=0)
                case _:
                    return
        except Exception:
            return
