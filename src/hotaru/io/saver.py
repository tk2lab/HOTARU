from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd


def save(path, obj):
    if isinstance(path, (list, tuple)):
        if len(path) == 1:
            save(path[0], obj[0])
        else:
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
            case ".flag":
                path.write_text(obj)
            case _:
                raise ValueError()


def try_load(path):
    if isinstance(path, (list, tuple)):
        if len(path) == 1:
            return try_load(path[0])
        else:
            return [try_load(p) for p in path]
    else:
        path = Path(path)
        print(path)
        try:
            match path.suffix:
                case ".npz":
                    with np.load(path) as npz:
                        return namedtuple("LoadedData", npz.files)(**dict(npz.items()))
                case ".npy":
                    return np.load(path)
                case ".csv":
                    return pd.read_csv(path, index_col=0)
                case ".flag":
                    return path.read_text()
                case _:
                    return
        except Exception as e:
            print("fail")
            print(e)
            return
