from collections import namedtuple
from pathlib import Path
from logging import getLogger

import numpy as np
import pandas as pd
import h5py

logger = getLogger(__name__)


def h5_save(path, obj):
    layout = h5py.VirtualLayout(shape=obj.shape, dtype=obj.dtype)
    with h5py.File(path, "w") as h5:
        for i, val in enumerate(obj):
            my = np.where(np.any(val > 0, axis=1))[0]
            mx = np.where(np.any(val > 0, axis=0))[0]
            if my.size > 0:
                y0, y1, x0, x1 = my[0], my[-1] + 1, mx[0], mx[-1] + 1
                val = val[y0:y1, x0:x1]
                data = h5.create_dataset(f"{i}", data=val)
                layout[i, y0:y1, x0:x1] = h5py.VirtualSource(data)
        h5.create_virtual_dataset("data", layout, fillvalue=0)


def h5_load(path):
    with h5py.File(path, "r") as h5:
        return h5["data"][...]


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
            case ".h5":
                h5_save(path, obj)
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
        logger.debug("%s", path)
        try:
            match path.suffix:
                case ".npz":
                    with np.load(path) as npz:
                        return namedtuple("LoadedData", npz.files)(**dict(npz.items()))
                case ".npy":
                    return np.load(path)
                case ".csv":
                    return pd.read_csv(path, index_col=0)
                case ".h5":
                    return h5_load(path)
                case ".flag":
                    return path.read_text()
                case _:
                    return
        except Exception as e:
            logger.debug("%s", e)
            return
