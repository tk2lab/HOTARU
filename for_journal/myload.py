import numpy as np
import pandas as pd
import h5py


def load(path, stage, df_only=False):
    if stage == 0:
        apath = f"{path}/../../../.."
        upath = f"{path}/../.."
    else:
        apath = path
        upath = path
    if df_only:
        return pd.read_csv(f"{upath}/{stage:03d}stats.csv", index_col=0)
    with h5py.File(f"{apath}/{stage:03d}footprints.h5") as h5:
        a = h5["data"][...]
    u = np.load(f"{upath}/{stage:03d}spike.npy")
    df = pd.read_csv(f"{upath}/{stage:03d}stats.csv", index_col=0)
    return a, u, df
