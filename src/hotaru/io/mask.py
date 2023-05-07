import pathlib

import numpy as np
import tifffile


def load_mask(mask_path, imgs, reverse=False):
    path = pathlib.Path(mask_path)
    nt, h, w = imgs.shape
    if path.suffix == ".pad":
        pad = int(path.stem)
        mask = np.zeros([h, w], bool)
        mask[pad : h - pad, pad : w - pad] = True
    else:
        if path.suffix == ".tif":
            mask = tifffile.imread(path)
        elif path.suffix == ".npy":
            mask = np.load(path)
        else:
            raise RuntimeErorr("bad file type: {maskfile}")
        mask = mask > 0
        if reverse:
            mask = ~mask
    return mask


def mask_range(mask):
    my = np.where(np.any(mask, axis=1))[0]
    mx = np.where(np.any(mask, axis=0))[0]
    return mx[0], my[0], mx[-1] - mx[0] + 1, my[-1] - mx[0] + 1