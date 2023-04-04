import pathlib

import numpy as np
import tensorflow as tf
import tifffile


def get_mask(mask, imgs, reverse=False):
    nt, h, w = imgs.shape
    path = pathlib.Path(mask)
    if path.suffix == ".pad":
        pad = int(path.stem)
        mask = np.zeros([h, w], bool)
        mask[pad : h - pad, pad : w - pad] = True
    else:
        with tf.io.gfile.GFile(path, "rb") as fp:
            if path.suffix == ".tif":
                mask = tifffile.imread(fp)
            elif path.suffix == ".npy":
                mask = np.load(fp)
            else:
                raise RuntimeErorr("bad file type: {maskfile}")
        mask = mask > 0
        if reverse:
            mask = ~mask
    return mask


def get_mask_range(mask):
    my = np.where(np.any(mask, axis=1))[0]
    mx = np.where(np.any(mask, axis=0))[0]
    return my[0], my[-1] + 1, mx[0], mx[-1] + 1
