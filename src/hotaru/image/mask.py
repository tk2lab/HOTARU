import os

import tensorflow as tf
import numpy as np
import tifffile

from ..util.gs import ensure_local_file


def get_mask(masktype, h, w, job_dir='.'):
    if masktype[-4:] == '.pad':
        pad = int(masktype.split('.')[-2])
        mask = np.zeros([h, w], bool)
        mask[pad:h-pad, pad:w-pad] = True
    else:
        if masktype[:2] == 'r:':
            maskfile = ensure_local_file(os.path.join(job_dir, masktype[2:]))
        else:
            maskfile = ensure_local_file(os.path.join(job_dir, masktype))
        if masktype[-4:] == '.tif':
            mask = tifffile.imread(maskfile)
        elif masktype[-4:] == '.npy':
            mask = np.load(maskfile)
        else:
            raise RuntimeErorr('bad file type: {maskfile}')
        mask = mask > 0
        if masktype[:2] == 'r:':
            mask = ~mask
    return mask


def get_mask_range(mask):
    my = np.where(np.any(mask, axis=1))[0]
    mx = np.where(np.any(mask, axis=0))[0]
    return my[0], my[-1] + 1, mx[0], mx[-1] + 1
