import os

import tensorflow as tf
import numpy as np
import tifffile


def get_mask(masktype, h, w, job_dir='.'):
    if masktype[-4:] == '.pad':
        pad = int(masktype.split('.')[-2])
        mask = np.zeros([h, w], bool)
        mask[pad:h-pad,pad:w-pad] = True
    else:
        with tf.io.gfile.GFile(os.path.join(job_dir, masktype), 'rb') as fp:
            if masktype[-4:] == '.tif':
                mask = tifffile.imread(fp)
            elif masktype[-4:] == '.npy':
                mask = np.load(fp)
        mask = mask > 0
    return mask


def get_mask_range(mask):
    my = np.where(np.any(mask, axis=1))[0]
    mx = np.where(np.any(mask, axis=0))[0]
    return my[0], my[-1] + 1, mx[0], mx[-1] + 1


def make_maskfile(mask_file, mask):
    with tf.io.gfile.GFile(mask_file, 'bw') as fp:
        np.save(fp, mask)


def load_maskfile(mask_file):
    with tf.io.gfile.GFile(mask_file, 'br') as fp:
        return np.load(fp)
