import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import label, binary_closing
from scipy.ndimage import generate_binary_structure, binary_dilation
from matplotlib import cm


_jet = cm.get_cmap('jet')
_reds = cm.get_cmap('Reds')
_greens = cm.get_cmap('Greens')


def normalized_and_sort(val):
    mag = val.max(axis=1)
    idx = np.argsort(mag)[::-1]
    val = val / mag[:, None]
    val = val[idx]
    return val, mag


def summary_stat(val, stage, step=0):
    val, mag = normalized_and_sort(val)
    spc = val.mean(axis=1)
    tf.summary.histogram(f'val_max/{stage}', mag, step=step)
    tf.summary.histogram(f'val_avg/{stage}', spc, step=step)


def summary_spike(val, stage, step=0):
    val, mag = normalized_and_sort(val)
    tf.summary.image(f'spike/{stage}', _reds(val)[None, ...], step=step)


def summary_footprint_sample(val, mask, stage, step=0):
    h, w = mask.shape
    imgs = np.zeros((4, h, w))
    imgs[:, mask] = val[[0, 1, -2, -1]]
    tf.summary.image(
        f'footprint-sample/{stage}',
        _greens(imgs), max_outputs=4, step=step,
    )


def summary_footprint_max(val, mask, stage, step=0):
    val_max = val.max(axis=0)
    val_max /= val.max()
    h, w = mask.shape
    imgs_max = np.zeros((1, h, w))
    imgs_max[0, mask] = val_max
    tf.summary.image(f'max/{stage}', _greens(imgs_max), step=step)


def summary_segment(val, mask, flag, gauss, thr_out, stage):
    struct = generate_binary_structure(2, 2)
    h, w = mask.shape
    out = 0
    b0 = False
    b1 = False
    for f, v in zip(flag[::-1], val[::-1]):
        img = np.zeros((h, w))
        img[mask] = v
        g = gaussian(img, gauss)
        g /= g.max()
        peak = np.argmax(g)
        lbl, n = label(g > thr_out)
        size = 0
        for i in range(1, n+1):
            tmp = binary_closing(lbl == i)
            ts = np.count_nonzero(tmp)
            if ts > size:
                size = ts
                obj = tmp
        out += obj
        bound = binary_dilation(obj, struct) ^ obj
        if f:
            b0 |= bound
        else:
            b1 |= bound
    out = out / out.max()
    out = _greens(out)
    out[b0] = (0, 0, 0, 1)
    out[b1] = (1, 0, 0, 1)
    tf.summary.image(f'segment/{stage}', out[None, ...], step=0)
