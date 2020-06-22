import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from skimage.filters import gaussian
from skimage.morphology import label
from skimage.segmentation import find_boundaries
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
    #cor = val @ val.T
    #scale = np.sqrt(np.diag(cor))
    #cor /= scale * scale[:, None]
    #tf.summary.image(f'cor/{stage}', _jet(cor)[None, ...], step=step)
    tf.summary.histogram(f'max_val/{stage}', mag, step=step)
    tf.summary.histogram(f'avg_val/{stage}', spc, step=step)
    return val


def summary_spike(val, stage, step=0):
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
    h, w = mask.shape
    imgs_max = np.zeros((1, h, w))
    imgs_max[0, mask] = val.max(axis=0)
    tf.summary.image(f'max/{stage}', _greens(imgs_max), step=step)


def summary_segment(val, mask, flag, stage):
    h, w = mask.shape
    out = 0
    b0 = False
    b1 = False
    for f, v in zip(flag[::-1], val[::-1]):
        img = np.zeros((h, w))
        img[mask] = v
        g = gaussian(img, 2)
        g /= g.max()
        peak = np.argmax(g)
        y, x = peak // w, peak % w
        lbl = label(g > 0.8, connectivity=1)
        lbl = lbl == lbl[y, x]
        out += lbl
        if f:
            b0 |= find_boundaries(lbl)
        else:
            b1 |= find_boundaries(lbl)
    out = out / out.max()
    out = _greens(out)
    out[b0] = (0, 0, 0, 1)
    out[b1] = (1, 0, 0, 1)
    tf.summary.image(f'segment/{stage}', out[None, ...], step=0)