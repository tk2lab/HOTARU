import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
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
    cor = val @ val.T
    scale = np.sqrt(np.diag(cor))
    cor /= scale * scale[:, None]
    tf.summary.image(f'cor/{stage:03d}', _jet(cor)[None, ...], step=step)
    tf.summary.histogram(f'peak/{stage:03d}', mag, step=step)
    tf.summary.histogram(f'sparseness/{stage:03d}', spc, step=step)
    return val


def summary_spike(val, stage, step=0):
    tf.summary.image(f'spike/{stage:03d}', _reds(val)[None, ...], step=step)


def summary_footprint_sample(val, mask, stage, step=0):
    h, w = mask.shape
    imgs = np.zeros((4, h, w))
    imgs[:, mask] = val[[0, 1, -2, -1]]
    tf.summary.image(
        f'footprint-sample/{stage:03d}',
        _greens(imgs), max_outputs=4, step=step,
    )


def summary_footprint_max(val, mask, stage, step=0):
    h, w = mask.shape
    imgs_max = np.zeros((1, h, w))
    imgs_max[0, mask] = val.max(axis=0)
    tf.summary.image(f'max/{stage:03d}', _greens(imgs_max), step=step)


def summary_footprint(val, mask, stage):
    h, w = mask.shape
    img = np.zeros((h, w))
    for i, v in enumerate(val[::-1]):
        img[:] = 0.0
        img[mask] = v
        tf.summary.image(
            f'footprint/{stage:03d}', _greens(img)[None, ...], step=i+1,
        )
