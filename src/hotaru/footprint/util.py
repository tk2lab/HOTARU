import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import label, binary_closing
from scipy.ndimage import generate_binary_structure, binary_dilation
from matplotlib import cm


_greens = cm.get_cmap('Greens')


def get_normalized_val(img, pos):
    val = tf.gather_nd(img, pos)
    min_val = K.min(val)
    max_val = K.max(val)
    return (val - min_val) / (max_val - min_val)


def get_magnitude(img, pos):
    val = tf.gather_nd(img, pos)
    return K.max(val) - K.min(val)


class ToDense(object):

    def __init__(self, mask):
        mask = K.constant(mask, tf.bool)
        nx = tf.cast(tf.math.count_nonzero(mask), tf.int32)
        ids = tf.cast(tf.where(mask), tf.int32)
        rmap = tf.scatter_nd(ids, tf.range(nx) + 1, tf.shape(mask)) - 1
        self.rmap = rmap
        self.nx = nx

    def __call__(self, pos, val):
        pos = tf.gather_nd(self.rmap, pos)
        out = tf.scatter_nd(pos[:, None], val, (self.nx,))
        return out


def calc_sim_area(segment, mask, gauss, thr_sim_area):
    nk = segment.shape[0]
    h, w = mask.shape
    seg = np.zeros((nk, h, w), np.float32)
    seg[:, mask] = segment
    if gauss > 0.0:
        seg = gaussian(seg, gauss)
    seg -= seg.min(axis=(1, 2), keepdims=True)
    mag = seg.max(axis=(1, 2))
    cond = mag > 0.0
    seg[cond] /= mag[cond, None, None]
    seg[~cond] = 1.0
    seg = seg > thr_sim_area
    cor = np.zeros((nk,))
    for j in np.arange(nk)[::-1]:
        cj = 0.0
        for i in np.arange(j):
            ni = np.count_nonzero(seg[i])
            nij = np.count_nonzero(seg[i] & seg[j])
            cij = nij / ni
            if cij > cj:
                cj = cij
        cor[j] = cj
    return cor


def calc_sim_cos(segment):
    nk = segment.shape[0]
    scale = np.sqrt((segment ** 2).sum(axis=1))
    seg = segment / scale[:, None]
    cor = np.zeros((nk,))
    for j in np.arange(nk)[::-1]:
        cj = 0.0
        for i in np.arange(j)[::-1]:
            cij = np.dot(seg[i], seg[j])
            if cij > cj:
                cj = cij
        cor[j] = cj
    return cor
