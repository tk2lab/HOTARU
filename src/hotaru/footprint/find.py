from collections import namedtuple
from logging import getLogger

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from ..filter import (
    gaussian,
    gaussian_laplace,
    mapped_imgs,
    max_pool,
)
from ..utils import get_gpu_env
from .radius import get_radius

logger = getLogger(__name__)

PeakVal = namedtuple("PeakVal", ["radius", "t", "r", "v"])


def find_peaks(data, radius, env=None, factor=100):
    def calc(ts, imgs):
        t, r, g = _find_peaks(imgs, mask, radius)
        t = ts[t]
        return t, r, g

    nt, h, w = data.imgs.shape
    mask = data.mask

    radius = get_radius(radius)
    batch = get_gpu_env(env).batch(float(factor) * h * w * len(radius), nt)

    logger.info("find: %d %d %d %f %f", nt, h, w, radius[0], radius[-1])
    dataset = tf.data.Dataset.from_generator(
        lambda: enumerate(data.data(mask_type=False)),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((h, w), tf.float32),
        ),
    )
    types = [("argmax", -1), ("argmax", -1), "max"]
    init = [
        jnp.full((h, w), -1, jnp.int32),
        jnp.full((h, w), -1, jnp.int32),
        jnp.full((h, w), -jnp.inf),
    ]
    logger.info("%s: %s %s %d", "pbar", "start", "find", nt)
    jnp_out = mapped_imgs(dataset, nt, calc, types, init, batch)
    logger.info("%s: %s", "pbar", "close")
    for i, r in enumerate(radius):
        logger.info("%f %d", r, (jnp_out[1] == i).sum())
    return PeakVal(np.array(radius, np.float32), *map(np.array, jnp_out))


def simple_peaks(img, gauss, maxpool, pbar=None):
    if pbar is not None:
        pbar.set_count(3)
    g = gaussian(img[None, ...], gauss)[0]
    if pbar is not None:
        pbar.update(1)
    m = max_pool(g, (maxpool, maxpool), (1, 1), "same")
    if pbar is not None:
        pbar.update(1)
    y, x = jnp.where(g == m)
    if pbar is not None:
        pbar.update(1)
    return np.array(y), np.array(x)


def simple_find(imgs, mask, radius):
    t, r, v = (np.array(o) for o in _find_peaks(imgs, mask, get_radius(**radius)))
    idx = jnp.where(np.isfinite(v))
    return t[idx], idx[:, 0], idx[:, 1], r[idx], v[idx]


def _find_peaks(imgs, mask, radius):
    nt, h, w = imgs.shape
    gl = gaussian_laplace(imgs, radius, 1)
    gl_max = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
    gl_peak = gl == gl_max
    if mask is not None:
        gl_peak &= mask
    gl_peak_val = jnp.where(gl_peak, gl, -jnp.inf)
    idx = jnp.argmax(gl_peak_val.reshape(-1, h, w), axis=0)
    t, r = jnp.divmod(idx, len(radius))
    return t, r, gl_peak_val.max(axis=(0, 1))
