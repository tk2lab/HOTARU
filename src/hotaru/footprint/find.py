from collections import namedtuple
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from ..filter import (
    gaussian,
    gaussian_laplace,
    max_pool,
)
from ..utils import (
    from_tf,
    get_gpu_env,
)
from .radius import get_radius

logger = getLogger(__name__)

PeakVal = namedtuple("PeakVal", ["radius", "t", "r", "v"])


def find_peaks(data, radius, env=None, factor=1, prefetch=1):
    @jax.jit
    def update(ts, rs, gs, index, imgs):
        t, r, g = _find_peaks(imgs, mask, radius)
        cond = g < gs
        ts = jnp.where(cond, ts, index[t])
        rs = jnp.where(cond, rs, r)
        gs = jnp.where(cond, gs, g)
        return ts, rs, gs

    nt, h, w = data.imgs.shape

    radius = get_radius(radius)
    env = get_gpu_env(env)
    nd = env.num_devices
    batch = env.batch(float(factor) * h * w * len(radius), nt)
    sharding = env.sharding((nd, 1))

    logger.info(
        "find: nt=%d h=%d w=%d rmin=%f rmax=%f batch=%d",
        nt,
        h,
        w,
        radius[0],
        radius[-1],
        batch,
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: zip(range(nt), data.data(mask_type=False)),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((h, w), tf.float32),
        ),
    )
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    mask = None if data.mask is None else jnp.array(data.mask, bool)
    ts = jnp.full((h, w), -1, jnp.int32)
    rs = jnp.full((h, w), -1, jnp.int32)
    gs = jnp.full((h, w), -jnp.inf)

    logger.info("%s: %s %s %d", "pbar", "start", "find", nt)
    for d in dataset:
        d = (from_tf(v) for v in d)
        index, imgs = (jax.device_put(v, sharding) for v in d)

        count = index.size
        diff = batch - count
        if diff > 0:
            index = jnp.pad(index, ((0, diff)), constant_values=-1)
            imgs = jnp.pad(imgs, ((0, diff), (0, 0), (0, 0)), constant_values=jnp.nan)

        ts, rs, gs = update(ts, rs, gs, index, imgs)
        logger.info("%s: %s %d", "pbar", "update", count)
    logger.info("%s: %s", "pbar", "close")

    for i, r in enumerate(radius):
        logger.info("radius=%f num=%d", r, (rs == i).sum())
    return PeakVal(np.array(radius, np.float32), *map(np.array, (ts, rs, gs)))


def simple_peaks(img, gauss, maxpool):
    g = gaussian(img[None, ...], gauss)[0]
    m = max_pool(g, (maxpool, maxpool), (1, 1), "same")
    y, x = jnp.where(g == m)
    return np.array(y), np.array(x)


def simple_find(imgs, mask, radius):
    t, r, v = (np.array(o) for o in _find_peaks(imgs, mask, get_radius(**radius)))
    idx = jnp.where(np.isfinite(v))
    return t[idx], idx[:, 0], idx[:, 1], r[idx], v[idx]


def _find_peaks(imgs, mask, radius):
    nt, h, w = imgs.shape
    nr = len(radius)
    gl = gaussian_laplace(imgs, radius, axis=1)
    gl_max = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
    gl_peak = gl == gl_max
    if mask is not None:
        gl_peak &= mask
    gl = jnp.where(gl_peak, gl, -jnp.inf)
    gl_reshape = gl.reshape(nt * nr, h, w)
    idx = jnp.argmax(gl_reshape, axis=0)
    gl_max = jnp.take_along_axis(gl_reshape, idx[jnp.newaxis, ...], axis=0)[0]
    t, r = jnp.divmod(idx, nr)
    return t, r, gl_max
