from functools import partial
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from scipy.ndimage import grey_closing

from ..filter.laplace import gaussian_laplace_single
from ..utils import (
    from_tf,
    get_gpu_env,
)
from .segment import get_segment_mask

logger = getLogger(__name__)


def make_footprints(data, peaks, env=None, factor=1, prefetch=1):
    nk = peaks.shape[0]
    h, w = data.shape
    ts, ys, xs, rs = (np.array(v) for v in (peaks.t, peaks.y, peaks.x, peaks.radius))

    env = get_gpu_env(env)
    #nd = env.num_devices
    #sharding = env.sharding((nd, 1))
    batch = env.batch(float(factor) * h * w, nk)
    logger.info("make: %s batch=%d", peaks.shape[0], batch)

    logger.info("%s: %s %s %d", "pbar", "start", "make", ts.size)
    out = np.empty((nk, h, w))
    for r in np.unique(rs):
        index = np.where(rs == r)[0]
        dataset = tf.data.Dataset.from_generator(
            lambda: zip(index, data.data(ts[index]), ys[index], xs[index]),
            output_signature=(
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((h, w), tf.float32),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
            ),
        )
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(prefetch)
        for d in dataset:
            d = (from_tf(v) for v in d)
            #idx, imgs, y, x = (jax.device_put(v, sharding) for v in d)
            idx, imgs, y, x = (jnp.array(v) for v in d)

            count = idx.size
            diff = batch - count
            if diff > 0:
                pad = (0, diff), (0, 0), (0, 0)
                imgs = jnp.pad(imgs, pad, constant_values=jnp.nan)
                y = jnp.pad(y, ((0, diff)), constant_values=-1)
                x = jnp.pad(x, ((0, diff)), constant_values=-1)
            out[idx] = np.array(_make_segments_simple(imgs, y, x, r)[:count])
            logger.info("%s: %s %d", "pbar", "update", count)
    out = grey_closing(out, (1, 10, 10))
    logger.info("%s: %s", "pbar", "close")
    return out


def make_segments_simple(imgs, y, x, r):
    return np.array(_make_segments_simple(imgs, y, x, r))


@partial(jax.jit, static_argnames=("r",))
def _make_segments_simple(imgs, y, x, r):
    g = gaussian_laplace_single(imgs, r)
    seg = jax.vmap(get_segment_mask)(g, y, x)
    val = jnp.where(seg, g, jnp.nan)
    dmin = jnp.nanmin(val, axis=(1, 2), keepdims=True)
    dmax = jnp.nanmax(val, axis=(1, 2), keepdims=True)
    out = jnp.where(seg, (val - dmin) / (dmax - dmin), 0)
    return out
