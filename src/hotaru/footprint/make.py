from functools import partial
from logging import getLogger

import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np

from ..utils import get_gpu_env
from ..filter.laplace import gaussian_laplace_single
from ..filter.map import mapped_imgs
from .segment import get_segment_mask

logger = getLogger(__name__)


def make_footprints(data, peaks, env=None, factor=10):
    @partial(jax.jit, static_argnames="r")
    def _calc(imgs, y, x, index, r):
        return make_segments_simple(imgs, y, x, r), index

    logger.info("make: %s", peaks.shape[0])
    h, w = data.shape
    ts, ys, xs, rs = (np.array(v) for v in (peaks.t, peaks.y, peaks.x, peaks.radius))

    logger.info("%s: %s %s %d", "pbar", "start", "make", ts.size)
    out = jnp.empty((rs.size + 1, h, w))
    for r in np.unique(rs):
        index = np.where(rs == r)[0]
        #logger.info("make: %f %d", r, index.size)
        batch = get_gpu_env(env).batch(float(factor) * h * w, index.size)
        dataset = tf.data.Dataset.from_generator(
            lambda: zip(data.data(ts[index]), ys[index], xs[index], index),
            output_signature=(
                tf.TensorSpec((h, w), tf.float32),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
            ),
        )
        types = [("stack", -1)]
        calc = partial(_calc, r=r)
        out, = mapped_imgs(dataset, index.size, calc, types, [out], batch)
    out = out[:-1]
    logger.info("%s: %s", "pbar", "close")
    return np.array(out)


def make_segments_simple(imgs, y, x, r):
    g = gaussian_laplace_single(imgs, r)
    seg = jax.vmap(get_segment_mask)(g, y, x)
    val = jnp.where(seg, g, jnp.nan)
    dmin = jnp.nanmin(val, axis=(1, 2), keepdims=True)
    dmax = jnp.nanmax(val, axis=(1, 2), keepdims=True)
    out = jnp.where(seg, (val - dmin) / (dmax - dmin), 0)
    return out
