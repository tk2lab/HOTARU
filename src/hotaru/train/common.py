from logging import getLogger
from functools import partial

import jax
import jax.numpy as jnp
import tensorflow as tf

from ..utils import from_tf

logger = getLogger(__name__)


def matmul_batch(x, y, trans, sharding, batch, prefetch):
    if trans:

        @jax.jit
        def calc(out, x, t, yt):
            return out.at[:, t].set(x @ yt.T)

        nk, ns = x.shape
        nt = y.nt
        ntmod = batch * ((nt + batch - 1) // batch)
        out = jax.device_put(jnp.empty((nk, ntmod)), sharding)
    else:

        @jax.jit
        def calc(out, x, t, yt):
            return out + x[:, t] @ yt

        nk, nt = x.shape
        ns = y.ns
        out = jax.device_put(jnp.zeros((nk, ns)), sharding)

    dataset = tf.data.Dataset.from_generator(
        lambda: enumerate(y.data(mask_type=True)),
        output_signature=(
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((ns,), tf.float32),
        ),
    )
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    end = 0
    for data in dataset:
        t, yt = tuple(jax.device_put(from_tf(d), sharding.replicate()) for d in data)
        out = calc(out, x, t, yt)
        start, end = end, end + batch
        n = batch if end < nt else nt - start
        logger.info("%s: %s %d", "pbar", "update", n)

    if trans:
        out = out[:, :nt]
    return out


def loss_fn(x, ycov, yout, ydot, nx, ny, bx, by):
    nn = nx * ny
    nm = nn + nx + ny
    xsum = x.sum(axis=1)
    xavg = xsum / nx
    xdif = x - xavg[:, jnp.newaxis]
    xcov = xdif @ xdif.T
    xout = jnp.outer(xavg, xavg)
    diff = (ycov * xcov).sum() - 2 * (ydot * xdif).sum() + nn
    penalty = bx * (yout * xcov).sum() + by * (ycov * xout).sum()
    var = diff + penalty
    return (nm / 2) * (jnp.log(var) - jnp.log(nm))
