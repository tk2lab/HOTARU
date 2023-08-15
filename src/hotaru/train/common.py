from logging import getLogger
from functools import partial

import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.dlpack import from_dlpack

logger = getLogger(__name__)


def from_tf(x_tf):
    x_dl = tf.experimental.dlpack.to_dlpack(x_tf)
    return from_dlpack(x_dl)


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


def loss_fn(x, nn, nm, a, b, c):
    nk, nx = x.shape
    xdot = (a * x).sum()
    xcov = x @ x.T
    xsum = x.sum(axis=1)
    xmean = xsum / nx
    var = (nn + (b * xcov).sum() + (xsum @ c @ xmean) - 2 * xdot) / nm
    return jnp.log(var) / 2
