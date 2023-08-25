from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from ..utils import (
    from_tf,
    get_gpu_env,
)

logger = getLogger(__name__)


def prepare_matrix(data, y, trans, env, factor, prefetch):
    nt = data.nt
    ns = data.ns
    nk, ny = y.shape

    env = get_gpu_env(env)
    nd = env.num_devices
    sharding = env.sharding((nd, 1))
    batch = env.batch(float(factor) * nk * ns, nt)
    logger.info("prepare: nt=%d nk=%d ns=%d batch=%d", nt, nk, ns, batch)

    nkmod = nd * ((nk + nd - 1) // nd)
    y = np.pad(y, ((0, nkmod - nk), (0, 0)))
    yval = jax.device_put(y, sharding)

    yavg = yval.mean(axis=1)
    ydif = yval - yavg[:, jnp.newaxis]

    ycov = ydif @ ydif.T
    yout = jnp.outer(yavg, yavg)

    logger.info("%s: %s %s %d", "pbar", "start", "prepare", nt)
    ydot = matmul_batch(ydif, data, trans, sharding, batch, prefetch)
    logger.info("%s: %s", "pbar", "close")

    return ycov[:nk, :nk], yout[:nk, :nk], ydot[:nk]


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
    penalty = ny * bx * (yout * xcov).sum() + nx * by * (ycov * xout).sum()
    var = diff + penalty
    return (nm / 2) * (jnp.log(var) - jnp.log(nm)), jnp.sqrt(var / nm)
