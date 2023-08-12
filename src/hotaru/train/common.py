from logging import getLogger

import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np

from ..filter.map import mapped_imgs
from ..utils import get_gpu_env

logger = getLogger(__name__)


def mask_fn(x, num_devices):
    n, *shape = x.shape
    d = n % num_devices
    xval = jnp.pad(x, [[0, d]] + [[0, 0]] * len(shape))
    return xval.reshape(num_devices, (n + d) // num_devices, *shape)


def clip(x, start, nd, batch):
    nk, nz = x.shape
    if start + nd * batch >= nz:
        x = jnp.pad(x, ((0, 0), (0, nz - (start + nd * batch))))
    return x.T[start:start + nd * batch].reshape((nd, batch, nk))


def calc_cov_out(x, env, factor):
    @jax.pmap
    def _calc(x):
        return x.sum(axis=0), x.T @ x

    nk, nz = x.shape
    nd, batch = get_gpu_env(env).batch(float(factor) * nk * nk, nz)
    #logger.info("calc_err: %f %d %d %d %d", factor, nk, nz, nd, batch)
    xsum = jnp.zeros((nk,))
    xcov = jnp.zeros((nk, nk))
    for i in range(0, nz, nd * batch):
        xsumi, xcovi = _calc(clip(x, i, nd, batch))
        xsum += xsumi.sum(axis=0)
        xcov += xcovi.sum(axis=0)
    return xcov, jnp.outer(xsum, xsum) / nz


def calc_err(nn, a, b, c, x, env, factor):
    @jax.pmap
    def _calc(a, x):
        return (a * x).sum(), x.sum(axis=0), x.T @ x

    nk, nz = x.shape
    nd, batch = get_gpu_env(env).batch(float(factor) * nk * nk, nz)
    #logger.info("calc_err: %f %d %d %d %d", factor, nk, nz, nd, batch)
    xdot = jnp.zeros(())
    xsum = jnp.zeros((nk,))
    xcov = jnp.zeros((nk, nk))
    for i in range(0, nz, nd * batch):
        xdoti, xsumi, xcovi = _calc(clip(a, i, nd, batch), clip(x, i, nd, batch))
        xdot += xdoti.sum()
        xsum += xsumi.sum(axis=0)
        xcov += xcovi.sum(axis=0)
    return nn + (b * xcov).sum() + (xsum @ c @ xsum) / nz - 2 * xdot


def matmul_batch(x, y, size, env, factor):
    nk, num = x.shape
    batch = get_gpu_env(env).batch(float(factor) * nk * size, num)
    dataset = tf.data.Dataset.from_generator(
        lambda: zip(x.T, y),
        output_signature=(
            tf.TensorSpec((nk,), tf.float32),
            tf.TensorSpec((size,), tf.float32),
        ),
    )
    init = [jnp.zeros((nk, size))]
    types = ["add"]
    def calc(x, y):
        return jnp.matmul(x.T, y),
    out, = mapped_imgs(dataset, num, calc, types, init, batch, jnp.zeros(()))
    return np.array(out)
