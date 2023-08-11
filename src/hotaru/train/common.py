from logging import getLogger

import tensorflow as tf
import jax.numpy as jnp
import numpy as np

from ..filter.map import mapped_imgs

logger = getLogger(__name__)


def mask_fn(x, num_devices):
    n, *shape = x.shape
    d = n % num_devices
    xval = jnp.pad(x, [[0, d]] + [[0, 0]] * len(shape))
    return xval.reshape(num_devices, (n + d) // num_devices, *shape)


def calc_cov_out(x0, i=0, num_devices=1):
    nx = x0.shape[1]
    xs = x0.sum(axis=1)
    xval = mask_fn(x0, num_devices)[i]
    xsum = mask_fn(xs, num_devices)[i]
    xcov = xval @ x0.T
    xout = xsum[:, None] * (xs / nx)
    return xval, xcov, xout


def matmul_batch(data, x, trans, batch):
    logger.info("matmul: %s %s %s", batch, data.imgs.shape, x.shape)
    nt = data.nt
    nx = data.nx
    nk = x.shape[0]

    if trans:
        dataset = tf.data.Dataset.from_generator(
            lambda: enumerate(data.data(mask_type=True)),
            output_signature=(
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((nx,), tf.float32),
            ),
        )
        types = [("stack", -1)]
        init = [jnp.empty((nt + 1, nk))]
        def calc(index, y):
            return jnp.matmul(y, x.T), index
    else:
        dataset = tf.data.Dataset.from_generator(
            lambda: zip(data.data(mask_type=True), x.T),
            output_signature=(
                tf.TensorSpec((nx,), tf.float32),
                tf.TensorSpec((nk,), tf.float32),
            ),
        )
        types = ["add"]
        init = [jnp.zeros((nx, nk))]
        def calc(y, x):
            return jnp.matmul(y.T, x),

    out, = mapped_imgs(dataset, nt, calc, types, init, batch, jnp.zeros(()))
    if trans:
        out = out[:-1]
    return np.array(out.T)
