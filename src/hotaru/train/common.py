import jax.numpy as jnp
import numpy as np

from ..filter.map import mapped_imgs
from ..utils import get_progress


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


def matmul_batch(x, y, trans, batch, pbar=None):
    imgs, mask, _, avgx, avgt, std0, *_ = y

    if trans:

        def prepare(start, end):
            return (
                jnp.array(imgs[start:end], jnp.float32),
                jnp.array(avgt[start:end], jnp.float32),
            )

        def calc(imgs, avgt):
            if mask is None:
                y = ((imgs - avgx).reshape(-1, h * w) - avgt[:, jnp.newaxis]) / std0
            else:
                y = ((imgs - avgx)[:, mask] - avgt[:, None]) / std0
            return jnp.matmul(y, x.T)

        types = [("stack", jnp.float32)]

    else:

        def prepare(start, end):
            return (
                jnp.array(x.T[start:end], jnp.float32),
                jnp.array(imgs[start:end], jnp.float32),
                jnp.array(avgt[start:end], jnp.float32),
            )

        def calc(x, imgs, avgt):
            if mask is None:
                y = ((imgs - avgx).reshape(-1, h * w) - avgt[:, None]) / std0
            else:
                y = ((imgs - avgx)[:, mask] - avgt[:, None]) / std0
            return jnp.matmul(y.T, x)

        types = [("add", jnp.float32)]

    nt, h, w = imgs.shape

    pbar = get_progress(pbar)
    pbar.set_count(nt)
    out, = mapped_imgs(nt, prepare, calc, types, batch, pbar)
    return np.array(out.T)
