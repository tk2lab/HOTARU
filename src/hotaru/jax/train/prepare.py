import jax.numpy as jnp

from ..utils.normalized import (
    apply_to_normalized,
    default_buffer,
)


def prepare(val, imgs, stats, bx, by, trans, buffer=None, num_devices=None, pbar=None):
    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)
    cor = matmul(val, imgs, stats, trans, buffer, num_devices, pbar)
    yval, ycov, yout = calc_cov_out(val)
    dat = -2 * cor
    cov = ycov - cx * yout
    out = yout - cy * ycov
    return dat, cov, out


def calc_cov_out(xval):
    nx = xval.shape[1]
    xcov = xval @ xval.T
    xsum = xval.sum(axis=1)
    xout = xsum[:, None] * (xsum / nx)
    return xval, xcov, xout


def matmul(x, imgs, stats, trans, buffer=None, num_devices=None, pbar=None):
    if buffer is None:
        buffer = default_buffer
    nt, h, w = imgs.shape
    size = 4 * h * w
    batch = (buffer + size - 1) // size
    if trans:

        def apply(imgs, mask, start, end):
            return jnp.matmul(x, imgs[:, mask].T)

        def finish(out):
            return jnp.concatenate(out, axis=1)

    else:

        def apply(imgs, mask, start, end):
            return jnp.matmul(x[start:end], imgs[:, mask])

        def finish(out):
            return out.sum(axis=0)

    gen = apply_to_normalized(apply, finish, imgs, stats, batch, num_devices, pbar)
    if trans:
        return jnp.concatenate([o for t, o in gen], axis=1)
    else:
        return jnp.sum([o for t, o in gen], axis=0)
