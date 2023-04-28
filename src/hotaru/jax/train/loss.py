import jax.numpy as jnp

from ...io.mask import mask_range
from ..filter.map import mapped_imgs
from .matmul import matmul_batch


def gen_loss(yval, imgs, mask, avgx, avgt, std0, ly, bx, by, trans, batch, pbar=None):
    if trans:
        yval = yval[:, mask]

    ycor = matmul_batch(yval, imgs, mask, avgx, avgt, std0, trans, batch, pbar)
    yval, ycov, yout = calc_cov_out(yval)
    penalty_y = ly(yval)

    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)

    a = -2 * ycor
    b = ycov - cx * yout
    c = yout - cy * ycov

    nt, h, w = imgs.shape
    nn = nt * h * w
    nm = nn + nt + h * w

    def loss(xval):
        xval, xcov, xout = calc_cov_out(xval)
        var = (nn + (a * xval).sum() + (b * xcov).sum() + (c * xout).sum()) / nm
        return jnp.log(var) / 2 + penalty_y, var

    return loss


def calc_cov_out(xval):
    nx = xval.shape[1]
    xcov = xval @ xval.T
    xsum = xval.sum(axis=1)
    xout = xsum[:, None] * (xsum / nx)
    return xval, xcov, xout
