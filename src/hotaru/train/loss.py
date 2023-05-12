from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from ..filter.map import mapped_imgs
from ..proxmodel.optimizer import ProxOptimizer
from .matmul import matmul_batch


Prepare = namedtuple("Prepare", "nt nx nk pena a b c")


def gen_factor(kind, yval, data, dynamics, penalty, batch, pbar=None):

    if kind == "temporal":
        bx = penalty.bt
        by = penalty.bx
        trans = True
        pena = penalty.la(yval)
    else:
        bx = penalty.bx
        by = penalty.bt
        trans = False
        pena = penalty.lu(yval)
        yval = dynamics(yval)

    ycor = matmul_batch(yval, data, trans, batch, pbar)
    yval, ycov, yout = calc_cov_out(yval)

    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)

    a = -2 * ycor
    b = ycov - cx * yout
    c = yout - cy * ycov

    nt, h, w = data.imgs.shape
    if data.mask is None:
        nx = h * w
    else:
        nx = np.count_nonzero(data.mask)
    nk = yval.shape[0]

    return Prepare(nt, nx, nk, pena, a, b, c)


def gen_optimizer(kind, factor, dynamics, penalty):

    def loss_fn(xval):
        if kind == "temporal":
            xval = dynamics(xval)
        xval, xcov, xout = calc_cov_out(xval)
        var = (nn + (a * xval).sum() + (b * xcov).sum() + (c * xout).sum()) / nm
        return jnp.log(var) / 2 + py, var

    nt, nx, nk, py, a, b, c = factor
    nn = float(nt * nx)
    nm = float(nn + nt + nx)

    if kind == "temporal":
        nu = nt + dynamics.size - 1
        init = [np.zeros((nk, nu))]
        pena = [penalty.lu]
    else:
        init = [np.zeros((nk, nx))]
        pena = [penalty.la]

    return ProxOptimizer(loss_fn, init, pena)


def calc_cov_out(xval):
    nx = xval.shape[1]
    xcov = xval @ xval.T
    xsum = xval.sum(axis=1)
    xout = xsum[:, None] * (xsum / nx)
    return xval, xcov, xout
