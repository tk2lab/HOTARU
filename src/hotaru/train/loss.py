from collections import namedtuple

import jax
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


def gen_optimizer(kind, factor, dynamics, penalty, lr, scale, num_devices=1):
    nt, nx, nk, py, a, b, c = factor
    nn = float(nt * nx)
    nm = float(nn + nt + nx)
    lr_scale = nm / b.diagonal().max()

    dev_id = jnp.arange(num_devices)
    a = mask_fn(a, num_devices)
    b = mask_fn(b, num_devices)
    c = mask_fn(c, num_devices)

    def _calc_err(x, i):
        if kind == "temporal":
            x = dynamics(x)
        xval, xcov, xout = calc_cov_out(x, i, num_devices)
        return (a[i] * xval).sum() + (b[i] * xcov).sum() + (c[i] * xout).sum()

    def loss_fwd(x):
        err = jax.pmap(_calc_err, in_axes=(None, 0))(x, dev_id).sum()
        var = (err + nn) / nm
        return jnp.log(var) / 2 + py, (x, err)

    def loss_bwd(res, g):
        x, err = res
        grad_err_fn = jax.pmap(jax.grad(_calc_err), in_axes=(None, 0))
        grad_err = grad_err_fn(x, dev_id).sum(axis=0)[:nk]
        return (g / 2 * grad_err / (err + nn),)

    @jax.custom_vjp
    def loss_fn(x):
        return loss_fwd(x)[0]

    loss_fn.defvjp(loss_fwd, loss_bwd)

    if kind == "temporal":
        nu = nt + dynamics.size - 1
        init = [np.zeros((nk, nu))]
        pena = [penalty.lu]
    else:
        init = [np.zeros((nk, nx))]
        pena = [penalty.la]

    opt = ProxOptimizer(loss_fn, init, pena)
    opt.set_params(lr * lr_scale, scale)
    return opt


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
