from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    calc_cov_out,
    mask_fn,
)
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .prepare import prepare

logger = getLogger(__name__)


def spatial(data, x, dynamics, penalty, env, factor, lr, scale, n_epoch, n_step, tol):
    common = dynamics, penalty, env
    mat = prepare("spatial", data, x, *common, factor)
    logger.info("mat: %d %d %d %f %s %s %s", *mat[:4], *(m.shape for m in mat[4:]))
    logger.info("opt: %f %f %d %d %f", lr, scale, n_epoch, n_step, tol)
    optimizer = get_optimizer("spatial", mat, *common, lr, scale)
    logger.info("opt: %s", optimizer)
    log = optimizer.fit(n_epoch, n_step, tol)
    logger.info("opt: %s", log)
    return optimizer.val[0]


def temporal(data, x, dynamics, penalty, env, factor, lr, scale, n_epoch, n_step, tol):
    common = dynamics, penalty, env
    mat = prepare("temporal", data, x, *common, factor)
    logger.info("mat: %d %d %d %f %s %s %s", *mat[:4], *(m.shape for m in mat[4:]))
    optimizer = get_optimizer("temporal", mat, *common, lr, scale)
    optimizer.fit(n_epoch, n_step, tol)
    return optimizer.val[0]


def get_optimizer(kind, factor, dynamics, penalty, env, lr, scale):
    def _calc_err(x, i):
        if kind == "temporal":
            x = dynamics(x)
        xval, xcov, xout = calc_cov_out(x, i, env.num_devices)
        return nn + (b[i] * xcov).sum() + (c[i] * xout).sum() - 2 * (a[i] * xval).sum()

    def loss_fwd(x):
        err = jax.pmap(_calc_err, in_axes=(None, 0))(x, dev_id).sum()
        var = err / nm
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

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)
    nt, nx, nk, py, a, b, c = factor
    ntf = float(nt)
    nxf = float(nx)
    nn = ntf * nxf
    nm = nn + ntf + nxf
    py /= nm

    dev_id = jnp.arange(env.num_devices)
    a = mask_fn(a, env.num_devices)
    b = mask_fn(b, env.num_devices)
    c = mask_fn(c, env.num_devices)

    match kind:
        case "spatial":
            init = [np.zeros((nk, nx))]
            pena = [penalty.la]
        case "temporal":
            nu = nt + dynamics.size - 1
            init = [np.zeros((nk, nu))]
            pena = [penalty.lu]
        case _:
            raise ValueError("kind must be temporal or spatial")
    logger.info("init: %s", init[0].shape)

    opt = ProxOptimizer(loss_fn, init, pena, env.num_devices, f"{kind} optimize")
    opt.set_params(lr / b.diagonal().max(), scale, nm)
    return opt
