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
    logger.info("spatial:")
    common = dynamics, penalty, env
    mat = prepare("spatial", data, *x, *common, factor)
    logger.debug("mat: %d %d %d %f %s %s %s", *mat[:4], *(m.shape for m in mat[4:]))
    logger.debug("opt: %f %f %d %d %f", lr, scale, n_epoch, n_step, tol)
    optimizer = get_optimizer("spatial", mat, *common, lr, scale)
    logger.debug("opt: %s", optimizer)
    log = optimizer.fit(n_epoch, n_step, tol)
    logger.debug("opt: %s", log)
    return np.array(jnp.concatenate(optimizer.x, axis=0))


def temporal(data, x, dynamics, penalty, env, factor, lr, scale, n_epoch, n_step, tol):
    logger.info("temporal:")
    common = dynamics, penalty, env
    mat = prepare("temporal", data, *x, *common, factor)
    logger.debug("mat: %d %d %d %f %s %s %s", *mat[:4], *(m.shape for m in mat[4:]))
    optimizer = get_optimizer("temporal", mat, *common, lr, scale)
    optimizer.fit(n_epoch, n_step, tol)
    return optimizer.val


def get_optimizer(kind, factor, dynamics, penalty, env, lr, scale):
    def _calc_err(x, i):
        xval, xcov, xout = calc_cov_out(x, i, env.num_devices)
        return nn + (b[i] * xcov).sum() + (c[i] * xout).sum() - 2 * (a[i] * xval).sum()

    calc_err = jax.pmap(_calc_err, in_axes=(None, 0))
    calc_grad_err = jax.pmap(jax.grad(_calc_err), in_axes=(None, 0))

    def loss_fwd(x):
        err = calc_err(x, dev_id).sum()
        var = err / nm
        return jnp.log(var) / 2 + py, (x, err)

    def loss_bwd(res, g):
        x, err = res
        grad_err = calc_grad_err(x, dev_id).sum(axis=0)
        return (g / 2 * grad_err / (err + nn),)

    @jax.custom_vjp
    def _loss_fn(x):
        return loss_fwd(x)[0]

    _loss_fn.defvjp(loss_fwd, loss_bwd)

    def loss_fn(x1, x2):
        if kind == "temporal":
            x1 = mask_fn(x1, env.num_devices)
            x1 = jnp.concatenate(jax.pmap(dynamics)(x1), axis=0)[:nk]
        x = jnp.concatenate([x1, x2], axis=0)
        return _loss_fn(x)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)
    nt, nx, nk, nb, py, a, b, c = factor
    ntf = float(nt)
    nxf = float(nx)
    nn = ntf * nxf
    nm = nn + ntf + nxf
    py /= nm
    logger.info("%s", (nt, nx, nk, nb, py, a.shape, b.shape, c.shape))

    dev_id = jnp.arange(env.num_devices)
    a = mask_fn(a, env.num_devices)
    b = mask_fn(b, env.num_devices)
    c = mask_fn(c, env.num_devices)

    match kind:
        case "spatial":
            init = [np.zeros((nk, nx)), np.zeros((nb, nx))]
            pena = [penalty.la, penalty.lx]
        case "temporal":
            nu = nt + dynamics.size - 1
            init = [np.zeros((nk, nu)), np.zeros((nb, nt))]
            pena = [penalty.lu, penalty.lt]
        case _:
            raise ValueError("kind must be temporal or spatial")
    logger.debug("init: %s", init[0].shape)

    opt = ProxOptimizer(loss_fn, init, pena, env.num_devices, f"{kind} optimize")
    opt.set_params(lr / b.diagonal().max(), scale, nm)
    return opt
