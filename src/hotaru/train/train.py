from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import get_gpu_env
from .common import mask_fn, calc_err
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .prepare import prepare

logger = getLogger(__name__)


def spatial(data, x, dynamics, penalty, env, factor, lr, scale, n_epoch, n_step, tol):
    logger.info("spatial:")
    common = dynamics, penalty, env
    mat = prepare("spatial", data, *x, *common, factor[0])
    logger.debug("mat: %d %d %d %f %s %s %s", *mat[:4], *(m.shape for m in mat[4:]))
    logger.debug("opt: %f %f %d %d %f", lr, scale, n_epoch, n_step, tol)
    optimizer = get_optimizer("spatial", mat, *common, lr, scale, factor[1])
    logger.debug("opt: %s", optimizer)
    log = optimizer.fit(n_epoch, n_step, tol)
    logger.debug("opt: %s", log)
    return np.array(jnp.concatenate(optimizer.x, axis=0))


def temporal(data, x, dynamics, penalty, env, factor, lr, scale, n_epoch, n_step, tol):
    logger.info("temporal:")
    common = dynamics, penalty, env
    mat = prepare("temporal", data, *x, *common, factor[0])
    logger.debug("mat: %d %d %d %f %s %s %s", *mat[:4], *(m.shape for m in mat[4:]))
    optimizer = get_optimizer("temporal", mat, *common, lr, scale, factor[1])
    optimizer.fit(n_epoch, n_step, tol)
    return optimizer.val


def get_optimizer(kind, mat, dynamics, penalty, env, lr, scale, factor):
    def _loss_fn(x):
        err = calc_err(nn, a, b, c, x, env, factor)
        var = err / nm
        loss = jnp.log(var) / 2 + py
        jax.debug.print("{x} {y} {z}", x=err, y=var, z=loss)
        return loss

    def loss_fn(x1, x2):
        if kind == "temporal":
            x1 = mask_fn(x1, env.num_devices)
            x1 = jnp.concatenate(jax.pmap(dynamics)(x1), axis=0)[:nk]
        x = jnp.concatenate([x1, x2], axis=0)
        return _loss_fn(x)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)
    env = get_gpu_env(env)

    nt, nx, nk, nb, py, a, b, c = mat
    ntf = float(nt)
    nxf = float(nx)
    nn = ntf * nxf
    nm = nn + ntf + nxf
    py /= nm
    logger.info("%s", (nt, nx, nk, nb, py, a.shape, b.shape, c.shape))

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

    opt = ProxOptimizer(loss_fn, init, pena, env, f"{kind} optimize")
    opt.set_params(lr / b.diagonal().max(), scale, nm)
    return opt
