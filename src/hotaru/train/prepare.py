from collections import namedtuple
from logging import getLogger

import jax.numpy as jnp

from ..utils import (
    get_gpu_env,
)
from .common import (
    calc_cov_out,
    matmul_batch,
)
from .dynamics import get_dynamics
from .penalty import get_penalty

logger = getLogger(__name__)

Prepare = namedtuple("Prepare", "nt nx nk nb pena a b c")


def prepare(kind, data, y1, y2, dynamics, penalty, env, factor=10):
    nt = data.nt
    nx = data.nx
    nk = y1.shape[0]
    nb = y2.shape[0]
    logger.info("prepare: %d %d %s %s", nt, nx, y1.shape, y2.shape)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)
    batch = get_gpu_env(env).batch(float(factor) * (nk + nb) * nx, nt)

    y1 = jnp.array(y1, jnp.float32)
    y2 = jnp.array(y2, jnp.float32)
    match kind:
        case "spatial":
            bx = penalty.bx
            by = penalty.bt
            trans = False
            pena = penalty.lu(y1) + penalty.lt(y2)
            yval = jnp.concatenate([dynamics(y1), y2], axis=0)
            logger.debug("yval: %s %s", yval.min(axis=1), yval.max(axis=1))
            yval /= yval.max(axis=1, keepdims=True)
        case  "temporal":
            bx = penalty.bt
            by = penalty.bx
            trans = True
            pena = penalty.la(y1) + penalty.lx(y2)
            yval = jnp.concatenate([y1, y2], axis=0)
            yval = data.apply_mask(yval, mask_type=True)
        case _:
            raise ValueError()

    logger.info("%s: %s %s %d", "pbar", "start", f"{kind} prepare", nt)
    ycor = matmul_batch(data, yval, trans, batch)
    logger.info("%s: %s", "pbar", "close")
    yval, ycov, yout = calc_cov_out(yval)

    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)

    a = ycor
    b = ycov - cx * yout
    c = yout - cy * ycov

    logger.debug("ycor %s", ycor)
    logger.debug("ycov %s", ycov)
    logger.debug("yout %s", yout)
    logger.debug("c %f %f", cx, cy)
    return Prepare(nt, nx, nk, nb, pena, a, b, c)
