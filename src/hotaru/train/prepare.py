from collections import namedtuple
from logging import getLogger

import jax.numpy as jnp

from .common import (
    calc_cov_out,
    matmul_batch,
)
from .dynamics import get_dynamics
from .penalty import get_penalty

logger = getLogger(__name__)

Prepare = namedtuple("Prepare", "nt nx nk nb pena a b c")


def prepare(kind, data, x1, x2, dynamics, penalty, env, factor):
    nt = data.nt
    nx = data.nx
    nk = x1.shape[0]
    nb = x2.shape[0]
    logger.info("prepare: %d %d %d %d", nt, nx, nk, nb)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)

    x1 = jnp.array(x1, jnp.float32)
    x2 = jnp.array(x2, jnp.float32)
    match kind:
        case "spatial":
            bx = penalty.bx
            by = penalty.bt
            pena = penalty.lu(x1) + penalty.lt(x2)
            xval = jnp.concatenate([dynamics(x1), x2], axis=0)
            xval /= xval.max(axis=1, keepdims=True)
            yval = data.data(mask_type=True)
            num, size = nt, nx
        case  "temporal":
            bx = penalty.bt
            by = penalty.bx
            pena = penalty.la(x1) + penalty.lx(x2)
            xval = jnp.concatenate([x1, x2], axis=0)
            xval = data.apply_mask(xval, mask_type=True)
            yval = data.datax()
            num, size = nx, nt
        case _:
            raise ValueError()

    logger.info("%s: %s %s %d", "pbar", "start", f"{kind} cor", num)
    xcor = matmul_batch(xval, yval, size, env, factor)
    logger.info("%s: %s", "pbar", "close")
    xcov, xout = calc_cov_out(xval, env, factor)

    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)

    a = xcor
    b = xcov - cx * xout
    c = xout - cy * xcov
    logger.info("%s", a)
    logger.info("%s", b)
    logger.info("%s", c)
    logger.info("%f", pena)
    return Prepare(nt, nx, nk, nb, pena, a, b, c)
