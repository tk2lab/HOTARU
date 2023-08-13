from collections import namedtuple
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PositionalSharding

from ..utils import get_gpu_env
from .common import matmul_batch
from .dynamics import get_dynamics
from .penalty import get_penalty

logger = getLogger(__name__)

Prepare = namedtuple("Prepare", "nt nx nk nb pena a b c")


def prepare_matrix(kind, data, x1, x2, dynamics, penalty, env, factor, prefetch):
    nt = data.nt
    ns = data.ns
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nk = n1 + n2

    nd, batch = get_gpu_env(env).batch(float(factor) * nk * ns, nt)
    sharding = PositionalSharding(create_device_mesh((nd, 1)))
    logger.info("prepare: %f %d %d %d %d %d %d", factor, nt, ns, n1, n2, nd, batch)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)

    n1mod = nd * ((n1 + nd - 1) // nd)
    n2mod = nd * ((n2 + nd - 1) // nd)
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x1 = np.pad(x1, ((0, n1mod - n1),) + ((0, 0),) * (x1.ndim - 1))
    x2 = np.pad(x2, ((0, n2mod - n2),) + ((0, 0),) * (x2.ndim - 1))
    x1 = jax.device_put(x1, sharding)
    x2 = jax.device_put(x2, sharding)

    match kind:
        case "spatial":
            pena = penalty.lu(x1) + penalty.lt(x2)
            x1 = dynamics(x1)
            xval = jnp.concatenate([x1, x2], axis=0)
            xval /= xval.max(axis=1, keepdims=True)
            trans = False
            bx = penalty.bx
            by = penalty.bt
            nx, ny = ns, nt
        case "temporal":
            pena = penalty.la(x1) + penalty.lx(x2)
            xval = jnp.concatenate([x1, x2], axis=0)
            xval = data.apply_mask(xval, mask_type=True)
            trans = True
            bx = penalty.bt
            by = penalty.bx
            nx, ny = nt, ns
        case _:
            raise ValueError()

    xcov = (xval @ xval.T)[:nk, :nk]
    xsum = xval.sum(axis=1)[:nk]
    xout = jnp.outer(xsum, xsum) / ny

    logger.info("%s: %s %s %d", "pbar", "start", f"{kind} prepare", nt)
    xcor = matmul_batch(xval, data, trans, sharding, nd * batch, prefetch)
    logger.info("%s: %s", "pbar", "close")

    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)

    pena = np.array(pena)
    a = np.array(xcor)
    b = np.array(xcov - cx * xout)
    c = np.array(xout - cy * xcov)
    return Prepare(nx, ny, n1, n2, pena, a, b, c)
