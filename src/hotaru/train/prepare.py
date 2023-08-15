from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import get_gpu_env
from .common import matmul_batch

logger = getLogger(__name__)


def prepare_matrix(data, y, trans, env, factor, prefetch):
    nt = data.nt
    ns = data.ns
    nk, ny = y.shape

    env = get_gpu_env(env)
    nd = env.num_devices
    sharding = env.sharding((nd, 1))
    batch = env.batch(float(factor) * nk * ns, nt)

    y = jax.device_put(y, sharding)

    nkmod = nd * ((nk + nd - 1) // nd)
    y = np.pad(y, ((0, nkmod - nk), (0, 0)))
    yval = jax.device_put(y, sharding)

    ycov = (yval @ yval.T)[:nk, :nk]
    ysum = yval.sum(axis=1)[:nk]
    yout = jnp.outer(ysum, ysum) / ny

    logger.info("%s: %s %s %d", "pbar", "start", "prepare", nt)
    ycor = matmul_batch(yval, data, trans, sharding, batch, prefetch)
    logger.info("%s: %s", "pbar", "close")

    return ycov, yout, ycor
