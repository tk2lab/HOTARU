from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PositionalSharding

from ..utils import get_gpu_env
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .prepare import prepare_matrix

logger = getLogger(__name__)


def spatial(data, x, dynamics, penalty, env, prepare, optimize, step):
    logger.info("spatial:")
    common = dynamics, penalty, env
    mat = prepare_matrix("spatial", data, *x, *common, **prepare)
    optimizer = get_optimizer("spatial", mat, *common, **optimize)
    optimizer.fit(**step)
    return np.array(jnp.concatenate(optimizer.x, axis=0))


def temporal(data, x, dynamics, penalty, env, prepare, optimize, step):
    logger.info("temporal:")
    common = dynamics, penalty, env
    mat = prepare_matrix("temporal", data, *x, *common, **prepare)
    optimizer = get_optimizer("temporal", mat, *common, **optimize)
    optimizer.fit(**step)
    return optimizer.val


def get_optimizer(kind, mat, dynamics, penalty, env, lr, scale):
    def loss_fn(x1, x2):
        if kind == "temporal":
            x1 = dynamics(x1)
        x = jnp.concatenate([x1, x2], axis=0)
        xcov = x @ x.T
        xsum = x.sum(axis=1)
        err = nn + (b * xcov).sum() + (xsum @ c @ xsum) / nx - 2 * (a * x).sum()
        var = err / nm
        loss = jnp.log(var) / 2 + py
        return loss

    nx, ny, n1, n2, py, a, b, c = mat
    nk = n1 + n2
    nd = get_gpu_env(env).num_devices
    n1mod = nd * ((n1 + nd - 1) // nd)
    n2mod = nd * ((n2 + nd - 1) // nd)
    nkmod = n1mod + n2mod
    logger.info("optimize: %d %d %d %d", nx, ny, n1, n2)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)
    sharding = PositionalSharding(create_device_mesh((nd, 1)))

    match kind:
        case "spatial":
            init = [np.zeros((n1mod, nx)), np.zeros((n2mod, nx))]
            pena = [penalty.la, penalty.lx]
        case "temporal":
            nu = nx + dynamics.size - 1
            init = [np.zeros((n1mod, nu)), np.zeros((n2mod, nx))]
            pena = [penalty.lu, penalty.lt]
        case _:
            raise ValueError("kind must be temporal or spatial")

    a = jax.device_put(jnp.pad(a, ((0, nkmod - nk), (0, 0))), sharding)
    b = jnp.pad(b, ((0, nkmod - nk), (0, nkmod - nk)))
    c = jnp.pad(c, ((0, nkmod - nk), (0, nkmod - nk)))
    nxf = float(nx)
    nyf = float(ny)
    nn = nxf * nyf
    nm = nn + nxf + nyf
    py /= nm

    opt = ProxOptimizer(loss_fn, init, pena, sharding, f"{kind} optimize")
    opt.set_params(lr / b.diagonal().max(), scale, nm)
    return opt
