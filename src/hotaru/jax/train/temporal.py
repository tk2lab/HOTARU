import jax.numpy as jnp

from ..proxmodel.optimizer import ProxOptimizer
from ..proxmodel.regularizer import Regularizer
from .loss import gen_loss


def temporal_optimizer(footprint, data, dynamics, lu, la, bt, bx, batch, pbar=None):
    def loss(spike):
        calsium = dynamics(spike)
        return _loss(calsium)

    _loss = gen_loss(footprint, *data, la, bt, bx, True, batch, pbar)
    nk = footprint.shape[0]
    nt = data[0].shape[0]
    spike0 = jnp.zeros((nk, nt + dynamics.size - 1))
    return ProxOptimizer(loss, [spike0], [lu])
