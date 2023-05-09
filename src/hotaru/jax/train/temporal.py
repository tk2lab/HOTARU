import jax.numpy as jnp

from ..proxmodel.optimizer import ProxOptimizer
from .loss import gen_loss


def temporal_optimizer(footprint, data, dynamics, penalty, batch, pbar=None):

    def loss_fn(spike):
        calsium = dynamics(spike)
        loss, var = _loss(calsium)
        return loss + _pena, var

    _loss = gen_loss("temporal", footprint, data, penalty, batch, pbar)
    _pena = penalty.la(footprint)
    nk = footprint.shape[0]
    nu = data.imgs.shape[0] + dynamics.size - 1
    spike = jnp.zeros((nk, nu))
    return ProxOptimizer(loss_fn, [spike], [penalty.lu])
