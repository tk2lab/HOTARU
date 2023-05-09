import jax.numpy as jnp

from ..proxmodel.optimizer import ProxOptimizer
from .loss import gen_loss


def spatial_optimizer(spike, data, dynamics, penalty, batch, pbar=None):

    def loss_fn(footprint):
        loss, var = _loss(footprint)
        return loss + _pena, var

    calcium = dynamics(spike)
    _loss = gen_loss("spatial", calcium, data, penalty, batch, pbar)
    _pena = penalty.lu(spike)
    nk = spike.shape[0]
    nx = jnp.count_nonzero(data.mask)
    footprint = jnp.zeros((nk, nx))
    return ProxOptimizer(loss_fn, [footprint], [penalty.lu])
