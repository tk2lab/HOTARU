import numpy as np
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .base import BaseModel
from .loss import LossLayer


class TemporalModel(BaseModel):
    """Temporal Model"""

    def __init__(
        self, data, nk, nx, nt, hz, rise, fall, scale, bx, bt, la, lu, **args
    ):
        layers = self.prepare_layers(nk, nx, nt, hz, rise, fall, scale)
        spike_to_calcium, dummy, footprint_l, spike_l = layers

        loss_l = LossLayer(nk, nt, nx, bt, bx)
        footprint_l.prox.set_l(la / loss_l._nm)
        spike_l.prox.set_l(lu / loss_l._nm)

        spike, spike_penalty = spike_l(dummy)
        calcium = spike_to_calcium(spike)
        loss = loss_l(calcium)
        super().__init__(dummy, loss, **args)

        footprint, footprint_penalty = footprint_l(dummy)
        penalty = footprint_penalty + spike_penalty
        self.add_metric(loss + penalty, "score")

        self.spike_to_calcium = spike_to_calcium
        self.footprint_tensor = lambda: footprint_l.val
        self.get_footprint = footprint_l.get_val
        self.set_footprint = footprint_l.set_val
        self.spike = spike_l
        self.data = data
        self._cache = loss_l._cache

    def prepare_fit(self, footprint, batch, prog=None):
        @distributed(ReduceOp.CONCAT, strategy=self.local_strategy)
        def _matmul(data, footprint):
            return (tf.matmul(data, footprint, False, True),)

        nk = footprint.shape[0]
        self.spike.set_val(np.zeros((nk, self.nu)))
        self.set_footprint(footprint)

        data = self.data.batch(batch)
        footprint = self.footprint_tensor()
        (adat,) = _matmul(data, footprint, prog=prog)
        lipschitz = self._cache(footprint, tf.transpose(adat))
        gsum = tf.math.reduce_sum(self.spike_to_calcium.kernel)
        self.lipschitz = (lipschitz * gsum).numpy()
