import numpy as np
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .base import BaseModel
from .loss import LossLayer


class SpatialModel(BaseModel):
    """Spatial Model"""

    def __init__(
        self, data, nk, nx, nt, hz, rise, fall, scale, bx, bt, la, lu, **args
    ):
        layers = self.prepare_layers(nk, nx, nt, hz, rise, fall, scale)
        spike_to_calcium, dummy, footprint_l, spike_l = layers

        loss_l = LossLayer(nk, nx, nt, bx, bt)
        footprint_l.prox.set_l(la / loss_l._nm)
        spike_l.prox.set_l(lu / loss_l._nm)

        footprint, footprint_penalty = footprint_l(dummy)
        loss = loss_l(footprint)
        super().__init__(dummy, loss, **args)

        spike, spike_penalty = spike_l(dummy)
        penalty = footprint_penalty + spike_penalty
        self.add_metric(loss + penalty, "score")

        self.spike_to_calcium = spike_to_calcium
        self.spike_tensor = lambda: spike_l.val
        self.get_spike = spike_l.get_val
        self.set_spike = spike_l.set_val
        self.footprint = footprint_l
        self.data = data
        self._cache = loss_l._cache

    def prepare_fit(self, spike, batch, prog=None):
        @distributed(ReduceOp.SUM, strategy=self.distribute_strategy)
        def _matmul(data, calcium):
            t, d = data
            c_p = tf.gather(calcium, t, axis=1)
            return (tf.matmul(c_p, d),)

        nk = spike.shape[0]
        self.footprint.set_val(np.zeros((nk, self.nx)))
        self.set_spike(spike)

        data = self.data.enumerate().batch(batch)
        spike = self.spike_tensor()
        calcium = self.spike_to_calcium(spike)
        (vdat,) = _matmul(data, calcium, prog=prog)
        self.lipschitz = self._cache(calcium, vdat).numpy()
