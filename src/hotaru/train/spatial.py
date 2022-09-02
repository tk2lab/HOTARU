import numpy as np
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .base import BaseModel
from .variance import VarianceMode


class SpatialModel(BaseModel):
    """Spatial Model"""

    def __init__(self, data, nk, nx, nt, tau, bx, bt, la, lu, **args):
        footprint_l, spike_l, variance_l, loss_l = self.prepare_layers(
            VarianceMode.Footprint,
            data,
            nk,
            nx,
            nt,
            tau,
            bx,
            bt,
            la,
            lu,
        )

        dummy = tf.keras.Input((1,))
        footprint = footprint_l(dummy)
        variance = variance_l(footprint)
        loss = loss_l(variance)
        super().__init__(dummy, loss, **args)
        self.set_metric(footprint_l, spike_l, variance, loss)

        self.spike_tensor = spike_l.call
        self.get_spike = spike_l.get_val
        self.set_spike = spike_l.set_val
        self.footprint = footprint_l
        self.variance = variance_l

    def prepare_fit(self, spike, batch, prog=None):
        @distributed(ReduceOp.SUM, strategy=self.distribute_strategy)
        def _matmul(data, calcium):
            t, d = data
            c_p = tf.gather(calcium, t, axis=1)
            return (tf.matmul(c_p, d),)

        nk = spike.shape[0]
        nx = self.variance.nx
        self.footprint.set_val(np.zeros((nk, nx)))
        self.set_spike(spike)

        data = self.variance.data.enumerate().batch(batch)
        spike = self.spike_tensor()
        calcium = self.variance.spike_to_calcium(spike)
        (vdat,) = _matmul(data, calcium, prog=prog)
        self.lipschitz = self.variance._cache(calcium, vdat)
