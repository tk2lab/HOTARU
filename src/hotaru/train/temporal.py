import numpy as np
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .base import BaseModel
from .variance import VarianceMode


class TemporalModel(BaseModel):
    """Temporal Model"""

    def __init__(self, data, nk, nx, nt, tau, bx, bt, la, lu, **args):
        footprint_l, spike_l, variance_l, loss_l = self.prepare_layers(
            VarianceMode.Spike,
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
        spike = spike_l(dummy)
        calcium = variance_l.spike_to_calcium(spike)
        variance = variance_l(calcium)
        loss = loss_l(variance)
        super().__init__(dummy, loss, **args)

        footprint = footprint_l(dummy)
        penalty = footprint_l.penalty(footprint) + spike_l.penalty(spike)
        self.set_metric(penalty, variance, loss)

        self.footprint_tensor = footprint_l.call
        self.get_footprint = footprint_l.get_val
        self.set_footprint = footprint_l.set_val
        self.spike = spike_l
        self.variance = variance_l

    def prepare_fit(self, footprint, batch, prog=None):
        @distributed(ReduceOp.CONCAT, strategy=self.distribute_strategy)
        def _matmul(data, footprint):
            return (tf.matmul(data, footprint, False, True),)

        nk = footprint.shape[0]
        nu = self.variance.nu
        self.spike.set_val(np.zeros((nk, nu)))
        self.set_footprint(footprint)

        data = self.variance.data.batch(batch)
        footprint = self.footprint_tensor()
        (adat,) = _matmul(data, footprint, prog=prog)
        self.lipschitz = self.variance._cache(footprint, tf.transpose(adat))
