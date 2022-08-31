import os

import click
import numpy as np
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .model import BaseModel
from .summary import FootprintSummaryCallback
from .variance import VarianceMode


class FootprintModel(BaseModel):
    """Footprint Model"""

    Callback = FootprintSummaryCallback

    def __init__(self, data, nk, nx, nt, tau, bx, bt, la, lu, **args):
        super().__init__(**args)
        footprint, spike, variance = self.prepare(
            VarianceMode.Footprint, data, nk, nx, nt, tau, bx, bt, la, lu,
        )
        self.footprint = footprint
        self.spike_tensor = spike.call
        self.get_spike = spike.get_val
        self.set_spike = spike.set_val
        super().__init__()

    def call(self, dummy=None):
        footprint = self.footprint(dummy)
        loss, footprint_penalty, spike_penalty = self.call_common(footprint)
        self.add_metric(footprint_penalty, "penalty")
        return loss

    def fit(self, spike, batch, **kwargs):
        nk = spike.shape[0]
        nx = self.variance.nx
        self.footprint.set_val(np.zeros((nk, nx)))
        self.set_spike(spike)
        self.start(batch)
        return self.fit_common(**kwargs)

    def start(self, batch):
        @distributed(ReduceOp.SUM, strategy=self.distribute_strategy)
        def _matmul(data, calcium):
            t, d = data
            c_p = tf.gather(calcium, t, axis=1)
            return (tf.matmul(c_p, d),)

        data = self.variance.data.enumerate().batch(batch)
        spike = self.spike_tensor()
        calcium = self.variance.spike_to_calcium(spike)
        with click.progressbar(
            length=self.variance.nt, label="Initialize"
        ) as prog:
            (vdat,) = _matmul(data, calcium, prog=prog)
        self.variance._cache(calcium, vdat)
