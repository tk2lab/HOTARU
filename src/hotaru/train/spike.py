import os

import click
import numpy as np
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .model import BaseModel
from .variance import VarianceMode


class SpikeModel(BaseModel):
    """Spike Model"""

    def __init__(self, data, nk, nx, nt, tau, bx, bt, la, lu, **args):
        super().__init__(**args)
        footprint, spike, variance = self.prepare(
            VarianceMode.Spike, data, nk, nx, nt, tau, bx, bt, la, lu,
        )
        self.spike = spike
        self.footprint_tensor = footprint.call
        self.get_footprint = footprint.get_val
        self.set_footprint = footprint.set_val

    def call(self, inputs):
        spike = self.spike(inputs)
        calcium = self.variance.spike_to_calcium(spike)
        loss, footprint_penalty, spike_penalty = self.call_common(calcium)
        self.add_metric(footprint_penalty, "penalty")
        return loss

    def prepare_fit(self, footprint, batch):
        @distributed(ReduceOp.CONCAT, strategy=self.distribute_strategy)
        def _matmul(data, footprint):
            return (tf.matmul(data, footprint, False, True),)

        nk = footprint.shape[0]
        nu = self.variance.nu
        self.spike.set_val(np.zeros((nk, nu)))
        self.set_footprint(footprint)

        data = self.variance.data.batch(batch)
        footprint = self.footprint_tensor()
        with click.progressbar(
            length=self.variance.nt, label="Initialize"
        ) as prog:
            (adat,) = _matmul(data, footprint, prog=prog)
        return self.variance._cache(footprint, tf.transpose(adat))
