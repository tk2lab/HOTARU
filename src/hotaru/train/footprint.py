import os

import tensorflow as tf
import numpy as np
import click

from ..util.distribute import ReduceOp
from ..util.distribute import distributed

from .model import BaseModel
from .summary import normalized_and_sort
from .summary import summary_stat, summary_footprint_sample, summary_footprint_max


class FootprintModel(BaseModel):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(footprint, spike, variance, **kwargs)
        self.footprint = footprint
        self.get_spike = spike.get_val
        self.set_spike = spike.set_val
        self.spike_tensor = spike.call

    def call(self, inputs):
        footprint = self.footprint(inputs)
        loss, footprint_penalty, spike_penalty = self.call_common(footprint)
        self.add_metric(footprint_penalty, 'penalty')
        return loss

    def fit(self, spike, lr, batch, **kwargs):
        nk = spike.shape[0]
        nx = self.variance.nx
        self.set_spike(spike)
        self.start(batch)
        self.footprint.val = np.zeros((nk, nx))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_a
        return self.fit_common(FootprintCallback, **kwargs)

    def start(self, batch):

        @distributed(ReduceOp.SUM)
        def _matmul(data, calcium):
            t, d = data
            c_p = tf.gather(calcium, t, axis=1)
            return tf.matmul(c_p, d),

        data = self.variance._data.enumerate().batch(batch)
        spike = self.spike_tensor()
        calcium = self.variance.spike_to_calcium(spike)
        with click.progressbar(length=self.variance.nt, label='Initialize') as prog:
            vdat, = _matmul(data, calcium, prog=prog)
        self.variance._cache(1, calcium, vdat)


class FootprintCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def set_model(self, model):
        super().set_model(model)
        self._train_dir = os.path.join(self._log_write_dir, 'footprint')

    def on_epoch_end(self, epoch, logs=None):
        stage = self.stage
        with self._train_writer.as_default():
            val = self.model.footprint.val
            summary_stat(val, stage, step=epoch)
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        stage = self.stage
        with self._train_writer.as_default():
            val = self.model.footprint.val
            mask = self.model.footprint.mask
            summary_footprint_max(val, mask, stage, step=0)
        super().on_train_end(logs)
