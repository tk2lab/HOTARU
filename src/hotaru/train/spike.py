import os

import tensorflow as tf
import numpy as np
import click

from ..util.distribute import ReduceOp
from ..util.distribute import distributed

from .model import BaseModel
from .summary import summary_stat, summary_spike


class SpikeModel(BaseModel):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(footprint, spike, variance, **kwargs)
        self.spike = spike
        self.get_footprint = footprint.get_val
        self.set_footprint = footprint.set_val
        self.footprint_tensor = footprint.call

    def call(self, inputs):
        spike = self.spike(inputs)
        calcium = self.variance.spike_to_calcium(spike)
        loss, footprint_penalty, spike_penalty = self.call_common(calcium)
        self.add_metric(footprint_penalty, 'penalty')
        return loss

    def fit(self, footprint, lr, batch, **kwargs):
        nk = footprint.shape[0]
        nu = self.variance.nu
        self.set_footprint(footprint)
        self.start(batch)
        self.spike.val = np.zeros((nk, nu))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_u
        return self.fit_common(SpikeCallback, **kwargs)

    def start(self, batch):

        @distributed(ReduceOp.CONCAT)
        def _matmul(data, footprint):
            return tf.matmul(data, footprint, False, True),

        data = self.variance._data.batch(batch)
        footprint = self.footprint_tensor()
        with click.progressbar(length=self.variance.nt, label='Initialize') as prog:
            adat, = _matmul(data, footprint, prog=prog)
        self.variance._cache(0, footprint, tf.transpose(adat))


class SpikeCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def set_model(self, model):
        super().set_model(model)
        self._train_dir = os.path.join(self._log_write_dir, 'spike')

    def on_epoch_end(self, epoch, logs=None):
        stage = self.stage
        with self._train_writer.as_default():
            summary_stat(self.model.spike.val, stage, step=epoch)
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        stage = self.stage
        with self._train_writer.as_default():
            summary_spike(self.model.spike.val, stage, step=0)
        super().on_train_end(logs)
