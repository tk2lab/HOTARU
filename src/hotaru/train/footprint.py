import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

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
        self.add_metric(footprint_penalty, 'mean', 'penalty')
        return loss

    def fit(self, spike, lr, batch, **kwargs):
        nk = spike.shape[0]
        nx = self.variance.nx
        self.set_spike(spike)
        self.start(batch)
        self.footprint.val = np.zeros((nk, nx))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_a
        self.fit_common(FootprintCallback, **kwargs)

    def start(self, batch):
        data = self.variance._data.enumerate().batch(batch)
        spike = self.spike_tensor()
        calcium = self.variance.spike_to_calcium(spike)
        vdat = K.constant(0.0)
        prog = tf.keras.utils.Progbar(self.variance.nt)
        for t, d in data:
            c_p = tf.gather(calcium, t, axis=1)
            vdat += tf.matmul(c_p, d)
            prog.add(tf.size(t).numpy())
        self.variance._cache(1, calcium, vdat)


class FootprintCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def on_train_end(self, logs=None):
        super().on_train_end(logs)

        stage = self.stage
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            mask = self.model.footprint.mask
            val = self.model.footprint.val
            val = summary_stat(val, stage, step=0)
            summary_footprint_max(val, mask, stage, step=0)
