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
        self.spike_get = spike.get_val
        self.spike_set = spike.set_val

    def start(self, spike, batch):
        data = self.variance._data.enumerate().batch(batch)
        spike = K.constant(spike)
        calcium = self.variance.spike_to_calcium(spike)
        vdat = K.constant(0.0)
        prog = tf.keras.utils.Progbar(self.variance.nt)
        for t, d in data:
            c_p = tf.gather(calcium, t, axis=1)
            vdat += tf.matmul(c_p, d)
            prog.add(tf.size(t).numpy())
        self.variance._cache(1, calcium, vdat)

    def call(self, inputs):
        footprint = self.footprint(inputs)
        loss, footprint_penalty, spike_penalty = self.call_common(footprint)
        self.add_metric(footprint_penalty, 'mean', 'penalty')
        return loss

    def fit(self, lr, batch, **kwargs):
        spike = self.spike_get()
        self.start(spike, batch)
        nk = spike.shape[0]
        nx = self.variance.nx
        self.footprint.val = np.zeros((nk, nx))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_a
        self.fit_common(FootprintCallback, **kwargs)


class FootprintCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        stage = self.stage
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            mask = self.model.footprint.mask
            val = self.model.footprint.val
            val = summary_stat(val, stage, step=epoch)
            summary_footprint_max(val, mask, stage, step=epoch)
            #summary_footprint_sample(val, mask, stage, step=epoch)
