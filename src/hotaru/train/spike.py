import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

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
        self.add_metric(footprint_penalty, 'mean', 'penalty')
        return loss

    def fit(self, footprint, lr, batch, verbose, **kwargs):
        nk = footprint.shape[0]
        nu = self.variance.nu
        self.set_footprint(footprint)
        self.start(batch, verbose)
        self.spike.val = np.zeros((nk, nu))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_u
        self.fit_common(SpikeCallback, verbose=verbose, **kwargs)

    def start(self, batch, verbose):
        data = self.variance._data.batch(batch)
        footprint = self.footprint_tensor()
        adat = tf.TensorArray(tf.float32, 0, True)
        prog = tf.keras.utils.Progbar(self.variance.nt, verbose=verbose)
        for d in data:
            adat_p = tf.matmul(d, footprint, False, True)
            for p in adat_p:
                adat = adat.write(adat.size(), p)
                prog.add(1)
        self.variance._cache(0, footprint, tf.transpose(adat.stack()))


class SpikeCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        stage = self.stage
        writer = self._get_writer('spike')
        with writer.as_default():
            summary_stat(self.model.spike.val, stage, step=epoch)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)

        stage = self.stage
        writer = self._get_writer('spike')
        with writer.as_default():
            summary_spike(self.model.spike.val, stage, step=0)
