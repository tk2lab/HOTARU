import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

#from ..optimizer.prox_optimizer import ProxOptimizer as Optimizer
from ..optimizer.prox_nesterov import ProxNesterov as Optimizer
from ..optimizer.callback import Callback as OptCallback
from .variance import Variance
from .callback import SpikeCallback, FootprintCallback


class BaseModel(tf.keras.Model):

    def compile(self, **kwargs):
        super().compile(
            optimizer=Optimizer(), loss=Loss(), **kwargs,
        )

    def fit(self, callback, log_dir=None, stage=None, callbacks=None,
            steps_per_epoch=100, epochs=100, min_delta=1e-3, **kwargs):
        if callbacks is None:
            callbacks = []

        callbacks += [
            OptCallback(),
            tf.keras.callbacks.EarlyStopping(
                'score', min_delta=min_delta, patience=3,
                restore_best_weights=True, verbose=0,
            ),
        ]

        if log_dir is not None:
            callbacks += [
                callback(
                    log_dir=log_dir, stage=stage,
                    update_freq='batch', write_graph=False,
                ),
            ]

        dummy = tf.zeros((1, 1))
        data = tf.data.Dataset.from_tensor_slices((dummy, dummy)).repeat()
        super().fit(
            data,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs,
        )


class FootprintModel(BaseModel):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(**kwargs)
        self.footprint = footprint
        self.spike_penalty = spike.penalty
        self.spike_val = lambda: K.get_value(spike._val)
        self.variance = variance

    def call(self, inputs):
        footprint = self.footprint(inputs)
        variance = self.variance(footprint)
        loss = 0.5 * K.log(variance)
        footprint_penalty = self.footprint.penalty()
        spike_penalty = self.spike_penalty()
        me = loss + footprint_penalty + spike_penalty
        self.add_metric(me, 'mean', 'score')
        self.add_metric(footprint_penalty, 'mean', 'penalty')
        return loss

    def fit(self, lr, batch, **kwargs):
        spike = self.spike_val()
        scale0 = spike.max(axis=1)
        spike /= scale0[:, None]

        self.variance.start_footprint_mode(spike, batch)
        nk = self.spike_val().shape[0]
        nx = self.variance.nx
        self.footprint.val = np.zeros((nk, nx))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_a
        super().fit(FootprintCallback, **kwargs)

        footprint = self.footprint.val
        scale = footprint.max(axis=1)
        self.footprint.val = footprint / scale[:, None]
        return scale / scale0


class SpikeModel(BaseModel):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(**kwargs)
        self.footprint_penalty = footprint.penalty
        self.footprint_val = lambda: K.get_value(footprint._val)
        self.spike = spike
        self.variance = variance

    def call(self, inputs):
        spike = self.spike(inputs)
        calcium = self.variance.spike_to_calcium(spike)
        variance = self.variance(calcium)
        loss = 0.5 * K.log(variance)
        footprint_penalty = self.footprint_penalty()
        spike_penalty = self.spike.penalty()
        me = loss + footprint_penalty + spike_penalty
        self.add_metric(me, 'mean', 'score')
        self.add_metric(spike_penalty, 'mean', 'penalty')
        return loss

    def fit(self, lr, batch, **kwargs):
        self.variance.start_spike_mode(self.footprint_val(), batch)
        nk = self.footprint_val().shape[0]
        nu = self.variance.nu
        self.spike.val = np.zeros((nk, nu))
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_u
        super().fit(SpikeCallback, **kwargs)


class Loss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return y_pred
