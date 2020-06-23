import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

#from ..optimizer.prox_optimizer import ProxOptimizer as Optimizer
from ..optimizer.prox_nesterov import ProxNesterov as Optimizer
from ..optimizer.callback import Callback as OptCallback
from .variance import Variance


class BaseModel(tf.keras.Model):

    def __init__(self, footprint, spike, variance, **kwargs):
        super().__init__(**kwargs)
        self.footprint_penalty = footprint.penalty
        self.spike_penalty = spike.penalty
        self.la_set = footprint.set_l
        self.lu_set = spike.set_l
        self.variance = variance

    def set_penalty(self, la, lu, bx, bt):
        self.variance.set_baseline(bx, bt)
        nm = K.get_value(self.variance._nm)
        self.la_set(la / nm)
        self.lu_set(lu / nm)

    def compile(self, **kwargs):
        super().compile(
            optimizer=Optimizer(), loss=Loss(), **kwargs,
        )

    def call_common(self, val):
        variance = self.variance(val)
        loss = 0.5 * K.log(variance)
        footprint_penalty = self.footprint_penalty()
        spike_penalty = self.spike_penalty()
        me = loss + footprint_penalty + spike_penalty
        self.add_metric(K.sqrt(variance), 'mean', 'sigma')
        self.add_metric(me, 'mean', 'score')
        return loss, footprint_penalty, spike_penalty

    def fit_common(self, callback, log_dir=None, stage=None, callbacks=None,
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


class Loss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return y_pred
