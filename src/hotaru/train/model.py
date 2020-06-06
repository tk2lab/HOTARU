import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from ..optimizer.prox_nesterov import ProxNesterov
from ..optimizer.regularizer import MaxNormNonNegativeL1
from .input import InputLayer
from .variance import VarianceLayer


class HotaruModel(tf.keras.Model):

    def __init__(self, data, nk, nx, nt, tau1, tau2, hz=None, tscale=6.0,
                 la=0.0, lu=0.0, bx=0.0, bt=0.0, batch=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        variance = VarianceLayer(data, tau1, tau2, hz, tscale, bx, bt, batch)
        footprint_regularizer = MaxNormNonNegativeL1(la / nx / nt, 1)
        spike_regularizer = MaxNormNonNegativeL1(lu / nx / nt, 1)
        nu = nt + variance.calcium_to_spike.pad

        self.footprint = InputLayer(nk, nx, footprint_regularizer)
        self.spike = InputLayer(nk, nu, spike_regularizer)
        self.variance = variance

    def update_spike(self, *args, **kwargs):
        nk = self.footprint.val.shape[0]
        nu = self.spike.val.shape[1]
        self.spike.val =  np.zeros((nk, nu), np.float32)
        self.fit(self.variance.SPIKE_MODE, *args, **kwargs)
        scale = self.spike.val.max(axis=1)
        self.spike.val = self.spike.val[scale > 0.1]

    def update_footprint(self, *args, **kwargs):
        self.spike.val = self.spike.val / self.spike.val.max(axis=1, keepdims=True)
        nk = self.spike.val.shape[0]
        nx = self.footprint.val.shape[1]
        self.footprint.val = np.zeros((nk, nx), np.float32)
        self.fit(self.variance.FOOTPRINT_MODE, *args, **kwargs)
        scale = self.footprint.val.max(axis=1)
        cond = scale > 0.1
        self.spike.val = self.spike.val[cond] * scale[cond, None]
        self.footprint.val = self.footprint.val[cond] / scale[cond, None]

    def select(self, ids):
        self.footprint.val = self.footprint.val[ids]
        self.spike.val = self.spike.val[ids]

    def call(self, mode):
        _dummy = tf.zeros((1, 1))
        footprint, footprint_penalty = self.footprint(_dummy)
        spike, spike_penalty = self.spike(_dummy)
        variance = self.variance((mode, footprint, spike))
        sigma = K.sqrt(variance)
        ll = K.log(sigma)
        me = ll + footprint_penalty + spike_penalty
        return tf.stack([ll, me, sigma, footprint_penalty, spike_penalty])

    def compile(self, *args, **kwargs):
        super().compile(
            optimizer=ProxNesterov(),
            loss=HotaruLoss(),
            metrics=[HotaruMetric(pos) for pos in [1, 2, 3, 4]],
            *args, **kwargs,
        )

    def fit(self, mode, steps_per_epoch=100, *args, **kwargs):
        super().fit(
            _gen_data(mode),
            steps_per_epoch=steps_per_epoch,
            *args, **kwargs,
        )


def _gen_data(mode):
    while True:
        yield mode * tf.ones((1, 1), tf.int32), tf.zeros((5,))


class HotaruLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return y_pred[0]


class HotaruMetric(tf.python.keras.metrics.MeanMetricWrapper):

    def __init__(self, pos, dtype=None):
        names = ['ll', 'me', 'sigma', 'pa', 'pu']
        super().__init__(hotaru_metric(pos), name=names[pos], dtype=dtype)
        self._pos = pos


def hotaru_metric(pos):
    return lambda y_true, y_pred: y_pred[pos]
