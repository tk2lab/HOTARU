import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from ..optimizer.prox_nesterov import ProxNesterov
from ..optimizer.regularizer import MaxNormNonNegativeL1
from .input import InputLayer
from .variance import VarianceLoss


class HotaruModel(tf.keras.Model):

    def __init__(self, data, nk, nx, nt, tau1, tau2, hz, tscale,
                 la, lu, bx, bt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        variance = VarianceLoss(data, nk, nx, nt, tau1, tau2, hz, tscale, bx, bt)
        footprint_regularizer = MaxNormNonNegativeL1(la / nx / nt, 1)
        spike_regularizer = MaxNormNonNegativeL1(lu / nx / nt, 1)
        nu = variance.nu

        self.footprint = InputLayer(nk, nx, footprint_regularizer)
        self.spike = InputLayer(nk, nu, spike_regularizer)
        self.variance = variance
        self.status_shape = nk, nx, nu

    def update_spike(self, batch, *args, **kwargs):
        self.variance.start_spike_mode(self.footprint.val, batch)

        nk = self.footprint.val.shape[0]
        nu = self.spike.val.shape[1]
        self.spike.val = np.zeros((nk, nu), np.float32)

        self.fit(*args, **kwargs)

        scale = self.spike.val.max(axis=1)
        self.spike.val = self.spike.val[scale > 0.1]

    def update_footprint(self, batch, *args, **kwargs):
        self.variance.start_footprint_mode(self.spike.val, batch)

        nk = self.spike.val.shape[0]
        nx = self.footprint.val.shape[1]
        scale = self.spike.val.max(axis=1)
        self.spike.val = self.spike.val / scale[:, None]
        self.footprint.val = np.zeros((nk, nx), np.float32)

        self.fit(*args, **kwargs)

        scale = self.footprint.val.max(axis=1)
        cond = scale > 0.1
        self.spike.val = self.spike.val[cond] * scale[cond, None]
        self.footprint.val = self.footprint.val[cond] / scale[cond, None]

    def select(self, ids):
        self.footprint.val = self.footprint.val[ids]
        self.spike.val = self.spike.val[ids]

    def call(self, mode):
        _dummy = tf.zeros((1, 1))
        footprint = self.footprint(_dummy)
        spike = self.spike(_dummy)
        return tf.concat((footprint, spike), axis=1)

    def compile(self, *args, **kwargs):
        super().compile(
            optimizer=ProxNesterov(),
            loss=self.variance,
            #metrics=[HotaruMetric(pos) for pos in [1, 2, 3, 4]],
            *args, **kwargs,
        )

    def fit(self, steps_per_epoch=100, *args, **kwargs):
        def _gen_data():
            while True:
                yield x, y

        nk, nx, nu = self.status_shape
        x = tf.zeros((1, 1))
        y = tf.zeros((nk, nx + nu))
        super().fit(
            _gen_data(),
            steps_per_epoch=steps_per_epoch,
            *args, **kwargs,
        )


class HotaruMetric(tf.python.keras.metrics.MeanMetricWrapper):

    def __init__(self, pos, dtype=None):
        names = ['ll', 'me', 'sigma', 'pa', 'pu']
        super().__init__(hotaru_metric(pos), name=names[pos], dtype=dtype)
        self._pos = pos


def hotaru_metric(pos):
    return lambda y_true, y_pred: y_pred[pos]
