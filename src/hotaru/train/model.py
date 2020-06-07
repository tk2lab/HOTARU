import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from ..optimizer.prox_nesterov import ProxNesterov
from ..optimizer.regularizer import MaxNormNonNegativeL1
from .input import InputLayer
from .variance import Extract, Variance


class HotaruModel(tf.keras.Model):

    def __init__(self, data, nk, nx, nt, tau1, tau2, hz, tscale,
                 la, lu, bx, bt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        variance = Variance(data, nk, nx, nt, tau1, tau2, hz, tscale, bx, bt)
        footprint_regularizer = MaxNormNonNegativeL1(la / nx / nt, 1)
        spike_regularizer = MaxNormNonNegativeL1(lu / nx / nt, 1)
        nu = variance.nu

        self.footprint = InputLayer(nk, nx, footprint_regularizer, 'footprint')
        self.spike = InputLayer(nk, nu, spike_regularizer, 'spike')
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

    def call(self, inputs):
        # output
        footprint = self.footprint(inputs)
        spike = self.spike(inputs)
        out = tf.concat((footprint, spike), axis=1)

        # metrics
        footprint, spike = self.variance.extract(out)
        variance = self.variance((footprint, spike))
        footprint_penalty = self.footprint.penalty(footprint)
        spike_penalty = self.spike.penalty(spike)
        me = hotaru_loss(variance) + footprint_penalty + spike_penalty
        self.add_metric(me, 'mean', 'score')

        return out

    def compile(self, *args, **kwargs):
        super().compile(
            optimizer=ProxNesterov(),
            loss=HotaruLoss(self.variance),
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


def hotaru_loss(variance):
    return 0.5 * K.log(variance)


class HotaruLoss(tf.keras.losses.Loss):

    def __init__(self, variance, name='Variance'):
        super().__init__(name=name)
        self._variance = variance

    def call(self, y_true, y_pred):
        footprint, spike = self._variance.extract(y_pred)
        variance = self._variance((footprint, spike))
        return hotaru_loss(variance)
