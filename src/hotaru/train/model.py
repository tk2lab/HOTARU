import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

#from ..optimizer.prox_optimizer import ProxOptimizer as Optimizer
from ..optimizer.prox_nesterov import ProxNesterov as Optimizer
from ..optimizer.callback import Callback
from ..optimizer.input import MaxNormNonNegativeL1InputLayer
from .extract import Extract
from .variance import Variance
from .loss import hotaru_loss, HotaruLoss
from .callback import HotaruCallback


class HotaruModel(tf.keras.Model):

    def __init__(self, data, mask, nk, nx, nt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_shape = nk, nx, nt
        self.mask = mask
        self.variance = Variance(data, nk, nx, nt)
        self.la = 0.0
        self.lu = 0.0

    def set_double_exp(self, *args, **kwargs):
        self.variance.set_double_exp(*args, **kwargs)
        nu = self.variance.nu
        nk, nx, nt = self.status_shape
        if not hasattr(self, 'extract'):
            self.extract = Extract(nx, nu)
            self.footprint = MaxNormNonNegativeL1InputLayer(nk, nx, name='footprint')
            self.spike = MaxNormNonNegativeL1InputLayer(nk, nu, name='spike')

    def update_spike(self, batch, lr, *args, **kwargs):
        nk = self.footprint.val.shape[0]
        nu = self.spike.val.shape[1]
        self.spike.val = np.zeros((nk, nu), np.float32)

        self.variance.start_spike_mode(self.footprint.val, batch)
        nm = K.get_value(self.variance._nm)
        K.set_value(self.footprint._val.regularizer.l, self.la / nm)
        K.set_value(self.spike._val.regularizer.l, self.lu / nm)
        K.set_value(self.extract.nk, nk)
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_u
        self.fit(*args, **kwargs)

    def update_footprint(self, batch, lr, *args, **kwargs):
        spike = self.spike.val
        nk = spike.shape[0]
        nx = self.footprint.val.shape[1]
        scale = spike.max(axis=1)
        self.spike.val = spike / scale[:, None]
        self.footprint.val = np.zeros((nk, nx), np.float32)

        self.variance.start_footprint_mode(self.spike.val, batch)
        nm = K.get_value(self.variance._nm)
        K.set_value(self.footprint._val.regularizer.l, self.la / nm)
        K.set_value(self.spike._val.regularizer.l, self.lu / nm)
        K.set_value(self.extract.nk, nk)
        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz_a
        self.fit(*args, **kwargs)

        footprint = self.footprint.val
        scale = footprint.max(axis=1)
        self.spike.val = self.spike.val * scale[:, None]
        self.footprint.val = footprint / scale[:, None]

    def call(self, inputs):
        footprint = self.footprint(inputs)
        spike = self.spike(inputs)
        out = tf.concat((footprint, spike), axis=1)
        self.calc_metrics(out)
        return out

    def calc_metrics(self, out):
        footprint, spike = self.extract(out)
        variance = self.variance((footprint, spike))
        footprint_penalty = self.footprint.penalty(footprint)
        spike_penalty = self.spike.penalty(spike)
        me = hotaru_loss(variance) + footprint_penalty + spike_penalty
        penalty = tf.cond(
            self.variance._mode == 0,
            lambda: spike_penalty,
            lambda: footprint_penalty,
        )
        self.add_metric(me, 'mean', 'score')
        self.add_metric(penalty, 'mean', 'penalty')

    def compile(self, *args, **kwargs):
        super().compile(
            optimizer=Optimizer(),
            loss=HotaruLoss(self.extract, self.variance),
            *args, **kwargs,
        )

    def fit(self, steps_per_epoch=100, epochs=100, min_delta=1e-3,
            log_dir=None, stage='', callbacks=None, *args, **kwargs):
        def _gen_data():
            while True:
                yield x, y

        if callbacks is None:
            callbacks = []

        callbacks += [
            Callback(),
            tf.keras.callbacks.EarlyStopping(
                'score', min_delta=min_delta, patience=3,
                restore_best_weights=True, verbose=0,
            ),
        ]

        if log_dir is not None:
            callbacks += [
                HotaruCallback(log_dir=log_dir, stage=stage, update_freq='batch'),
            ]

        nk, nx, nt = self.status_shape
        nu = self.variance.nu
        x = tf.zeros((1, 1))
        y = tf.zeros((nk, nx + nu))
        super().fit(
            _gen_data(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            *args, **kwargs,
        )
