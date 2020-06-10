import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

#from ..optimizer.prox_optimizer import ProxOptimizer as Optimizer
from ..optimizer.prox_nesterov import ProxNesterov as Optimizer
from ..optimizer.callback import Callback
from ..optimizer.regularizer import MaxNormNonNegativeL1
from .input import InputLayer
from .extract import Extract
from .variance import Variance


class HotaruModel(tf.keras.Model):

    def __init__(self, data, mask, nk, nx, nt, tau1, tau2, hz, tscale,
                 la, lu, bx, bt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        footprint_regularizer = MaxNormNonNegativeL1(la / nx / nt, 1)
        spike_regularizer = MaxNormNonNegativeL1(lu / nx / nt, 1)
        variance = Variance(data, nk, nx, nt, tau1, tau2, hz, tscale, bx, bt)
        nu = variance.nu
        self.extract = Extract(nx, nu)

        self.mask = mask
        self.footprint = InputLayer(nk, nx, footprint_regularizer, 'footprint')
        self.spike = InputLayer(nk, nu, spike_regularizer, 'spike')
        self.variance = variance
        self.status_shape = nk, nx, nu

    def update_spike(self, batch, lr=0.01, *args, **kwargs):
        nk = self.footprint.val.shape[0]
        nu = self.spike.val.shape[1]
        self.spike.val = np.zeros((nk, nu), np.float32)

        K.set_value(self.extract.nk, nk)
        self.variance.start_spike_mode(self.footprint.val, batch)

        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz
        self.fit(*args, **kwargs)

        scale = self.spike.val.max(axis=1)
        self.spike.val = self.spike.val[scale > 0.1]

    def update_footprint(self, batch, lr=0.01, *args, **kwargs):
        nk = self.spike.val.shape[0]
        nx = self.footprint.val.shape[1]
        scale = self.spike.val.max(axis=1)
        self.spike.val = self.spike.val / scale[:, None]
        self.footprint.val = np.zeros((nk, nx), np.float32)

        K.set_value(self.extract.nk, nk)
        self.variance.start_footprint_mode(self.spike.val, batch)

        self.optimizer.learning_rate = lr * 2.0 / self.variance.lipschitz
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

        return out

    def compile(self, *args, **kwargs):
        super().compile(
            optimizer=Optimizer(),
            loss=HotaruLoss(self.extract, self.variance),
            *args, **kwargs,
        )

    def fit(self, steps_per_epoch=100, epochs=10, min_delta=1e-5,
            log_dir=None, callbacks=None, *args, **kwargs):
        def _gen_data():
            while True:
                yield x, y

        if callbacks is None:
            callbacks = []
        callbacks += [
            Callback(),
            tf.keras.callbacks.EarlyStopping('score', min_delta=min_delta),
        ]
        if log_dir is not None:
            callbacks += [
                HotaruCallback(log_dir, update_freq='epoch'),
            ]

        nk, nx, nu = self.status_shape
        x = tf.zeros((1, 1))
        y = tf.zeros((nk, nx + nu))
        super().fit(
            _gen_data(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            *args, **kwargs,
        )


def hotaru_loss(variance):
    return 0.5 * K.log(variance)


class HotaruLoss(tf.keras.losses.Loss):

    def __init__(self, extract, variance, name='Variance'):
        super().__init__(name=name)
        self._extract = extract
        self._variance = variance

    def call(self, y_true, y_pred):
        footprint, spike = self._extract(y_pred)
        variance = self._variance((footprint, spike))
        return hotaru_loss(variance)


class HotaruCallback(tf.keras.callbacks.TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        mode = K.get_value(self.model.variance._mode)
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            if mode == 0:
                m = self.model.spike.val.max(axis=1)
                tf.summary.histogram('spike/magnitude', m, step=epoch)
            else:
                m = self.model.footprint.val.max(axis=1)
                tf.summary.histogram('footprint/magnitude', m, step=epoch)
                footprint = self.model.footprint.val
                footprint /= footprint.max(axis=1, keepdims=True)
                mask = self.model.mask
                nk, nx = footprint.shape
                h, w = mask.shape
                imgs = np.zeros((nk, h, w, 1))
                imgs[:, mask, 0] = footprint
                tf.summary.image('footprint/max', imgs.max(axis=0, keepdims=True), step=epoch)
                tf.summary.image('footprint', imgs, step=epoch)
