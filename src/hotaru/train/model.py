import tensorflow as tf

from ..optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from ..optimizer.prox_optimizer import ProxOptimizer as Optimizer
from .variance import Variance


class Loss(tf.keras.losses.Loss):
    """Loss"""

    def call(self, y_true, y_pred):
        return y_pred


class BaseModel(tf.keras.Model):
    """Base Model"""

    def prepare(self, mode, data, nk, nx, nt, tau, bx, bt, la, lu):
        variance = Variance(mode, data, nk, nx, nt)
        variance.set_double_exp(**tau)
        variance.set_baseline(bx, bt)
        footprint = Input(nk, nx, name="footprint")
        spike = Input(nk, variance.nu, name="spike")
        footprint.set_l(la / variance._nm)
        spike.set_l(lu / variance._nm)
        self.footprint_penalty = footprint.penalty
        self.spike_penalty = spike.penalty
        self.variance = variance
        return footprint, spike, variance

    def compile(self, reset=100, lr=1.0, **kwargs):
        super().compile(
            optimizer=Optimizer(reset=reset),
            loss=Loss(),
            **kwargs,
        )
        self._lr = lr
        self._reset = reset

    def call_common(self, val):
        variance = self.variance(val)
        loss = 0.5 * tf.math.log(variance)
        footprint_penalty = self.footprint_penalty()
        spike_penalty = self.spike_penalty()
        me = loss + footprint_penalty + spike_penalty
        self.add_metric(tf.math.sqrt(variance), "sigma")
        self.add_metric(me, "score")
        return loss, footprint_penalty, spike_penalty

    def fit(self, val, batch, epochs=100,
            min_delta=1e-3, patience=3, **kwargs):
        callbacks = kwargs.setdefault("callbacks", [])
        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                "score",
                min_delta=min_delta,
                patience=patience,
                restore_best_weights=True,
            ),
        ]
        lipschitz = self.prepare_fit(val, batch)
        self.optimizer.learning_rate = self._lr * (2 / lipschitz)
        return super().fit(
            self.dummy_data(),
            epochs=epochs,
            steps_per_epoch=self._reset,
            **kwargs,
        )

    def dummy_data(self):
        dummy = tf.zeros((1, 1))
        return tf.data.Dataset.from_tensor_slices((dummy, dummy)).repeat()
