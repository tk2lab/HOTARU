import tensorflow as tf

from ..optimizer.prox_optimizer import ProxOptimizer as Optimizer
from .input import MaxNormNonNegativeL1InputLayer as Input
from .variance import Variance


class Loss(tf.keras.losses.Loss):
    """Loss"""

    def call(self, y_true, y_pred):
        return y_pred


class BaseModel(tf.keras.Model):
    """Base Model"""

    def prepare_layers(self, mode, data, nk, nx, nt, tau, bx, bt, la, lu):
        variance = Variance(mode, data, nk, nx, nt)
        variance.set_double_exp(**tau)
        variance.set_baseline(bx, bt)
        footprint = Input(nk, nx, name="footprint")
        spike = Input(nk, variance.nu, name="spike")
        footprint.set_l(la / variance._nm)
        spike.set_l(lu / variance._nm)
        loss = tf.keras.layers.Lambda(lambda x: 0.5 * tf.math.log(x))
        return footprint, spike, variance, loss

    def set_metric(self, footprint, spike, variance, loss):
        footprint_penalty = footprint.penalty()
        spike_penalty = spike.penalty()
        me = loss + footprint_penalty + spike_penalty
        self.add_metric(tf.math.sqrt(variance), "sigma")
        self.add_metric(me, "score")
        self.add_metric(me - loss, "penalty")

    def compile(self, reset=100, lr=1.0, **kwargs):
        super().compile(
            optimizer=Optimizer(reset=reset),
            loss=Loss(),
            **kwargs,
        )
        self._lr = lr
        self._reset = reset

    def fit(self, epochs=100, min_delta=1e-3, patience=3, **kwargs):
        self.optimizer.learning_rate = self._lr * (2 / self.lipschitz)
        callbacks = kwargs.setdefault("callbacks", [])
        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                "score",
                min_delta=min_delta,
                patience=patience,
                restore_best_weights=True,
            ),
        ]
        return super().fit(
            self.dummy_data(),
            steps_per_epoch=self._reset,
            epochs=epochs,
            **kwargs,
        )

    def dummy_data(self):
        dummy = tf.zeros((1, 1))
        return tf.data.Dataset.from_tensor_slices((dummy, dummy)).repeat()
