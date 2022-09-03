import tensorflow as tf

from .input import DynamicInputLayer
from .optimizer import ProxOptimizer as Optimizer
from .prox import MaxNormNonNegativeL1
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

        footprint_prox = MaxNormNonNegativeL1(axis=-1, name="footprint_prox")
        footprint_prox.set_l(la / variance._nm)
        footprint = DynamicInputLayer(nk, nx, footprint_prox, name="footprint")

        spike_prox = MaxNormNonNegativeL1(axis=-1, name="spike_prox")
        spike_prox.set_l(lu / variance._nm)
        spike = DynamicInputLayer(nk, variance.nu, spike_prox, name="spike")

        loss = tf.keras.layers.Lambda(lambda x: 0.5 * tf.math.log(x))
        return footprint, spike, variance, loss

    def set_metric(self, penalty, variance, loss):
        score = loss + penalty
        self.add_metric(tf.math.sqrt(variance), "sigma")
        self.add_metric(score, "score")
        self.add_metric(penalty, "penalty")

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
