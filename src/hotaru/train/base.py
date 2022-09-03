import tensorflow as tf

from .dynamics import CalciumToSpikeDoubleExp
from .dynamics import SpikeToCalciumDoubleExp
from .input import DynamicInputLayer
from .optimizer import ProxOptimizer as Optimizer
from .prox import MaxNormNonNegativeL1


class Loss(tf.keras.losses.Loss):
    """Loss"""

    def call(self, y_true, y_pred):
        return y_pred


class BaseModel(tf.keras.Model):
    """Base Model"""

    def prepare_layers(self, nk, nx, nt, tau):
        spike_to_calcium = SpikeToCalciumDoubleExp(**tau, name="to_cal")
        nu = nt + spike_to_calcium.pad

        dummy = tf.keras.Input(type_spec=tf.TensorSpec((), tf.float32))

        footprint_prox = MaxNormNonNegativeL1(axis=-1, name="footprint_prox")
        footprint = DynamicInputLayer(nk, nx, footprint_prox, name="footprint")

        spike_prox = MaxNormNonNegativeL1(axis=-1, name="spike_prox")
        spike = DynamicInputLayer(nk, nu, spike_prox, name="spike")

        self.nx = nx
        self.nt = nt
        self.nu = nu
        return spike_to_calcium, dummy, footprint, spike

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
        return tf.data.Dataset.from_tensors((dummy, dummy)).repeat()
