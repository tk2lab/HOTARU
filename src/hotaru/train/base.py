import tensorflow as tf

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

    def prepare_layers(self, nk, nx, nt, hz, tau1, tau2, tscale):
        spike_to_calcium = SpikeToCalciumDoubleExp(
            hz, tau1, tau2, tscale, name="to_cal"
        )
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

    def compile(
        self,
        learning_rate=1e-2,
        nesterov_scale=20.0,
        reset_interval=100,
        **kwargs
    ):
        super().compile(
            optimizer=Optimizer(reset_interval=reset_interval),
            loss=Loss(),
            **kwargs,
        )
        self._lr = learning_rate
        self._reset = reset_interval

    def fit(self, epochs=100, min_delta=1e-4, patience=3, **kwargs):
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
