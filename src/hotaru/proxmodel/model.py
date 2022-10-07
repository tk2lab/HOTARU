import tensorflow as tf

from ..util.progress import ProgressCallback
from .optimizer import ProxOptimizer


class ProxModel(tf.keras.Model):

    def compile(self, *args, **kwargs):
        super().compile(*args, optimizer=ProxOptimizer(), **kwargs)

    def fit(self, data, epochs, **kwargs):
        callbacks = kwargs.pop("callbacks", [])
        callbacks.append(ProgressCallback(self.name, epochs))
        callbacks = tf.keras.callbacks.CallbackList(callbacks, add_histroy=True)
        return super().fit(
            data,
            epochs=epochs,
            steps_per_epoch=self.optimizer.reset_interval.numpy(),
            callbacks=callbacks,
            **kwargs,
        )

    def compute_loss(self, x, y, y_pred, sample_weight):
        loss = self.compiled_loss(y, y_pred, sample_weight)
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metrics = super().compute_metrics(x, y, y_pred, sample_weight)
        metrics["penalty"] = tf.math.reduce_sum(self.losses)
        metrics["loss"] += metrics["penalty"]
        return metrics
