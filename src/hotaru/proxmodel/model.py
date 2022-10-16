import tensorflow as tf

from ..util.progress import ProgressCallback
from .optimizer import ProxOptimizer


class ProxModel(tf.keras.Model):

    def compile(self, *args, **kwargs):
        kwargs.setdefault("optimizer", ProxOptimizer())
        super().compile(*args, **kwargs)

    def fit(self, data, epochs, verbose=1, **kwargs):
        steps = self.optimizer.reset_interval.numpy()
        callbacks = tf.keras.callbacks.CallbackList(
            kwargs.pop("callbacks", []) + [ProgressCallback()],
            add_histroy=True,
            add_progbar=False,
            model=self,
            epochs=epochs,
            steps=steps,
            verbose=verbose,
        )
        return super().fit(
            data,
            epochs=epochs,
            steps_per_epoch=steps,
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
