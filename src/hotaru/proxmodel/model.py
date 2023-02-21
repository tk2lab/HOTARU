import tensorflow as tf

from ..util.progress import ProgbarLogger
from .optimizer import ProxOptimizer as Optimizer


def is_split_variable(x):
    return hasattr(x, "_variable_list") or hasattr(x, "_variables")


class ProxModel(tf.keras.Model):
    """"""

    @staticmethod
    def add_regularizer_prox(weight, regularizer):
        weight.regularizer = regularizer
        if is_split_variable(weight):
            for v in weight:
                v.prox = prox
        else:
            weight.prox = prox

    @property
    def penalty(self):
        out = tf.cast(0, self.dtype)
        for w in self.weights:
            if hasattr(w, "penalty"):
                out += w.penalty()
        return out

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metrics = super().compute_metrics(x, y, y_pred, sample_weight)
        penalty = self.penalty
        metrics["loss"] += penalty
        metrics["penalty"] = penalty
        return metrics

    def compile(
        self,
        learning_rate=0.01,
        nesterov_scale=20.0,
        reset_interval=100,
        **kwargs,
    ):
        optimizer = Optimizer(learning_rate, nesterov_scale, reset_interval)
        super().compile(optimizer=optimizer, **kwargs)

    def fit(self, *args, epochs=1, verbose=1, **kwargs):
        steps = self.optimizer.reset_interval
        callbacks = tf.keras.callbacks.CallbackList(
            kwargs.pop("callbacks", []) + [ProgbarLogger()],
            # kwargs.pop("callbacks", []),
            add_history=True,
            add_progbar=False,
            model=self,
            epochs=epochs,
            steps=steps,
            verbose=verbose,
        )
        callbacks._check_timing = False
        return super().fit(
            *args,
            epochs=epochs,
            steps_per_epoch=steps,
            callbacks=callbacks,
            **kwargs,
        )
