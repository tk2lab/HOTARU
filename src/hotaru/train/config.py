import tensorflow as tf


class HotaruConfigMixin:
    """Model"""

    def set_early_stop(self, *args, **kwargs):
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping("score", *args, **kwargs),
        ]

    def set_optimizer(self, *args, **kwargs):
        self.spatial.optimizer.set(*args, **kwargs)
        self.temporal.optimizer.set(*args, **kwargs)
