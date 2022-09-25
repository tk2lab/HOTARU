import tensorflow as tf


class HotaruConfigMixin:
    """Model"""

    def set_early_stop(self, *args, **kwargs):
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping("score", *args, **kwargs),
        ]
