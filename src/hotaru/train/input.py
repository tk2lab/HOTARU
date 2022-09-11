import tensorflow as tf


class DynamicInputLayer(tf.keras.layers.Layer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nx, **kwargs):
        super().__init__(**kwargs)
        self._nk = self.add_weight("nk", (), tf.int32, trainable=False)
        self._val = self.add_weight("val", (nk, nx))

    def set_regularizer(self, regularizer):
        self._val.regularizer = regularizer

    def set_val(self, val):
        nk = val.shape[0]
        self._nk.assign(nk)
        self._val[:nk].assign(val)

    def clear(self, nk):
        self._nk.assign(nk)
        self._val[:nk].assign(tf.zeros_like(self._val[:nk]))

    @property
    def val(self):
        return self._val[: self._nk]

    def get_num(self):
        return self._nk.numpy()

    def get_val(self):
        return self.val.numpy()

    def call(self, dummy=None):
        return self.val

    def penalty(self, val):
        return self._val.regularizer(val)


class TemporalBackground(DynamicInputLayer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nt, **kwargs):
        super().__init__(nk, nt - 1, **kwargs)

    def set_val(self, val):
        nk = val.shape[0]
        diff = val[:, 1:] - val[:, :-1]
        self._nk.assign(nk)
        self._val[:nk].assign(diff)

    @property
    def val(self):
        return tf.pad(
            tf.math.cumsum(self._val[: self._nk], axis=1), [[0, 0], [1, 0]]
        )

    def call(self, dummy=None):
        return self.val
