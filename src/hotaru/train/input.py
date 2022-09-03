import tensorflow as tf

from .prox import ProxOp


class DynamicInputLayer(tf.keras.layers.Layer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nx, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._nk = self.add_weight("nk", (), tf.int32, trainable=False)
        self._val = self.add_weight("val", (nk, nx))
        self._val.regularizer = regularizer

    @property
    def val(self):
        return self._val[: self._nk]

    def get_val(self):
        return self.val.numpy()

    def set_val(self, val):
        nk = val.shape[0]
        self._nk.assign(nk)
        self._val[:nk].assign(val)

    def call(self, dummy=None):
        return self.val

    def penalty(self, x):
        return self._val.regularizer(x)
