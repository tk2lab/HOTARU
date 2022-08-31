import tensorflow as tf

from .regularizer import MaxNormNonNegativeL1
from .regularizer import ProxOp


class DynamicInputLayer(tf.keras.layers.Layer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nx, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._nk = self.add_weight("nk", (), tf.int32, trainable=False)
        self._val = self.add_weight("val", (nk, nx))
        self._val.regularizer = regularizer or ProxOp(name="prox")
        self.built = True

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

    def penalty(self):
        return self._val.regularizer(self.val)


class MaxNormNonNegativeL1InputLayer(DynamicInputLayer):
    """Max Norm NonNegative L1 Input Layer"""

    def __init__(self, nk, nx, axis=-1, **kwargs):
        reg = MaxNormNonNegativeL1(axis=axis, name="prox")
        super().__init__(nk, nx, reg, **kwargs)

    def get_l(self):
        return self._val.regularizer.get_l()

    def set_l(self, val):
        self._val.regularizer.set_l(val)
