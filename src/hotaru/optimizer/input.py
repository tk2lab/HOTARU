import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from .regularizer import ProxOp, MaxNormNonNegativeL1


class DynamicInputLayer(tf.keras.layers.Layer):

    def __init__(self, nk, nx, regularizer=None, name='input', **kwargs):
        super().__init__(name=f'{name}_layer', **kwargs)
        regularizer = regularizer or ProxOp()
        self.max_nk = nk
        self._nk = self.add_weight(f'{name}/nk', (), tf.int32, trainable=False)
        self._val = self.add_weight(f'{name}/val', (nk, nx))
        self._val.regularizer = regularizer

    @property
    def l(self):
        return K.get_value(self._val.regularizer.l)

    @l.setter
    def l(self, val):
        K.set_value(self._val.regularizer.l, val)

    @property
    def val(self):
        return K.get_value(self.call())

    @val.setter
    def val(self, val):
        nk = val.shape[0]
        val = np.pad(val, ((0, self.max_nk - nk), (0, 0)))
        K.set_value(self._nk, nk)
        K.set_value(self._val, val)

    def penalty(self, x=None):
        if x is None:
            x = self.call()
        return self._val.regularizer(x)

    def call(self, dummy=None):
        nk = self._nk
        nx = tf.shape(self._val)[1]
        return tf.slice(self._val, (0, 0), (nk, nx))


class MaxNormNonNegativeL1InputLayer(DynamicInputLayer):

    def __init__(self, nk, nx, name='mnnnl1', **kwargs):
        reg = MaxNormNonNegativeL1(1)
        super().__init__(nk, nx, reg, name=name, **kwargs)
