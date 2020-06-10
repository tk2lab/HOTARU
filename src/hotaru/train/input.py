import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from ..optimizer.regularizer import ProxOp


class InputLayer(tf.keras.layers.Layer):

    def __init__(self, nk, nx, regularizer=None, name='Input', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        regularizer = regularizer or ProxOp()
        self.max_nk = nk
        self._nk = self.add_weight('nk', (), tf.int32, trainable=False)
        self._val = self.add_weight('val', (nk, nx))
        self._val.regularizer = regularizer

    @property
    def val(self):
        return K.get_value(self.val_tensor())

    @val.setter
    def val(self, val):
        nk = val.shape[0]
        val = np.pad(val, ((0, self.max_nk - nk), (0, 0)))
        K.set_value(self._nk, nk)
        K.set_value(self._val, val)

    def val_tensor(self):
        nk = self._nk
        nx = tf.shape(self._val)[1]
        return tf.slice(self._val, (0, 0), (nk, nx))

    def penalty(self, x=None):
        if x is None:
            x = self.val_tensor()
        return self._val.regularizer(x)

    def call(self, dummy):
        return self._val
