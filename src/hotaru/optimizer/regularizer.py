import tensorflow.keras.backend as K
import tensorflow as tf


def get_prox(var):
    if hasattr(var, 'regularizer') and hasattr(var.regularizer, 'prox'):
        return var.regularizer.prox
    else:
        return lambda x, _: x


def get_penalty(var):
    if hasattr(var, 'regularizer'):
        return var.regularizer(var)
    else:
        return 0.0


class ProxOp(tf.keras.layers.Layer):

    def call(self, x):
        return x

    def prox(self, y, eta):
        return y


class NonNegativeL1(ProxOp):

    def __init__(self, l):
        super().__init__()
        self._l = l

    def call(self, x):
        return self._l * K.sum(K.relu(x))

    def prox(self, y, eta):
        return K.relu(y - eta * self._l)


class MaxNormNonNegativeL1(ProxOp):

    def __init__(self, l, axis=-1):
        super().__init__()
        self._l = l
        self._axis = axis

    def call(self, x):
        x = K.relu(x)
        s = K.sum(x, axis=self._axis)
        m = K.max(x, axis=self._axis)
        cond = m > 0.0
        s = tf.boolean_mask(s, cond)
        m = tf.boolean_mask(m, cond)
        return self._l * K.sum(s / m)

    def prox(self, y, eta):
        y = K.relu(y)
        m = K.max(y, axis=self._axis, keepdims=True)
        return K.relu(tf.where(K.equal(y, m), y, y - eta * self._l / m))
