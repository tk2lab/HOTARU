import tensorflow as tf
import tensorflow.keras.backend as K


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


class ProxOp(tf.keras.regularizers.Regularizer):

    def prox(self, y, eta):
        return y

    def get_config(self):
        return dict()


class NonNegativeL1(ProxOp):

    def __init__(self, l=None):
        super().__init__()
        self.l = l

    def __call__(self, x):
        return self.l * K.sum(K.relu(x))

    def prox(self, y, eta):
        return K.relu(y - eta * self.l)

    def get_config(self):
        return dict(l=self.l)


class MaxNormNonNegativeL1(ProxOp):

    def __init__(self, l=None, axis=-1):
        super().__init__()
        self.l = l
        self.axis = axis

    def __call__(self, x):
        x = K.relu(x)
        s = K.sum(x, axis=self.axis)
        m = K.max(x, axis=self.axis)
        cond = m > 0.0
        s = tf.boolean_mask(s, cond)
        m = tf.boolean_mask(m, cond)
        return self.l * K.sum(s / m)

    def prox(self, y, eta):
        y = K.relu(y)
        m = K.max(y, axis=self.axis, keepdims=True)
        return K.relu(tf.where(K.equal(y, m), y, y - eta * self.l / m))

    def get_config(self):
        return dict(l=self.l, axis=self.axis)
