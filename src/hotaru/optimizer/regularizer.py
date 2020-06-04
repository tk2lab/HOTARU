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
        return self.l * tf.reduce_sum(tf.nn.relu(x))

    def prox(self, y, eta):
        return tf.nn.relu(y - eta * self.l)

    def get_config(self):
        return dict(l=self.l)


class MaxNormNonNegativeL1(ProxOp):

    def __init__(self, l=None, axis=-1):
        super().__init__()
        self.l = l
        self.axis = axis

    def __call__(self, x):
        x = tf.nn.relu(x)
        s = tf.reduce_sum(x, axis=self.axis)
        m = tf.reduce_max(x, axis=self.axis)
        cond = m > 0.0
        s = tf.boolean_mask(s, cond)
        m = tf.boolean_mask(m, cond)
        return self.l * tf.reduce_sum(s / m)

    def prox(self, y, eta):
        y = tf.nn.relu(y)
        m = tf.reduce_max(y, axis=self.axis, keepdims=True)
        return tf.nn.relu(tf.where(tf.equal(y, m), y, y - eta * self.l / m))

    def get_config(self):
        return dict(l=self.l, axis=self.axis)
