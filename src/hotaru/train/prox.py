import tensorflow as tf


def get_prox(var):
    if hasattr(var, "_distributed_container"):
        var = var._distributed_container()
    if hasattr(var, "regularizer") and hasattr(var.regularizer, "prox"):
        return var.regularizer.prox
    else:
        return lambda x, _: x


def get_penalty(var):
    if hasattr(var, "regularizer"):
        return var.regularizer(var)
    else:
        return 0.0


class ProxOp(tf.keras.layers.Layer):
    """ProxOp"""

    def call(self, x):
        return x

    def prox(self, y, eta):
        return y


class NonNegativeL1(ProxOp):
    """NonNegativeL1"""

    def __init__(self, **args):
        super().__init__(**args)
        self._l = self.add_weight("l", (), trainable=False)
        self.built = True

    @property
    def l(self):
        return self._l

    def get_l(self):
        self.l.numpy()

    def set_l(self, val):
        self._l.assign(val)

    def call(self, x):
        return self.l * tf.math.reduce_sum(tf.nn.relu(x))

    def prox(self, y, eta):
        return tf.nn.relu(y - eta * self.l)


class MaxNormNonNegativeL1(ProxOp):
    """MaxNormNonNegativeL1"""

    def __init__(self, axis=-1, **args):
        super().__init__(**args)
        self._l = self.add_weight("l", (), trainable=False)
        self.axis = axis
        self.built = True

    @property
    def l(self):
        return self._l

    def get_l(self):
        self.l.numpy()

    def set_l(self, val):
        self._l.assign(val)

    def call(self, x):
        x = tf.nn.relu(x)
        s = tf.math.reduce_sum(x, axis=self.axis)
        m = tf.math.reduce_max(x, axis=self.axis)
        cond = m > 0.0
        s = tf.boolean_mask(s, cond)
        m = tf.boolean_mask(m, cond)
        return self.l * tf.math.reduce_sum(s / m)

    def prox(self, y, eta):
        y = tf.nn.relu(y)
        m = tf.math.reduce_max(y, axis=self.axis, keepdims=True)
        return tf.nn.relu(tf.where(tf.equal(y, m), y, y - eta * self.l / m))
