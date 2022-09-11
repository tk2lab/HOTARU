import tensorflow as tf


class ProxOp(tf.keras.layers.Layer):
    """Identity ProxOp"""

    def call(self, x):
        return x

    def prox(self, y, eta):
        return y


class L2(ProxOp):
    """L2 regularizer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._l = self.add_weight("l", (), trainable=False)

    @property
    def l(self):
        return self._l

    def get_l(self):
        self.l.numpy()

    def set_l(self, val):
        self._l.assign(val)

    def call(self, x):
        return self.l * tf.math.reduce_sum(tf.math.square(x))

    def prox(self, y, eta):
        return y / (1 + 2 * eta * self.l)


class NonNegativeL1(ProxOp):
    """NonNegativeL1"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._l = self.add_weight("l", (), trainable=False)

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

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self._l = self.add_weight("l", (), trainable=False)
        self._axis = axis

    @property
    def l(self):
        return self._l

    def get_l(self):
        self.l.numpy()

    def set_l(self, val):
        self._l.assign(val)

    def call(self, x):
        x = tf.nn.relu(x)
        s = tf.math.reduce_sum(x, axis=self._axis)
        m = tf.math.reduce_max(x, axis=self._axis)
        cond = m > 0.0
        s = tf.boolean_mask(s, cond)
        m = tf.boolean_mask(m, cond)
        return self.l * tf.math.reduce_sum(s / m)

    def prox(self, y, eta):
        y = tf.nn.relu(y)
        m = tf.math.reduce_max(y, axis=self._axis, keepdims=True)
        return tf.nn.relu(tf.where(tf.equal(y, m), y, y - eta * self.l / m))
