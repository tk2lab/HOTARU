import tensorflow as tf


class ProxOp(tf.keras.regularizers.Regularizer):
    """Identity ProxOp"""

    def __call__(self, x):
        return x

    def prox(self, y, eta):
        return y


class L2(ProxOp):
    """L2 regularizer"""

    def __init__(self, l):
        self.l = l if tf.is_tensor(l) else tf.convert_to_tensor(l)

    def __call__(self, x):
        return self.l * tf.math.reduce_sum(tf.math.square(x))

    def prox(self, y, eta):
        return y / (1 + 2 * eta * self.l)


class NonNegativeL1(ProxOp):
    """NonNegativeL1"""

    def __init__(self, l):
        self.l = l if tf.is_tensor(l) else tf.convert_to_tensor(l)

    def __call__(self, x):
        return self.l * tf.math.reduce_sum(tf.nn.relu(x))

    def prox(self, y, eta):
        return tf.nn.relu(y - eta * self.l)


class MaxNormNonNegativeL1(ProxOp):
    """MaxNormNonNegativeL1"""

    def __init__(self, l, axis=-1):
        self.l = l if tf.is_tensor(l) else tf.convert_to_tensor(l)
        self.axis = axis

    def __call__(self, x):
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
