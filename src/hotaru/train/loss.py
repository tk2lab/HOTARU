import tensorflow as tf


class LossLayer(tf.keras.layers.Layer):
    """Log Standard Deviation"""

    def __init__(self, nk, n0, n1, **args):
        super().__init__(**args)

        self._n1 = n1
        self._nn = n0 * n1
        self._nm = n0 * n1 + n0 + n1

        self._b0 = self.add_weight("b0", (), trainable=False)
        self._b1 = self.add_weight("b1", (), trainable=False)
        self._dat = self.add_weight("dat", (nk, n0), trainable=False)
        self._cov = self.add_weight("cov", (nk, nk), trainable=False)
        self._out = self.add_weight("out", (nk, nk), trainable=False)

    def set_background_penalty(self, b0, b1):
        self._b0.assign(b0)
        self._b1.assign(b1)

    def call(self, xdat):
        nk, nx = tf.shape(xdat)[0], tf.shape(xdat)[1]
        xcov = tf.linalg.matmul(xdat, xdat, False, True)
        xsum = tf.math.reduce_sum(xdat, axis=1)
        xout = xsum[:, None] * xsum / tf.cast(nx, tf.float32)

        ydat = self._dat[:nk]
        ycov = self._cov[:nk, :nk]
        yout = self._out[:nk, :nk]

        variance = (
            tf.math.reduce_sum(ydat * xdat)
            + tf.math.reduce_sum(ycov * xcov)
            + tf.math.reduce_sum(yout * xout)
        )
        sigma = tf.math.sqrt((self._nn + variance) / self._nm)
        self.add_metric(sigma, "sigma")
        return tf.math.log(sigma)

    @tf.function
    def _cache(self, yval, dat):
        ycov = tf.matmul(yval, yval, False, True)
        ysum = tf.math.reduce_sum(yval, axis=1)
        yout = ysum[:, None] * ysum / self._n1
        cx = 1.0 - tf.math.square(self._b0)
        cy = 1.0 - tf.math.square(self._b1)
        dat = -2.0 * dat
        cov = ycov - cx * yout
        out = yout - cy * ycov

        nk = tf.shape(dat)[0]
        self._dat[:nk].assign(dat)
        self._cov[:nk, :nk].assign(cov)
        self._out[:nk, :nk].assign(out)


class HotaruLoss(tf.keras.losses.Loss):
    """Loss"""

    def call(self, y_true, y_pred):
        return y_pred
