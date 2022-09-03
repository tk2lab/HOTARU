import enum

import tensorflow as tf


class LossLayer(tf.keras.layers.Layer):
    """Variance"""

    def __init__(self, nk, n0, n1, b0, b1, **args):
        super().__init__(**args)

        nn = n0 * n1
        nm = nn
        if b0 > 0.0:
            nm += n0
        if b1 > 0.0:
            nm += n1
        self._b0 = b0
        self._b1 = b1
        self._n1 = n1
        self._nn = nn
        self._nm = nm

        self._dat = self.add_weight("dat", (nk, n0), trainable=False)
        self._cov = self.add_weight("cov", (nk, nk), trainable=False)
        self._out = self.add_weight("out", (nk, nk), trainable=False)

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
        variance = (self._nn + variance) / self._nm
        self.add_metric(tf.math.sqrt(variance), "sigma")
        return tf.math.log(variance) / 2

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

        lipschitz = tf.math.reduce_max(tf.linalg.eigvalsh(cov)) / self._nm
        return lipschitz
