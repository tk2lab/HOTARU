import math

import tensorflow as tf


def calc_cov_out(xval):
    nx = tf.cast(tf.shape(xval)[1], tf.float32)
    xcov = tf.linalg.matmul(xval, xval, False, True)
    xsum = tf.math.reduce_sum(xval, axis=1)
    xout = xsum[:, None] * (xsum / nx)
    return xval, xcov, xout


def prepare(yval, cor, bx, by):
    yval, ycov, yout = calc_cov_out(yval)
    cx = 1 - tf.math.square(bx)
    cy = 1 - tf.math.square(by)
    dat = -2 * cor
    cov = ycov - cx * yout
    out = yout - cy * ycov
    return dat, cov, out


class CacheLayer(tf.keras.layers.Layer):
    """Dynamic Input Layer"""

    def __init__(self, nk, nx, bx, by, **kwargs):
        super().__init__(**kwargs)
        self._nk = self.add_weight("nk", (), tf.int32, trainable=False)
        self._dat = self.add_weight(
            "dat", (nk, nx), tf.float32, trainable=False
        )
        self._cov = self.add_weight(
            "cov", (nk, nk), tf.float32, trainable=False
        )
        self._out = self.add_weight(
            "out", (nk, nk), tf.float32, trainable=False
        )
        self._bx = bx
        self._by = by

    def prepare(self, val, cor):
        dat, cov, out = prepare(val, cor, self._bx, self._by)
        nk = tf.shape(dat)[0]
        self._nk.assign(nk)
        self._dat[:nk, :].assign(dat)
        self._cov[:nk, :nk].assign(cov)
        self._out[:nk, :nk].assign(out)

    def call(self, inputs):
        nk = self._nk
        dat = self._dat[:nk, :]
        cov = self._cov[:nk, :nk]
        out = self._out[:nk, :nk]
        return dat, cov, out


class OutputLayer(tf.keras.layers.Layer):
    """Output Statistics"""

    def call(self, val):
        return calc_cov_out(val)


class LossLayer(tf.keras.layers.Layer):
    """"""

    def __init__(self, nx, nt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nn = nx * nt
        self._nm = nx * nt + nx + nt

    def call(self, inputs):
        x, y = inputs
        xval, xcov, xout = x
        yval, ycov, yout = y
        variance = (
            self._nn
            + tf.math.reduce_sum(xval * yval)
            + tf.math.reduce_sum(xcov * ycov)
            + tf.math.reduce_sum(xout * yout)
        ) / self._nm
        return (tf.math.log(variance) + math.log(math.pi) + 1) / 2


class IdentityLoss(tf.keras.losses.Loss):
    """"""

    def call(self, y_true, y_pred):
        return y_pred
