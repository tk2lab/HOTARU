import enum

import tensorflow as tf

from .dynamics import CalciumToSpikeDoubleExp
from .dynamics import SpikeToCalciumDoubleExp


class VarianceMode(enum.Enum):
    Spike = 1
    Footprint = 2


class Variance(tf.keras.layers.Layer):
    """Variance"""

    def __init__(self, mode, data, nk, nx, nt, **args):
        super().__init__(**args)

        self.mode = mode
        self.data = data
        self.nx = nx
        self.nt = nt

        if self.mode == VarianceMode.Spike:
            nz = nt
        elif self.mode == VarianceMode.Footprint:
            nz = nx
        self.nk = nk
        self._nn = self.add_weight("nn", (), trainable=False)
        self._nm = self.add_weight("nm", (), trainable=False)
        self._dat = self.add_weight("dat", (nk, nz), trainable=False)
        self._cov = self.add_weight("cov", (nk, nk), trainable=False)
        self._out = self.add_weight("out", (nk, nk), trainable=False)

    def set_double_exp(self, *args, **kwargs):
        self.spike_to_calcium = SpikeToCalciumDoubleExp(
            *args, **kwargs, name="to_cal"
        )
        self.calcium_to_spike = CalciumToSpikeDoubleExp(
            *args, **kwargs, name="to_spk"
        )
        self.nu = self.nt + self.calcium_to_spike.pad

    def set_baseline(self, bx, bt):
        nxf = tf.convert_to_tensor(self.nx, tf.float32)
        ntf = tf.convert_to_tensor(self.nt, tf.float32)
        nn = nxf * ntf
        nm = nn
        if bx > 0.0:
            nm += nxf
        if bt > 0.0:
            nm += ntf
        self.bx = bx
        self.bt = bt
        self._nn.assign(nn)
        self._nm.assign(nm)

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
        return (self._nn + variance) / self._nm

    def _cache(self, yval, dat):
        if self.mode == VarianceMode.Spike:
            ny, bx, by = self.nx, self.bt, self.bx
        elif self.mode == VarianceMode.Footprint:
            ny, bx, by = self.nt, self.bx, self.bt

        ycov = tf.matmul(yval, yval, False, True)
        ysum = tf.math.reduce_sum(yval, axis=1)
        yout = ysum[:, None] * ysum / ny
        cx = 1.0 - tf.math.square(bx)
        cy = 1.0 - tf.math.square(by)
        dat = -2.0 * dat
        cov = ycov - cx * yout
        out = yout - cy * ycov

        nk = tf.shape(dat)[0]
        self._dat[:nk].assign(dat)
        self._cov[:nk, :nk].assign(cov)
        self._out[:nk, :nk].assign(out)

        lipschitz = tf.math.reduce_max(tf.linalg.eigvalsh(cov)) / self._nm
        if self.mode == VarianceMode.Spike:
            gsum = tf.math.reduce_sum(self.spike_to_calcium.kernel)
            return (lipschitz * gsum).numpy()
        elif self.mode == VarianceMode.Footprint:
            return lipschitz.numpy()
