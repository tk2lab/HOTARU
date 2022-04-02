import tensorflow.keras.backend as K
import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed

from .dynamics import SpikeToCalcium
from .dynamics import CalciumToSpike


class Variance(tf.keras.layers.Layer):

    def __init__(self, data, nk, nx, nt, batch=100, name='Variance'):
        super().__init__(name=name, dtype=tf.float32)

        self._data = data
        self.nx = nx
        self.nt = nt
        self._batch = batch

        self.spike_to_calcium = SpikeToCalcium(name='to_cal')
        self.calcium_to_spike = CalciumToSpike(name='to_spk')
        self._bx = 0.0
        self._bt = 0.0

        nz = max(nx, nt)
        self.nk = nk
        self._nn = self.add_weight('nn', (), trainable=False)
        self._nm = self.add_weight('nm', (), trainable=False)
        self._dat = self.add_weight('dat', (nk, nz), trainable=False)
        self._cov = self.add_weight('cov', (nk, nk), trainable=False)
        self._out = self.add_weight('out', (nk, nk), trainable=False)

    def set_double_exp(self, *args, **kwargs):
        self.spike_to_calcium.set_double_exp(*args, **kwargs)
        self.calcium_to_spike.set_double_exp(*args, **kwargs)
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
        self._bx = bx
        self._bt = bt
        K.set_value(self._nn, nn)
        K.set_value(self._nm, nm)

    def call(self, xdat):

        '''
        @distributed(ReduceOp.CONCAT)
        def _matmul(dat, xdat):
            print(tf.shape(dat), tf.shape(xdat))
            return tf.linalg.matmul(dat, xdat, False, True),
        '''

        nk, nx = tf.shape(xdat)[0], tf.shape(xdat)[1]
        data = tf.data.Dataset.from_tensor_slices(xdat).batch(self._batch)

        #xcov, = _matmul(data, xdat)
        xcov = tf.linalg.matmul(xdat, xdat, False, True)
        xsum = tf.math.reduce_sum(xdat, axis=1)
        xout = xsum[:, None] * xsum / tf.cast(nx, tf.float32)

        ydat = tf.slice(self._dat, [0, 0], [nk, nx])
        ycov = tf.slice(self._cov, [0, 0], [nk, nk])
        yout = tf.slice(self._out, [0, 0], [nk, nk])

        variance = (
              tf.math.reduce_sum(ydat * xdat)
            + tf.math.reduce_sum(ycov * xcov)
            + tf.math.reduce_sum(yout * xout)
        )
        return (self._nn + variance) / self._nm

    def _cache(self, mode, yval, dat):
        if mode == 0:
            ny, bx, by = self.nx, self._bt, self._bx
        else:
            ny, bx, by = self.nt, self._bx, self._bt
        max_nk, nmax = tf.shape(self._dat)

        ycov = tf.matmul(yval, yval, False, True)
        ysum = tf.math.reduce_sum(yval, axis=1)
        yout = ysum[:, None] * ysum / ny
        cx = 1.0 - tf.math.square(bx)
        cy = 1.0 - tf.math.square(by)
        dat = -2.0 * dat
        cov = ycov - cx * yout
        out = yout - cy * ycov

        lipschitz = tf.math.reduce_max(tf.linalg.eigvalsh(cov)) / self._nm
        gsum = tf.math.reduce_sum(self.spike_to_calcium.kernel)
        self.lipschitz_a = lipschitz.numpy()
        self.lipschitz_u = (lipschitz * gsum).numpy()

        nk, nx = tf.shape(dat)[0], tf.shape(dat)[1]
        dk = max_nk - nk
        dx = nmax - nx
        dat = tf.pad(dat, [[0, dk], [0, dx]])
        cov = tf.pad(cov, [[0, dk], [0, dk]])
        out = tf.pad(out, [[0, dk], [0, dk]])

        K.set_value(self._dat, dat)
        K.set_value(self._cov, cov)
        K.set_value(self._out, out)
