import tensorflow.keras.backend as K
import tensorflow as tf

from hotaru.train.dynamics import SpikeToCalcium, CalciumToSpike


class Variance(tf.keras.layers.Layer):

    def __init__(self, data, nk, nx, nt, name='Variance'):
        super().__init__(name=name, dtype=tf.float32)

        self.nk = nk
        self.nx = nx
        self.nt = nt

        self.spike_to_calcium = SpikeToCalcium(name='to_cal')
        self.calcium_to_spike = CalciumToSpike(name='to_spk')
        self._bx = 0.0
        self._bt = 0.0

        nz = max(nx, nt)
        self._data = data
        self._nn = self.add_weight('nn', (), trainable=False)
        self._nm = self.add_weight('nm', (), trainable=False)
        self._dat = self.add_weight('dat', (nk, nz), trainable=False)
        self._cov = self.add_weight('cov', (nk, nk), trainable=False)
        self._out = self.add_weight('out', (nk, nk), trainable=False)

    def set_double_exp(self, *tau):
        self.spike_to_calcium.set_double_exp(*tau)
        self.calcium_to_spike.set_double_exp(*tau)
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

    def start_spike_mode(self, footprint, batch):
        data = self._data.batch(batch)
        footprint = tf.convert_to_tensor(footprint)

        prog = tf.keras.utils.Progbar(self.nt)
        adat = tf.TensorArray(tf.float32, self.nt)
        i = tf.constant(0)
        for d in data:
            adat_p = tf.matmul(footprint, d, False, True)
            for a in tf.transpose(adat_p):
                adat = adat.write(i, a)
                i += 1
                prog.add(1)
        adat = tf.transpose(adat.stack())
        self._cache(0, footprint, adat)

    def start_footprint_mode(self, spike, batch):
        data = self._data.batch(batch)
        spike = tf.convert_to_tensor(spike)
        calcium = self.spike_to_calcium(spike)
        nk = tf.shape(calcium)[0]

        prog = tf.keras.utils.Progbar(self.nt)
        vdat = tf.constant(0.0)
        e = tf.constant(0)
        for d in data:
            n = tf.shape(d)[0]
            s, e = e, e + n
            c_p = tf.slice(calcium, [0, s], [nk, n])
            vdat += tf.matmul(c_p, d)
            prog.add(n.numpy())
        self._cache(1, calcium, vdat)

    def call(self, xdat):
        nk, nx = tf.shape(xdat)[0], tf.shape(xdat)[1]
        xcov = tf.matmul(xdat, xdat, False, True)
        xsum = K.sum(xdat, axis=1)
        xout = xsum[:, None] * xsum / tf.cast(nx, tf.float32)

        ydat = tf.slice(self._dat, [0, 0], [nk, nx])
        ycov = tf.slice(self._cov, [0, 0], [nk, nk])
        yout = tf.slice(self._out, [0, 0], [nk, nk])

        variance = (
            self._nn
            + K.sum(ydat * xdat)
            + K.sum(ycov * xcov)
            + K.sum(yout * xout)
        ) / self._nm

        self.add_metric(K.sqrt(variance), 'mean', 'sigma')
        return variance

    def _cache(self, mode, yval, dat):
        if mode == 0:
            ny, bx, by = self.nx, self._bt, self._bx
        else:
            ny, bx, by = self.nt, self._bx, self._bt
        max_nk, nmax = tf.shape(self._dat)

        ycov = tf.matmul(yval, yval, False, True)
        ysum = K.sum(yval, axis=1)
        yout = ysum[:, None] * ysum / ny
        cx = 1.0 - K.square(bx)
        cy = 1.0 - K.square(by)
        dat = -2.0 * dat
        cov = ycov - cx * yout
        out = yout - cy * ycov

        lipschitz = K.max(tf.linalg.eigvalsh(cov)) / self._nm
        gsum = K.sum(self.spike_to_calcium.kernel)
        self.lipschitz_a = lipschitz.numpy()
        self.lipschitz_u = (lipschitz * gsum).numpy()

        nk, nx = tf.shape(dat)
        dk = max_nk - nk
        dx = nmax - nx
        dat = tf.pad(dat, [[0, dk], [0, dx]])
        cov = tf.pad(cov, [[0, dk], [0, dk]])
        out = tf.pad(out, [[0, dk], [0, dk]])

        K.set_value(self._dat, dat)
        K.set_value(self._cov, cov)
        K.set_value(self._out, out)
