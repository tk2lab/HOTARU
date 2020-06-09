import tensorflow as tf
import tensorflow.keras.backend as K

from hotaru.train.dynamics import SpikeToCalciumDefault
from hotaru.train.dynamics import CalciumToSpikeDefault


class Variance(tf.keras.layers.Layer):

    def __init__(self, data, nk, nx, nt,
                 tau1, tau2, hz, tscale, bx, bt, name='Variance'):
        super().__init__(name=name, dtype=tf.float32)

        spike_to_calcium = SpikeToCalciumDefault(
            tau1, tau2, hz, tscale, name='to_cal',
        )
        calcium_to_spike = CalciumToSpikeDefault(
            tau1, tau2, hz, tscale, name='to_spk',
        )
        nmax = max(nx, nt)
        nu = nt + calcium_to_spike.pad

        nxf = tf.convert_to_tensor(nx, tf.float32)
        ntf = tf.convert_to_tensor(nt, tf.float32)
        nn = nxf * ntf
        nm = nn
        if bx > 0.0:
            nm += nxf
        if bt > 0.0:
            nm += ntf

        self.nx = nx
        self.nt = nt
        self.nu = nu

        self.spike_to_calcium = spike_to_calcium
        self.calcium_to_spike = calcium_to_spike

        self._params = [ [nx, bt, bx], [nt, bx, bt] ]
        self._data = data
        self._nn = nn
        self._nm = nm
        self._mode = self.add_weight('mode', (), tf.int32, trainable=False)
        self._dat = self.add_weight('dat', (nk, nmax), tf.float32, trainable=False)
        self._cov = self.add_weight('cov', (nk, nk), tf.float32, trainable=False)
        self._out = self.add_weight('out', (nk, nk), tf.float32, trainable=False)

    def start_spike_mode(self, footprint, batch): 
        data = self._data.batch(batch)
        footprint = tf.convert_to_tensor(footprint)

        prog = tf.keras.utils.Progbar(self.nt)
        adat = tf.TensorArray(tf.float32, 0, True)
        for d in data:
            adat_p = tf.matmul(footprint, d, False, True)
            for a in tf.transpose(adat_p):
                adat = adat.write(adat.size(), a)
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

    def call(self, inputs):
        footprint, spike = inputs

        nk = tf.shape(footprint)[0]
        nx, xdat = tf.cond(
            self._mode == 0,
            lambda: (self.nt, self.spike_to_calcium(spike)),
            lambda: (self.nx, footprint),
        )
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
        max_nk, nmax = tf.shape(self._dat)
        ny, bx, by = self._params[mode]

        ycov = tf.matmul(yval, yval, False, True)
        ysum = K.sum(yval, axis=1)
        yout = ysum[:, None] * ysum / ny
        cx = 1.0 - K.square(bx)
        cy = 1.0 - K.square(by)
        dat = -2.0 * dat
        cov = ycov - cx * yout
        out = yout - cy * ycov

        lipschitz = K.max(tf.linalg.eigvalsh(cov)) / self._nm
        self.lipschitz = lipschitz.numpy()

        nk, nx = tf.shape(dat)
        dk = max_nk - nk
        dx = nmax - nx
        dat = tf.pad(dat, [[0, dk], [0, dx]]) 
        cov = tf.pad(cov, [[0, dk], [0, dk]]) 
        out = tf.pad(out, [[0, dk], [0, dk]]) 

        K.set_value(self._mode, mode)
        K.set_value(self._dat, dat)
        K.set_value(self._cov, cov)
        K.set_value(self._out, out)
