import tensorflow as tf

from hotaru.train.dynamics import SpikeToCalciumDefault
from hotaru.train.dynamics import CalciumToSpikeDefault


class VarianceLayer(tf.keras.layers.Layer):

    SPIKE_MODE = 1
    FOOTPRINT_MODE = 2

    def __init__(self, data, tau1, tau2, hz=None, tscale=6.0,
                 bx=0.0, bt=0.0, batch=100, *args, **kwargs):
        super().__init__(dynamic=True, *args, **kwargs)
        self.data = data
        self.spike_to_calcium = SpikeToCalciumDefault(
            tau1, tau2, hz, tscale, name='to_cal'
        )
        self.calcium_to_spike = CalciumToSpikeDefault(
            tau1, tau2, hz, tscale, name='to_spk'
        )
        self.bx = bx
        self.bt = bt
        self.batch = batch

    def build(self, input_shape):
        _, (nk, nx), (_, nu) = input_shape
        nt = nu - self.calcium_to_spike.pad

        nxf = tf.cast(nx, tf.float32)
        ntf = tf.cast(nt, tf.float32)
        nn = ntf * nxf
        nm = nn
        if self.bx > 0.0:
            nm += nxf
        if self.bt > 0.0:
            nm += ntf

        self.nt = nt
        self.scale = nn / nm
        self.params = [
            [ntf, nxf, nn, nm, self.bt, self.bx],
            [nxf, ntf, nn, nm, self.bx, self.bt]
        ]
        self.mode = self.add_weight('mode', (), tf.int32, trainable=False)
        self.cache = [[
            self.add_weight('adat', (nk, nt), trainable=False),
            self.add_weight('acov', (nk, nk), trainable=False),
            self.add_weight('aout', (nk, nk), trainable=False),
        ], [
            self.add_weight('vdat', (nk, nx), trainable=False),
            self.add_weight('vcov', (nk, nk), trainable=False),
            self.add_weight('vout', (nk, nk), trainable=False),
        ]]
        super().build(input_shape)

    def call(self, inputs):
        mode, footprint, spike = inputs

        cache_op = tf.cond(
            mode == self.mode,
            tf.no_op,
            lambda: tf.cond(
                mode == 1,
                lambda: self._cache_footprint(footprint),
                lambda: self._cache_spike(spike),
            ),
        )

        with tf.control_dependencies([cache_op]):
            (ydat, ycov, yout), xdat = tf.cond(
                mode == 1,
                lambda: (self.cache[0], self.spike_to_calcium(spike)),
                lambda: (self.cache[1], footprint),
            )
            nk, nx = tf.shape(xdat)
            xcov = tf.matmul(xdat, xdat, False, True)
            xsum = tf.reduce_sum(xdat, axis=1)
            xout = xsum[:, None] * xsum
            ydat = tf.slice(ydat, [0, 0], [nk, nx])
            ycov = tf.slice(ycov, [0, 0], [nk, nk])
            yout = tf.slice(ycov, [0, 0], [nk, nk])
            variance = (
                self.scale
                + tf.reduce_sum(ydat * xdat)
                + tf.reduce_sum(ycov * xcov)
                + tf.reduce_sum(yout * xout)
            )
        return variance

    def _cache_footprint(self, footprint): 
        adat = tf.TensorArray(tf.float32, 0, True)
        data = self.data.batch(self.batch)
        prog = tf.keras.utils.Progbar(self.nt)
        for d in data:
            adat_p = tf.matmul(footprint, d, False, True)
            for a in tf.transpose(adat_p):
                adat = adat.write(adat.size(), a)
                prog.add(1)
        adat = tf.transpose(adat.stack())
        return tf.group(self._cache(1, footprint, adat))

    def _cache_spike(self, spike): 
        calcium = self.spike_to_calcium(spike)
        nk = tf.shape(calcium)[0]
        data = self.data.batch(self.batch)
        vdat = tf.constant(0.0)
        prog = tf.keras.utils.Progbar(self.nt)
        e = tf.constant(0)
        for d in data:
            n = tf.shape(d)[0]
            s, e = e, e + n
            c_p = tf.slice(calcium, [0, s], [nk, n])
            vdat += tf.matmul(c_p, d)
            prog.add(n.numpy())
        return tf.group(self._cache(2, calcium, vdat))

    def _cache(self, mode, yval, dat):
        dat_v, cov_v, out_v = self.cache[mode - 1]
        nx, ny, nn, nm, bx, by = self.params[mode - 1]
        ycov = tf.matmul(yval, yval, False, True)
        xcov = 1
        ysum = tf.reduce_sum(yval, axis=1)
        yout = ysum[:, None] * ysum / ny
        xout = 1 / nx
        cx = 1.0 + tf.square(bx)
        cy = 1.0 + tf.square(by)
        dat = dat / nm
        cov = (xcov * ycov - cx * xcov * yout) / nm
        out = (xout * yout - cy * xout * ycov) / nm
        dk = tf.shape(dat_v)[0] - tf.shape(dat)[0]
        dat = tf.pad(dat, [[0, dk], [0, 0]]) 
        cov = tf.pad(cov, [[0, dk], [0, dk]]) 
        out = tf.pad(out, [[0, dk], [0, dk]]) 
        return [
            self.mode.assign(mode),
            dat_v.assign(dat),
            cov_v.assign(cov),
            out_v.assign(out),
        ]
