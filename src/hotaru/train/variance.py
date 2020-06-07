import tensorflow as tf

from hotaru.train.dynamics import SpikeToCalciumDefault
from hotaru.train.dynamics import CalciumToSpikeDefault


class VarianceLoss(tf.keras.losses.Loss):

    def __init__(self, data, nk, nx, nt, tau1, tau2, hz, tscale, bx, bt,
                 name='Variance', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.data = data
        self.spike_to_calcium = SpikeToCalciumDefault(
            tau1, tau2, hz, tscale, name='to_cal'
        )
        self.calcium_to_spike = CalciumToSpikeDefault(
            tau1, tau2, hz, tscale, name='to_spk'
        )
        nu = nt + self.calcium_to_spike.pad
        nmax = max(nx, nu)

        nxf = tf.cast(nx, tf.float32)
        ntf = tf.cast(nt, tf.float32)
        nn = ntf * nxf
        nm = nn
        if bx > 0.0:
            nm += nxf
        if bt > 0.0:
            nm += ntf

        self.params = [
            [nk, nmax, ntf, nxf, nn, nm, bt, bx],
            [nk, nmax, nxf, ntf, nn, nm, bx, bt]
        ]
        self.nx = nx
        self.nt = nt
        self.nu = nu

        self.mode = tf.Variable(tf.constant(0, tf.int32), False)
        self.nk = tf.Variable(tf.constant(nk, tf.int32), False)
        self.scale = nn / nm
        self.dat = tf.Variable(tf.zeros((nk, nmax), tf.float32), False)
        self.cov = tf.Variable(tf.zeros((nk, nk), tf.float32), False)
        self.out = tf.Variable(tf.zeros((nk, nk), tf.float32), False)

    def start_spike_mode(self, footprint, batch): 
        data = self.data.batch(batch)
        footprint = tf.convert_to_tensor(footprint)

        prog = tf.keras.utils.Progbar(self.nt)
        adat = tf.TensorArray(tf.float32, 0, True)
        for d in data:
            adat_p = tf.matmul(footprint, d, False, True)
            for a in tf.transpose(adat_p):
                adat = adat.write(adat.size(), a)
                prog.add(1)
        adat = tf.transpose(adat.stack())
        self._cache(1, footprint, adat)

    def start_footprint_mode(self, spike, batch): 
        data = self.data.batch(batch)
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
        self._cache(2, calcium, vdat)

    def call(self, y_true, y_pred):
        nk, nx, nt, nu = self.nk, self.nx, self.nt, self.nu
        inputs = tf.slice(y_pred, [0, 0], [nk, nx + nu])
        footprint, spike = tf.split(inputs, [nx, nu], axis=1)

        nz, xdat = tf.cond(
            self.mode == 1,
            lambda: (nt, self.spike_to_calcium(spike)),
            lambda: (nx, footprint),
        )
        xcov = tf.matmul(xdat, xdat, False, True)
        xsum = tf.reduce_sum(xdat, axis=1)
        xout = xsum[:, None] * xsum

        ydat = tf.slice(self.dat, [0, 0], [nk, nz])
        ycov = tf.slice(self.cov, [0, 0], [nk, nk])
        yout = tf.slice(self.cov, [0, 0], [nk, nk])

        variance = (
            self.scale
            + tf.reduce_sum(ydat * xdat)
            + tf.reduce_sum(ycov * xcov)
            + tf.reduce_sum(yout * xout)
        )
        return variance

    def _cache(self, mode, yval, dat):
        max_nk, nmax, nx, ny, nn, nm, bx, by = self.params[mode - 1]
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

        nk, nx = tf.shape(dat)
        dk = max_nk - nk
        dx = nmax - nx
        dat = tf.pad(dat, [[0, dk], [0, dx]]) 
        cov = tf.pad(cov, [[0, dk], [0, dk]]) 
        out = tf.pad(out, [[0, dk], [0, dk]]) 

        self.mode.assign(mode)
        self.nk.assign(nk)
        self.dat.assign(dat)
        self.cov.assign(cov)
        self.out.assign(out)
