import tensorflow as tf
import tensorflow.keras.backend as K


class SpikeToCalciumBase(tf.keras.layers.Layer):

    def call(self, u):
        return K.conv1d(
            u[..., None], self.kernel[::-1, None, None],
            1, 'valid', 'channels_last',
        )[..., 0]


class SpikeToCalciumDefault(SpikeToCalciumBase):

    def __init__(self, tau1, tau2, hz=None, tscale=6.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tau1 < tau2:
            tau1, tau2 = tau1, tau2
        if hz is not None:
            tau1 *= hz
            tau2 *= hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = K.pow(r, -tau2 / d) - K.pow(r, -tau1 / d)
        t = K.arange(1.0, tau1 * tscale + 2.0, dtype=tf.float32)
        e1 = K.exp(-t / tau1)
        e2 = K.exp(-t / tau2)
        kernel = (e1 - e2) / scale
        self.kernel = kernel


class CalciumToSpikeBase(tf.keras.layers.Layer):

    def call(self, v):
        v = tf.pad(v, [[0,0],[2,0]], 'CONSTANT')
        u = K.conv1d(
            v[..., None], self.kernel[::-1, None, None],
            1, 'valid', 'channels_last',
        )[..., 0]
        return tf.pad(u, [[0, 0], [self.pad, 0]], 'CONSTANT')


class CalciumToSpikeDefault(CalciumToSpikeBase):

    def __init__(self, tau1, tau2, hz=None, tscale=6.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tau1 < tau2:
            tau1, tau2 = tau1, tau2
        if hz is not None:
            tau1 *= hz
            tau2 *= hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = K.pow(r, -tau2 / d) - K.pow(r, -tau1 / d)
        e1 = K.exp(-1 / tau1)
        e2 = K.exp(-1 / tau2)
        kernel = K.stack([1.0, -e1 - e2, e1 * e2]) / (e1 - e2) * scale
        self.kernel = kernel
        self.pad = int(tau1 * tscale + 1)
