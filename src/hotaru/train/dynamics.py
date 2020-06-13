import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


class SpikeToCalcium(tf.keras.layers.Layer):

    def set_double_exp(self, tau1, tau2, hz=None, tscale=6.0):
        if tau1 < tau2:
            tau1, tau2 = tau1, tau2
        if hz is not None:
            tau1 *= hz
            tau2 *= hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        t = np.arange(1.0, tau1 * tscale + 2.0, dtype=np.float32)
        e1 = np.exp(-t / tau1)
        e2 = np.exp(-t / tau2)
        kernel = (e1 - e2) / scale
        if hasattr(self, 'kernel'):
            if kernel.size > K.int_shape(self.kernel)[0]:
                raise RuntimeError(f'filter size mismatch: {kernel.size}')
        else:
            self.kernel = self.add_weight('kernel', (kernel.size,), trainable=False)
            self.pad = int(tau1 * tscale + 1)
        K.set_value(self.kernel, kernel)

    def call(self, u):
        return K.conv1d(
            u[..., None], self.kernel[::-1, None, None],
            1, 'valid', 'channels_last',
        )[..., 0]


class CalciumToSpike(tf.keras.layers.Layer):

    def set_double_exp(self, tau1, tau2, hz=None, tscale=6.0):
        if tau1 < tau2:
            tau1, tau2 = tau1, tau2
        if hz is not None:
            tau1 *= hz
            tau2 *= hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        e1 = np.exp(-1 / tau1)
        e2 = np.exp(-1 / tau2)
        kernel = np.array([1.0, -e1 - e2, e1 * e2]) / (e1 - e2) * scale
        if hasattr(self, 'kernel'):
            if kernel.size > K.int_shape(self.kernel)[0]:
                raise RuntimeError(f'filter size mismatch: {kernel.size}')
        else:
            self.kernel = self.add_weight('kernel', (kernel.size,), trainable=False)
            self.pad = int(tau1 * tscale + 1)
        K.set_value(self.kernel, kernel)

    def call(self, v):
        v = tf.pad(v, [[0, 0], [2, 0]], 'CONSTANT')
        u = K.conv1d(
            v[..., None], self.kernel[::-1, None, None],
            1, 'valid', 'channels_last',
        )[..., 0]
        return tf.pad(u, [[0, 0], [self.pad, 0]], 'CONSTANT')
