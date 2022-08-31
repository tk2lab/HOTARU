import numpy as np
import tensorflow as tf


class SpikeToCalcium(tf.keras.layers.Layer):

    def call(self, u):
        return tf.nn.conv1d(
            u[..., None],
            self.kernel[::-1, None, None],
            1,
            "VALID",
            "NWC",
        )[..., 0]


class CalciumToSpike(tf.keras.layers.Layer):

    def call(self, v):
        v = tf.pad(v, [[0, 0], [2, 0]], "CONSTANT")
        u = tf.nn.conv1d(
            v[..., None],
            self.kernel[::-1, None, None],
            1,
            "VALID",
            "NWC",
        )[..., 0]
        return tf.pad(u, [[0, 0], [self.pad, 0]], "CONSTANT")


class DoubleExpMixin:

    def set(self, hz, tau1, tau2, tscale):
        if tau1 < tau2:
            tau1, tau2 = tau1, tau2
        self.hz = hz
        self.tau1 = tau1
        self.tau2 = tau2
        self.tscale = tscale

    def scale(self):
        tau1 = self.tau1 * self.hz
        tau2 = self.tau2 * self.hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        return tau1, tau2, scale

    def _pad(self):
        tau1, tau2, scale = self.scale()
        t = np.arange(1.0, tau1 * self.tscale + 2.0, dtype=np.float32)
        return t.size - 1

    def forward(self):
        tau1, tau2, scale = self.scale()
        t = np.arange(1.0, tau1 * self.tscale + 2.0, dtype=np.float32)
        e1 = np.exp(-t / tau1)
        e2 = np.exp(-t / tau2)
        kernel = (e1 - e2) / scale
        return kernel

    def backward(self):
        tau1, tau2, scale = self.scale()
        e1 = np.exp(-1 / tau1)
        e2 = np.exp(-1 / tau2)
        kernel = scale * np.array([1.0, -e1 - e2, e1 * e2]) / (e1 - e2)
        return kernel


class SpikeToCalciumDoubleExp(SpikeToCalcium, DoubleExpMixin):

    def __init__(self, hz, tau1, tau2, tscale, **kwargs):
        super().__init__(**kwargs)
        self.set(hz, tau1, tau2, tscale)
        self.pad = tf.convert_to_tensor(self._pad(), tf.int32)
        self.kernel = tf.convert_to_tensor(self.forward(), tf.float32)


class CalciumToSpikeDoubleExp(CalciumToSpike, DoubleExpMixin):

    def __init__(self, hz, tau1, tau2, tscale, **kwargs):
        super().__init__(**kwargs)
        self.set(hz, tau1, tau2, tscale)
        self.pad = tf.convert_to_tensor(self._pad(), tf.int32)
        self.kernel = tf.convert_to_tensor(self.backward(), tf.float32)
