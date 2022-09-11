import numpy as np
import tensorflow as tf


class DoubleExpMixin:

    def init_double_exp(self, hz, tausize):
        self.hz = hz
        self.tausize = tausize
        self.spike_to_calcium = SpikeToCalcium(tausize, name="to_cal")
        self.calcium_to_spike = CalciumToSpike(tausize, 3, name="to_cal")

    def set_double_exp(self, tau1, tau2):
        tau1 *= self.hz
        tau2 *= self.hz

        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)

        t = np.arange(1, self.tausize + 1)
        e1 = np.exp(-t / tau1)
        e2 = np.exp(-t / tau2)
        kernel = (e1 - e2) / scale
        self.spike_to_calcium.set_kernel(kernel)

        kernel = np.array([1.0, -e1[0] - e2[0], e1[0] * e2[0]]) / kernel[0]
        self.calcium_to_spike.set_kernel(kernel)


class SpikeToCalcium(tf.keras.layers.Layer):
    """"""

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self._kernel = self.add_weight("kernel", (size,), trainable=False)

    @property
    def kernel(self):
        return self._kernel

    def set_kernel(self, val):
        self._kernel.assign(val)

    def get_kernel(self):
        return self.kernel.numpy()

    def call(self, u):
        return tf.nn.conv1d(
            u[..., None],
            self.kernel[::-1, None, None],
            1,
            "VALID",
            "NWC",
        )[..., 0]


class CalciumToSpike(tf.keras.layers.Layer):
    """"""

    def __init__(self, fw_size, bw_size, **kwargs):
        super().__init__(**kwargs)
        self._kernel = self.add_weight("kernel", (bw_size,), trainable=False)
        self._pad = self.add_weight("pad", (), trainable=False)
        self._pad.assign(fw_size - 1)

    @property
    def kernel(self):
        return self._kernel

    def set_kernel(self, val):
        self._kernel.assign(val)

    def get_kernel(self):
        return self.kernel.numpy()

    def call(self, v):
        pad = tf.size(self.kernel) - 1
        v = tf.pad(v, [[0, 0], [2, 0]], "CONSTANT")
        u = tf.nn.conv1d(
            v[..., None],
            self.kernel[::-1, None, None],
            1,
            "VALID",
            "NWC",
        )[..., 0]
        return tf.pad(u, [[0, 0], [self._pad, 0]], "CONSTANT")
