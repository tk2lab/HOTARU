import tensorflow as tf


class SpikeToCalcium(tf.keras.layers.Layer):
    """"""

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self._kernel = self.add_weight("kernel", (size,), trainable=False)

    @property
    def kernel(self):
        return self.kernel_tensor().numpy()

    @kernel.setter
    def kernel(self, val):
        self.kernel_tensor().assign(val)

    def kernel_tensor(self):
        return self._kernel

    def call(self, u):
        return tf.nn.conv1d(
            u[..., None],
            self.kernel_tensor()[::-1, None, None],
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
        return self.kernel_tensor().numpy()

    @kernel.setter
    def kernel(self, val):
        self.kernel_tensor().assign(val)

    def kernel_tensor(self):
        return self._kernel

    def call(self, v):
        pad = tf.size(self.kernel) - 1
        v = tf.pad(v, [[0, 0], [2, 0]], "CONSTANT")
        u = tf.nn.conv1d(
            v[..., None],
            self.kernel_tensor()[::-1, None, None],
            1,
            "VALID",
            "NWC",
        )[..., 0]
        return tf.pad(u, [[0, 0], [self._pad, 0]], "CONSTANT")
