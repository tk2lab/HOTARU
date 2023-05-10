import jax.numpy as jnp
import jax.lax as lax
import numpy as np


class SpikeToCalcium:

    @classmethod
    def double_exp(cls, tau1, tau2, duration, hz=1.0):
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
        size = int(np.ceil(duration * hz))
        tau1 *= hz
        tau2 *= hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        t = np.arange(1, size + 1)
        e1 = np.exp(-t / tau1)
        e2 = np.exp(-t / tau2)
        return cls((e1 - e2) / scale)

    def __init__(self, kernel):
        self.kernel = jnp.array(kernel, jnp.float32)[None, None, ::-1]

    def __call__(self, spike):
        return lax.conv(spike[:, None, ...], self.kernel, (1,), "valid")[:, 0, ...]

    @property
    def size(self):
        return self.kernel.size
