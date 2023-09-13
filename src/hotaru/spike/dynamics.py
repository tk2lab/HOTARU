import jax.lax as lax
import jax.numpy as jnp
import numpy as np


def get_dynamics(dynamics):
    match dynamics:
        case SpikeToCalcium():
            return dynamics
        case {"label": _, "type": "double_exp", **args}:
            return SpikeToCalcium.double_exp(**args)
        case _:
            raise ValueError("invalid dynamics type")


def get_rdynamics(dynamics):
    match dynamics:
        case CalciumToSpike():
            return dynamics
        case {"label": _, "type": "double_exp", **args}:
            return CalciumToSpike.double_exp(**args)
        case _:
            raise ValueError("invalid dynamics type")


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
        self.kernel = jnp.array(kernel, jnp.float32)[jnp.newaxis, jnp.newaxis, ::-1]

    def __call__(self, spike):
        spike = spike[:, jnp.newaxis, ...]
        return lax.conv(spike, self.kernel, (1,), "valid")[:, 0, ...]

    @property
    def size(self):
        return self.kernel.size


class CalciumToSpike:
    @classmethod
    def double_exp(cls, tau1, tau2, duration, hz=1.0):
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
        tau1 *= hz
        tau2 *= hz
        r = tau1 / tau2
        d = tau1 - tau2
        scale = np.power(r, -tau2 / d) - np.power(r, -tau1 / d)
        e1 = np.exp(-1 / tau1)
        e2 = np.exp(-1 / tau2)
        kernel = np.array([1.0, -(e1 + e2), e1 * e2]) / (e1 - e2) * scale
        size = int(np.ceil(duration * hz))
        pad = size + 1
        return cls(kernel, pad)

    def __init__(self, kernel, pad):
        self.kernel = jnp.array(kernel, jnp.float32)[jnp.newaxis, jnp.newaxis, ::-1]
        self.pad = pad

    def __call__(self, calcium):
        calcium = calcium[:, jnp.newaxis, ...]
        spike = lax.conv(calcium, self.kernel, (1,), "valid")[:, 0, ...]
        spike = jnp.pad(spike, [[0, 0], [self.pad, 0]])
        spike += spike.min(axis=1, keepdims=True)
        return spike

    @property
    def size(self):
        return self.kernel.size
