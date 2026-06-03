from abc import ABC
from abc import abstractmethod
from math import ceil

from keras import ops

from ..models import Layer
from ..saving import Config
from ..typing import Tensor


class Kernel(Layer, ABC):
    def __init__(self, hz: float, length: float, alpha: float = 1e-2, **kwargs):
        super().__init__(**kwargs)
        self.hz = hz
        self.length = length
        self.alpha = alpha
        self._jax_static_argnames = 'upsample_factor'

    def get_config(self) -> Config:
        return {'hz': self.hz, 'length': self.length, 'alpha': self.alpha, **super().get_config()}

    @abstractmethod
    def kernel_fn(self, t: Tensor) -> Tensor: ...

    def pad_size(self, upsample_factor: int = 1) -> int:
        return ceil(self.hz * self.length * upsample_factor) - 1

    def kernel(
        self,
        hz: float | None = None,
        length: float | None = None,
        num: int | None = None,
        upsample_factor: int = 1,
    ) -> Tensor:
        if hz is None:
            hz = self.hz
        hz *= upsample_factor
        if length is None:
            length = self.length
        if num is None:
            num = ceil(hz * length)
        t = ops.arange(num) / hz
        return self.kernel_fn(t) / self.hz

    def call(self, core: Tensor, upsample_factor: int = 1) -> Tensor:
        kernel = ops.flip(self.kernel(upsample_factor=upsample_factor))[:, None, None]
        return ops.conv(core[:, :, None], kernel, 1, 'valid', 'channels_last')[:, :, 0]

    def reverse(self, obs: Tensor) -> Tensor:
        *_, size = obs.shape
        tau = ops.arange(size) / self.hz
        kernel_values = self.kernel_fn(tau)
        fft_kernel = ops.rfft(kernel_values)
        fft_kernel = ops.conj(fft_kernel) / (ops.abs(fft_kernel) ** 2 + self.alpha)
        fft_obs = ops.rfft(obs, axis=-1)
        core = ops.irfft(fft_obs * fft_kernel, axis=-1)
        return core


class ExpKernel(Kernel):
    def __init__(self, tau: float, **kwargs):
        kwargs.setdefault('length', 5 * tau)
        super().__init__(**kwargs)
        self.tau = tau

    def kernel_fn(self, t: Tensor) -> Tensor:
        return ops.exp(-t / self.tau) / self.tau


class DoubleExpKernel(Kernel):
    def __init__(self, tau1: float, tau2: float, **kwargs):
        if tau1 < tau2:
            tau1, tau2 = tau2, tau1
        kwargs.setdefault('length', 5 * tau1)
        super().__init__(**kwargs)
        self.tau1 = tau1
        self.tau2 = tau2

    def kernel_fn(self, t: Tensor) -> Tensor:
        tau1, tau2 = self.tau1, self.tau2
        #xmax = (tau1 * tau2) / (tau1 - tau2) * ops.log(tau1 / tau2)
        #ymax = ops.exp(-xmax / tau1) - ops.exp(-xmax / tau2)
        return (ops.exp(-t / tau1) - ops.exp(-t / tau2)) / (tau1 - tau2)
