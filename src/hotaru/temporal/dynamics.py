from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.fft import fft
from scipy.fft import ifft
from scipy.signal import fftconvolve

from ..typing import Array


@dataclass
class Kernel(ABC):
    length: float

    @abstractmethod
    def __call__(self, t: Array) -> Array: ...

    def get(self, hz: float, length: float | None = None, num: int | None = None) -> Array:
        match (num, length):
            case (num, None):
                num_samples = num
            case (None, length):
                num_samples = int(np.ceil(hz * length))
            case (None, None):
                num_samples = int(np.ceil(hz * self.length))
        t = np.arange(num_samples) / hz
        return self(t)

    def apply(self, core: Array, hz: float) -> Array:
        num_samples = int(np.ceil(self.length * hz))
        t = np.arange(num_samples) / hz
        kernel = self(t)
        out = fftconvolve(core, kernel, mode='valid')
        return out

    def reverse(self, obs: Array, hz: float, alpha: float = 1e-2):
        *_, size = obs.shape

        tau = np.arange(size) / hz
        kernel_values = self(tau)
        fft_kernel = fft(kernel_values)
        fft_kernel = np.conj(fft_kernel) / (np.abs(fft_kernel) ** 2 + alpha)

        fft_obs = fft(obs, axis=-1)
        core = np.real(ifft(fft_obs * fft_kernel, axis=-1))
        return core


@dataclass
class ExpKernel(Kernel):
    tau: float

    def __init__(self, tau: float, length: float | None = None):
        self.tau = tau
        super().__init__(5 * tau if length is None else length)

    def __call__(self, t: Array) -> Array:
        return np.exp(-t / self.tau)


@dataclass
class DoubleExpKernel(Kernel):
    tau1: float
    tau2: float

    def __init__(self, tau1: float, tau2: float, length: float | None = None):
        if tau1 < tau2:
            tau1, tau2 = tau2, tau1
        self.tau1 = tau1
        self.tau2 = tau2
        super().__init__(5 * tau1 if length is None else length)

    def __call__(self, t: Array) -> Array:
        tau1, tau2 = self.tau1, self.tau2
        xmax = (tau1 * tau2) / (tau1 - tau2) * np.log(tau1 / tau2)
        ymax = np.exp(-xmax / tau1) - np.exp(-xmax / tau2)
        return (np.exp(-t / tau1) - np.exp(-t / tau2)) / ymax
