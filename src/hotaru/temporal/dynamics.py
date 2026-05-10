import numpy as np


def exp_kernel(tau, hz, size=None):
    tau *= hz
    if size is None:
        size = int(np.ceil(5 * tau))
    t = np.arange(1, size + 1)
    kernel = np.exp(-t / tau)
    kernel /= kernel.sum()
    return kernel


def double_exp_kernel(tau1, tau2, hz, size=None):
    if tau1 > tau2:
        tau1, tau2 = tau2, tau1
    tau1 *= hz
    tau2 *= hz
    if size is None:
        size = int(np.ceil(5 * tau1))
    t = np.arange(1, size + 1)
    e1 = np.exp(-t / tau1)
    e2 = np.exp(-t / tau2)
    kernel = e1 - e2
    kernel /= kernel.sum()
    return kernel
