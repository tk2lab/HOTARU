from math import ceil
from math import pi
from math import sqrt

from keras import ops

from .typing import Tensor


def fft_convolve(core: Tensor, kernel: Tensor) -> Tensor:
    ndim = len(core.shape) - 1
    n_core = core.shape[-1]
    n_kernel = kernel.shape[-1]

    pad_core = ops.pad(core, ((0, 0),) * ndim + ((0, n_kernel - 1),))
    pad_kernel = ops.pad(kernel, ((0, n_core - 1),))

    fft_core = ops.rfft(pad_core, axis=-1)
    fft_kernel = ops.rfft(pad_kernel, axis=-1)
    conv_full = ops.irfft(fft_core * fft_kernel, axis=-1)
    start_idx = n_kernel - 1
    end_idx = n_core
    return conv_full[..., start_idx:end_idx]


def gaussian_kernel(r, *, nd: int = -1):
    if nd == -1:
        nd = int(4 * ceil(r))
    r = ops.convert_to_tensor(r, 'float32')
    z2 = ops.square(ops.arange(-nd, nd + 1, 1, dtype='float32') / r)
    kernel = ops.exp(-z2 / 2) / (sqrt(2 * pi) * r)
    return kernel, z2


def laplacian_of_gaussian_kernel(r, *, nd: int = -1):
    kernel0, z2 = gaussian_kernel(r, nd=nd)
    kernel2 = (1 - z2) * kernel0
    return kernel0, kernel2


def conv_1d(traces, kernel):
    shape = ops.shape(traces)
    g0 = ops.reshape(traces, (-1, shape[-1], 1))
    g1 = ops.conv(g0, kernel[:, None, None], 1, 'valid', 'channels_last')[:, :, 0]
    return ops.reshape(g1, (*shape[:-1], -1))


def conv_2d(imgs, kernel):
    shape = ops.shape(imgs)
    g0 = ops.reshape(imgs, (-1, *shape[-2:], 1))
    g1 = ops.conv(g0, kernel[:, None, None, None], (1, 1), 'valid', 'channels_last')
    g2 = ops.conv(g1, kernel[None, :, None, None], (1, 1), 'valid', 'channels_last')
    k = (kernel.size - 1) // 2
    return ops.reshape(ops.pad(g2, ((0, 0), (k, k), (k, k), (0, 0))), shape)


def laplace_2d(imgs, kernel0, kernel2):
    shape = ops.shape(imgs)
    g00 = ops.reshape(imgs, (-1, *shape[-2:], 1))
    g11 = ops.conv(g00, kernel0[:, None, None, None], (1, 1), 'valid', 'channels_last')
    g12 = ops.conv(g11, kernel2[None, :, None, None], (1, 1), 'valid', 'channels_last')
    g21 = ops.conv(g00, kernel0[None, :, None, None], (1, 1), 'valid', 'channels_last')
    g22 = ops.conv(g21, kernel2[:, None, None, None], (1, 1), 'valid', 'channels_last')
    k = (ops.size(kernel0) - 1) // 2
    return ops.reshape(ops.pad(g12 + g22, ((0, 0), (k, k), (k, k), (0, 0))), shape)


def gaussian_1d(traces, r, *, nd: int = -1):
    kernel, _ = gaussian_kernel(r, nd=nd)
    return conv_1d(traces, kernel)


def gaussian_2d(imgs, r, *, nd: int = -1):
    kernel, _ = gaussian_kernel(r, nd=nd)
    return conv_2d(imgs, kernel)


def gaussian_laplace_2d(imgs, r, *, nd: int = -1):
    kernel1, kernel2 = laplacian_of_gaussian_kernel(r, nd=nd)
    return laplace_2d(imgs, kernel1, kernel2)


def gaussian_laplace_2d_multi(imgs, rs, *, axis=-1):
    out = []
    for r in rs:
        out.append(gaussian_laplace_2d(imgs, r))
    return ops.stack(out, axis=axis)


def max_pool_2d(imgs, pool_size, strides = 1):
    shape = ops.shape(imgs)
    g0 = ops.reshape(imgs, (-1, *shape[-2:], 1))
    g1 = ops.max_pool(g0, pool_size, strides, 'same', 'channels_last')
    return ops.reshape(g1, shape)


def max_pool_3d(imgs, pool_size, strides = 1):
    shape = ops.shape(imgs)
    g0 = ops.reshape(imgs, (-1, *shape[-3:], 1))
    g1 = ops.max_pool(g0, pool_size, strides, 'same', 'channels_last')
    return ops.reshape(g1, shape)


def neighbor(imgs):
    kernel = ops.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], 'float32') / 8
    shape = ops.shape(imgs)
    g0 = ops.reshape(imgs, (-1, *shape[-2:], 1))
    g1 = ops.conv(g0, kernel[:, :, None, None], (1, 1), 'same', 'channels_last')
    return ops.reshape(g1, shape)


def simple_peaks(img, gauss_size, pool_size):
    g = gaussian_2d(img, gauss_size)
    m = max_pool_2d(g, pool_size)
    y, x = ops.nonzero(g == m)
    return y, x
