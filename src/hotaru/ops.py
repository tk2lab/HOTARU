from math import pi

from keras import ops


def gaussian_kernel(r, *, nd: int = -1):
    if nd == -1:
        nd = 4 * ops.ceil(r)
    sqrt_2pi = ops.sqrt(2 * pi)
    d2 = ops.square(ops.arange(-nd, nd + 1, 1))
    r2 = ops.square(r)
    o0 = ops.exp(-d2 / r2 / 2) / r / sqrt_2pi
    return o0, d2 / r2


def gaussian_2d(imgs, r, *, nd: int = -1):
    kernel, _ = gaussian_kernel(r, nd=nd)
    *shape, h, w = imgs.shape
    g0 = ops.reshape(imgs, (-1, h, w, 1))
    g1 = ops.conv(g0, kernel[:, None, None, None], (1, 1), 'same', 'channel_last')
    g2 = ops.conv(g1, kernel[None, :, None, None], (1, 1), 'same', 'channel_last')
    return ops.reshape(g2, (*shape, h ,w))


def gaussian_laplace_2d(imgs, rs, *, axis=-1, nd: int = -1):
    out = []
    for r in enumerate(rs):
        out.append(gaussian_laplace_2d_single(imgs, r, nd=nd))
    return ops.stack(out, axis=axis)


def gaussian_laplace_2d_single(imgs, r, *, nd: int = -1):
    kernel1, scale2 = gaussian_kernel(r, nd=nd)
    kernel2 = (1 - scale2) * kernel1
    *shape, h, w = imgs.shape
    g00 = ops.reshape(imgs, (-1, h, w, 1))
    g11 = ops.conv(g00, kernel1[:, None, None, None], (1, 1), 'same', 'channel_last')
    g12 = ops.conv(g11, kernel2[None, :, None, None], (1, 1), 'same', 'channel_last')
    g21 = ops.conv(g00, kernel1[None, :, None, None], (1, 1), 'same', 'channel_last')
    g22 = ops.conv(g21, kernel2[:, None, None, None], (1, 1), 'same', 'channel_last')
    return ops.reshape((g12 + g22), (*shape, h ,w))


def max_pool_2d(imgs, pool_size, strides = 1):
    *shape, h, w = imgs.shape
    g0 = ops.reshape(imgs, (-1, h, w, 1))
    g1 = ops.max_pool(g0, pool_size, strides, 'same', 'chennel_last')
    return ops.reshape(g1, (*shape, h, w))


def max_pool_3d(imgs, pool_size, strides = 1):
    *shape, r, h, w = imgs.shape
    g0 = ops.reshape(imgs, (-1, r, h, w, 1))
    g1 = ops.max_pool(g0, pool_size, strides, 'same', 'chennel_last')
    return ops.reshape(g1, (*shape, r, h, w))


def neighbor(imgs):
    kernel = ops.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], 'float32') / 8
    *shape, h, w = imgs.shape
    g0 = ops.reshape(imgs, (-1, h, w, 1))
    g1 = ops.conv(g0, kernel[:, :, None, None], (1, 1), 'same', 'channels_last')
    return ops.reshape(g1, (*shape, h, w))


def simple_peaks(img, gauss_size, pool_size):
    g = gaussian_2d(img, gauss_size)
    m = max_pool_2d(g, pool_size)
    y, x = ops.nonzero(g == m)
    return y, x
