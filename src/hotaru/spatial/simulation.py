from collections.abc import Sequence
from math import pi

import numpy as np
from scipy.ndimage import binary_closing
from tqdm import trange

from ..random import Generator
from ..typing import Array


def sim_footprints(
    height: int,
    width: int,
    mean0: Sequence[float],
    mean1: Sequence[float],
    cv2: Sequence[float],
    noise: float,
    thr_overwrap: float,
    rng: Generator,
) -> Array:
    num = len(mean0)
    fps = np.zeros((num, height, width), 'float32')
    pos_mask = np.ones((height, width), 'bool')
    accept_mask = np.zeros((height, width), 'bool')
    for i in trange(num, ncols=150, desc='make footprints'):
        if np.count_nonzero(pos_mask) == 0:
            raise ValueError()
        while True:
            fpi = sim_single_footprint(mean0[i], mean1[i], cv2[i], noise, rng)
            fpi_mask = binary_closing(fpi <= thr_overwrap)
            hi, wi = fpi.shape
            yl, xl = np.nonzero(pos_mask)

            valid_indices = np.nonzero((yl + hi < height) & (xl + wi < width))[0]
            if len(valid_indices) == 0:
                continue
            j_idx = int(rng.randint(minval=0, maxval=len(valid_indices)))
            j = valid_indices[j_idx]
            yi, xi = yl[j], xl[j]

            slices = slice(yi, yi + hi), slice(xi, xi + wi)
            if not np.any(fpi_mask & accept_mask[slices]):
                break
        fps[i, *slices] = fpi
        pos_mask &= fps[i] <= thr_overwrap
    return fps


def sim_single_footprint(
    mean0: float,
    mean1: float,
    cv2: float,
    noise: float,
    rng: Generator,
) -> Array:
    a = float(rng.invgauss(mean0, cv2))
    b = float(rng.invgauss(mean1, cv2))
    c = float(rng.uniform(0, pi))
    sin, cos = np.sin(c), np.cos(c)

    rotate = np.array(((cos, -sin), (sin, cos)))
    xsize = np.floor(np.hypot(a * cos, b * sin))
    ysize = np.floor(np.hypot(a * sin, b * cos))
    xr = np.arange(-xsize, xsize + 0.1)
    yr = np.arange(-ysize, ysize + 0.1)
    x, y = np.einsum('ij,jkl->ikl', rotate, np.stack(np.meshgrid(xr, yr)))
    fpi = np.maximum(0, 1 - np.square(x / a) - np.square(y / b))
    fpi += noise * np.where(fpi > 0, rng.normal(shape=fpi.shape), 0)
    return np.clip(fpi, 0, 1)


def sim_neuropil_basis(
    height: int,
    width: int,
    scale: float,
    nk: int,
) -> Array:
    kx = np.ones((width, nk), dtype='float32')
    xs = np.arange(width, dtype='float32') + 0.5
    for k in range((nk - 1) // 2):
        kx[:, 2 * k + 1] = np.sin(2 * pi * xs * (1 + k) / scale)
        kx[:, 2 * k + 2] = np.cos(2 * pi * xs * (1 + k) / scale)

    ky = np.ones((height, nk), dtype=np.float32)
    ys = np.arange(height, dtype=np.float32) + 0.5
    for k in range((nk - 1) // 2):
        ky[:, 2 * k + 1] = np.sin(2 * pi * ys * (1 + k) / scale)
        ky[:, 2 * k + 2] = np.cos(2 * pi * ys * (1 + k) / scale)

    s_basis = np.einsum('yk,xl->klyx', ky, kx).reshape(nk * nk, height, width)[1:]
    return s_basis
