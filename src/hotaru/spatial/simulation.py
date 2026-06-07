from collections.abc import Sequence
from math import pi

import numpy as np
from scipy.signal import convolve2d
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
    check_overwrap = thr_overwrap < 1.0
    if check_overwrap:
        accept = np.zeros((height, width), 'float32')
    else:
        mask = np.ones((height, width), 'bool')
    for i in (pbar := trange(num, ncols=150, desc='make footprints')):
        count_ng = 0
        while True:
            fpi = sim_single_footprint(mean0[i], mean1[i], cv2[i], noise, rng)
            hi, wi = fpi.shape
            if check_overwrap:
                normalized_fpi = fpi / np.linalg.norm(fpi)
                candidate = convolve2d(accept, normalized_fpi, 'valid') < thr_overwrap
                n_cand = np.count_nonzero(candidate)
                pbar.set_postfix(cand=n_cand)
                if n_cand > 0:
                    break
                count_ng += 1
                if count_ng == 10:
                    raise ValueError()
            else:
                candidate = mask[:-hi, :-wi]
                break

        yl, xl = np.nonzero(candidate)
        j = int(rng.randint(minval=0, maxval=yl.size))
        yi, xi = int(yl[j]), int(xl[j])
        slices = slice(yi, yi + hi), slice(xi, xi + wi)
        fps[i, *slices] = fpi
        if check_overwrap:
            accept[*slices] += normalized_fpi

    return fps


def sim_single_footprint(
    mean0: float,
    mean1: float,
    cv2: float,
    noise: float,
    rng: Generator,
) -> Array:
    a = float(rng.gamma(mean0, cv2))
    b = float(rng.gamma(mean1, cv2))
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
