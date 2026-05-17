from math import pi

import numpy as np
import scipy.stats as st

from ..typing import Array
from .footprint import Footprints


def sim_single_footprint(
    ylist: Array,
    xlist: Array,
    radius_mean: float,
    radius_shape: float,
    ratio_mean: float,
    ratio_shape: float,
    pos: Array,
    rng: np.random.Generator,
) -> tuple[Array, int, int]:
    i = rng.integers(ylist.size)
    y, x = ylist[i], xlist[i]
    radius = st.lognorm(radius_shape, scale=radius_mean).rvs(random_state=rng)
    ratio = st.lognorm(ratio_shape, scale=ratio_mean).rvs(random_state=rng)
    angle = st.uniform(pi).rvs(random_state=rng)
    rotate = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    lmd = np.square(radius) * np.array([[1.0, 0.0], [0.0, ratio]])
    sigma = rotate @ lmd @ rotate.T
    fp = st.multivariate_normal((y, x), sigma).pdf(pos).astype('float32')
    fp /= fp.max()
    return fp, int(y), int(x)


def sim_footprints(
    num: int,
    height: int,
    width: int,
    radius_mean: float,
    radius_shape: float,
    ratio_mean: float,
    ratio_shape: float,
    intensity_mean: float,
    intensity_min: float,
    thr_overwrap: float,
    margin: int = 10,
    rng_or_seed: np.random.Generator | int | None = None,
) -> Footprints:
    match rng_or_seed:
        case np.random.Generator() as rng:
            pass
        case seed:
            rng = np.random.default_rng(seed)
    params = radius_mean, radius_shape, ratio_mean, ratio_shape

    y, x = np.mgrid[:height, :width]
    pos = np.stack((y, x), -1)

    ylist, xlist = y.flatten(), x.flatten()
    ok = (xlist > margin) & (xlist < width - margin) & (ylist > margin) & (ylist < height - margin)
    ylist, xlist = ylist[ok], xlist[ok]

    fps = np.empty((num, height, width), 'float32')
    ys, xs, gs = [], [], []
    for i in range(num):
        if ylist.size == 0:
            raise ValueError()

        fpi, yi, xi = sim_single_footprint(ylist, xlist, *params, pos, rng)
        if len(ys) > 0:
            while np.any(fpi[ys, xs] > thr_overwrap):
                fpi, yi, xi = sim_single_footprint(ylist, xlist, *params, pos, rng)
        gi = st.expon(loc=intensity_min, scale=intensity_mean - intensity_min).rvs(random_state=rng)
        ys.append(yi)
        xs.append(xi)
        gs.append(gi)
        fps[i] = gi * fpi

        ok = fpi[ylist, xlist] < thr_overwrap
        ylist, xlist = ylist[ok], xlist[ok]
    return Footprints(fps, np.array(ys, 'int32'), np.array(xs, 'int32'))
