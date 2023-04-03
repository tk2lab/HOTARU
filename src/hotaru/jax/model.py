import pathlib
from functools import cached_property

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .filter.stats import calc_stats, Stats, ImageStats
from .filter.laplace import gaussian_laplace, gaussian_laplace_multi
from .io.image import load_imgs
from .io.mask import get_mask
from .footprint.find import find_peak, find_peak_batch, PeakVal
from .footprint.reduce import reduce_peak, reduce_peak_block
from .footprint.make import make_segment_batch


class Model:

    def load_imgs(self, path, mask, hz):
        path = pathlib.Path(path)
        self.imgs = load_imgs(path)
        self.mask = get_mask(mask, self.imgs)
        self.hz = hz
        self.stats_path = path.with_stem(f"{path.stem}_{mask}").with_suffix(".npz")
        self.istats_path = path.with_stem(f"{path.stem}_{mask}_i").with_suffix(".npz")

    def load_stats(self):
        if self.stats_path.exists():
            self.stats = Stats.load(self.stats_path)
            self.istats = ImageStats.load(self.istats_path)
            return True
        return False

    def calc_stats(self, pbar=None):
        nt, h, w = self.imgs.shape
        stats, istats = calc_stats(self.imgs, self.mask, pbar=pbar)
        stats.save(self.stats_path)
        istats.save(self.istats_path)
        self.stats = stats
        self.istats = istats

    @property
    def shape(self):
        nt, x0, y0, mask, avgx, avgt, std0, min0, max0 = self.stats
        return nt, *mask.shape

    @property
    def min(self):
        return self.stats.min

    @property
    def max(self):
        return self.stats.max

    def frame(self, t):
        nt, x0, y0, mask, avgx, avgt, std0, min0, max0 = self.stats
        img = np.array((self.imgs[t] - avgx - avgt[t]) / std0)
        img[~mask] = 0
        return img

    def load_peaks(self, rmin, rmax, rnum, rtype="log"):
        if rtype == "log":
            self.radius = np.power(2, np.linspace(np.log2(rmin), np.log2(rmax), rnum))
        elif rtype == "lin":
            self.radius = np.linspace(rmin, rmax, rnum)
        self.peakval_path = self.stats_path.with_stem(f"{self.stats_path.stem}_{rmin}_{rmax}_{rnum}_{rtype}").with_suffix(".npz")
        if self.peakval_path.exists():
            self.peakval = PeakVal.load(self.peakval_path)
            return True
        return False

    def calc_peaks(self, pbar=None):
        print(self.radius)
        self.peakval = find_peak_batch(self.imgs, self.radius, self.stats, pbar=pbar)
        self.peakval.save(self.peakval_path)

    def load_footprints(self, thr):
        self.thr = thr
        self.make_path = self.peakval_path.with_stem(f"{self.peakval_path.stem}_{thr}").with_suffix(".np")
        if self.make_path.exists():
            self.footprints = jnp.load(self.make_path)
            return True
        return False

    def make_footprints(self, pbar=None):
        peaks = reduce_peak_block(self.peakval, self.radius[0], self.radius[-1], self.thr, 100)
        t, r, y, x = (peaks[n].to_numpy() for n in "tryx")
        self.footprints = make_segment_batch(self.imgs, t, r, y, x, self.stats, pbar=pbar)
        jnp.save(self.make_path, self.footprints)
