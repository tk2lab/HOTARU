import pathlib

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..io.image import load_imgs
from ..io.mask import get_mask
from ..io.saver import (
    load,
    save,
)
from ..jax.filter.gaussian import gaussian
from ..jax.filter.laplace import gaussian_laplace
from ..jax.filter.pool import max_pool
from .filter.stats import (
    ImageStats,
    Stats,
    calc_stats,
)
from .footprint.find import (
    PeakVal,
    find_peak_batch,
)
from .footprint.make import make_segment_batch
from .footprint.reduce import reduce_peak_block

"""
from .train.prepare import prepare
"""


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = cfg.env.buffer
        self.num_devices = cfg.env.num_devices
        self.block_size = cfg.env.block_size

    def load_imgs(self, path, mask, hz):
        path = pathlib.Path(path)
        self.imgs = load_imgs(path)
        self.mask = get_mask(mask, self.imgs)
        self.hz = hz
        self.stats_path = path.with_stem(f"{path.stem}_{mask}").with_suffix(".npz")
        self.istats_path = path.with_stem(f"{path.stem}_{mask}_i").with_suffix(".npz")

    def load_stats(self):
        if not self.stats_path.exists():
            return False
        self.stats = load(self.stats_path)
        self.istats = load(self.istats_path)
        return True

    def calc_stats(self, path=None, mask=None, hz=None, pbar=None):
        if path is not None:
            self.load_imgs(path, mask, hz)
        if not self.load_stats():
            nt, h, w = self.imgs.shape
            self.stats, self.istats = calc_stats(self.imgs, self.mask, pbar)
            save(self.stats_path, self.stats)
            save(self.istats_path, self.istats)

    @property
    def nt(self):
        return self.imgs.shape[0]

    @property
    def shape(self):
        return self.mask.shape

    @property
    def min(self):
        return self.istats.imin.min()

    @property
    def max(self):
        return self.istats.imax.max()

    def simple_peaks(self, gauss, maxpool):
        imin, imax, istd, icor = self.istats
        if gauss > 0:
            fil_cori = gaussian(icor[None, ...], gauss)[0]
        else:
            fil_cori = cori
        max_cori = max_pool(fil_cori, (maxpool, maxpool), (1, 1), "same")
        return np.where(fil_cori == max_cori)

    def frame(self, t):
        nt, x0, y0, mask, avgx, avgt, std0 = self.stats
        img = np.array((self.imgs[t] - avgx - avgt[t]) / std0)
        img[~mask] = 0
        return img

    def load_peakval(self, rmin, rmax, rnum, rtype="log"):
        if rtype == "log":
            self.radius = np.power(2, np.linspace(np.log2(rmin), np.log2(rmax), rnum))
        elif rtype == "lin":
            self.radius = np.linspace(rmin, rmax, rnum)
        print(self.radius)
        self.peakval_path = self.stats_path.with_stem(
            f"{self.stats_path.stem}_{rmin}_{rmax}_{rnum}_{rtype}"
        ).with_suffix(".npz")
        if not self.peakval_path.exists():
            return False
        self.peakval = load(self.peakval_path)
        return True

    def calc_peakval(self, rmin, rmax, rnum, rtype="log", pbar=None):
        if not self.load_peakval(rmin, rmax, rnum, rtype):
            self.peakval = find_peak_batch(self.imgs, self.stats, self.radius, pbar)
            save(self.peakval_path, self.peakval)

    def load_peaks(self, rmin, rmax, thr):
        self.rmin = self.radius[rmin]
        self.rmax = self.radius[rmax]
        self.thr = thr
        print(self.radius)
        print(self.rmin)
        print(self.rmax)
        self.peak_path = self.peakval_path.with_stem(
            f"{self.peakval_path.stem}_{rmin}_{rmax}_{thr}"
        ).with_suffix(".csv")
        if not self.peak_path.exists():
            return False
        self.peaks = pd.read_csv(self.peak_path, index_col=0)
        return True

    def calc_peaks(self, rmin, rmax, thr):
        if not self.load_peaks(rmin, rmax, thr):
            self.peaks = reduce_peak_block(
                self.peakval,
                self.rmin,
                self.rmax,
                self.thr,
                self.cfg.env.block_size,
            )
            self.peaks.to_csv(self.peak_path)

    def load_footprints(self, path):
        self.footprints_path = pathlib.Path(path)
        if not self.footprints_path.exists():
            return False
        self.footprints = jnp.load(self.footprints_path)
        return True

    def make_footprints(self, pbar=None):
        t, r, y, x = (self.peaks[n].to_numpy() for n in "tryx")
        self.footprints = make_segment_batch(
            self.imgs,
            self.stats,
            t,
            r,
            y,
            x,
            pbar,
        )
        jnp.save(self.footprints_path, self.footprints)

    def load_spikes(self, path):
        self.spikes_path = pathlib.Path(path)
        if not self.spikes_path.exists():
            return False
        self.spikes = jnp.load(self.spikes_path)
        return True

    def update_spikes(self, pbar=None):
        pass

    def prepare(self, pbar=None):
        nc, h, w = self.footprints.shape
        val = self.footprints.reshape(nc, h * w)
        bx, by = 0.0, 0.0
        trans = True
        return prepare(
            val,
            self.imgs,
            self.stats,
            bx,
            by,
            trans,
            self.buffer,
            self.num_devices,
            pbar,
        )
