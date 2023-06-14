from collections import namedtuple
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from .filter.stats import calc_stats
from .footprint.clean import clean_segment_batch
from .footprint.find import find_peaks_batch
from .footprint.make import make_segment_batch
from .footprint.reduce import reduce_peaks_block
from .io.image import load_imgs
from .io.saver import (
    load,
    save,
)
from .proxmodel.regularizer import MaxNormNonNegativeL1
from .train.dynamics import SpikeToCalcium
from .train.loss import (
    gen_factor,
    gen_optimizer,
)

Data = namedtuple("Data", "imgs mask avgx avgt std0")
Stats = namedtuple("Stats", "imin imax istd icor")
Penalty = namedtuple("Penalty", "la lu bx bt")


class SilentProgress:
    def skip(self):
        pass

    def session(self, name):
        return self

    def set_count(self, total, status=None):
        pass

    def update(self, n, status=None):
        pass


class Progress:

    def __init__(self):
        self._curr = None

    def skip(self):
        pass

    def session(self, name):
        #print("session", name)
        if self._curr is not None:
            self._curr.close()
        self._name = name
        return self

    def set_count(self, total, status=None):
        #print("set_count", total, status)
        self._curr = tqdm(desc=self._name, total=total)

    def update(self, n, status=None):
        #print("update", n, status)
        if status is not None:
            self._curr.set_postfix_str(status, refresh=False)
        self._curr.update(n)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    pbar = Progress()
    nk = -1
    for stage in range(cfg.stage + 1):
        old_nk, nk = nk, spike_optimize(cfg, stage, pbar).shape[0]
        if old_nk == nk:
            break


def autoload(func, cfg, c, pbar=None, stage=None):
    if pbar is None:
        pbar = SilentProgress()
    cfg.stage, stage_saved = stage, cfg.stage
    file = Path(cfg.outdir) / c.outfile[0]
    force = c.force
    if stage is not None:
        force = force or (c.done is not None)
        if force:
            if c.done is None:
                c.done = -1
            force = stage > c.done
    if not force and file.exists():
        obj = load(file)
        pbar.skip()
    else:
        obj = func(c, pbar)
        cfg.stage = stage_saved
        save(file, obj)
        if stage is None:
            c.force = False
        else:
            c.done = stage
    cfg.stage = stage_saved
    return obj


def radius(c):
    return np.geomspace(c.rmin, c.rmax, c.rnum)


def dynamics(cfg):
    c = cfg.dynamics
    if c.kind == "double_exp":
        return SpikeToCalcium.double_exp(c.tau1, c.tau2, c.duration, cfg.data.hz)


def penalty(cfg, c):
    imgs, mask = load_imgs(cfg.data)
    if mask is None:
        nt, h, w = imgs.shape
        scale = nt * h * w
    else:
        scale = imgs.shape[0] * np.count_nonzero(mask)
    la = MaxNormNonNegativeL1(c.la / scale)
    lu = MaxNormNonNegativeL1(c.lu / scale)
    return Penalty(la, lu, c.bx, c.bt)


def stats(cfg, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    stats = autoload(
        lambda c, pbar: calc_stats(imgs, mask, c.batch, pbar.session("stats")),
        cfg,
        cfg.stats,
        pbar,
    )
    return Stats(stats.imin, stats.imax, stats.istd, stats.icor)


def get_frame(cfg, t, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    stats = autoload(
        lambda c, pbar: calc_stats(imgs, mask, c.batch, pbar.session("stats")),
        cfg,
        cfg.stats,
        pbar,
    )
    return (imgs[t] - stats.avgx - stats.avgt[t]) / stats.std0


def data(cfg, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    stats = autoload(
        lambda c, pbar: calc_stats(imgs, mask, c.batch, pbar.session("data")),
        cfg,
        cfg.stats,
        pbar,
    )
    return Data(imgs, mask, stats.avgx, stats.avgt, stats.std0)


def find(cfg, pbar=None):
    return autoload(
        lambda c, pbar: find_peaks_batch(
            data(cfg, pbar), radius(c), c.batch, pbar.session("find"),
        ),
        cfg,
        cfg.find,
        pbar,
    )


def reduce(cfg, pbar=None):
    return autoload(
        lambda c, pbar: reduce_peaks_block(
            find(cfg, pbar), c.rmin, c.rmax, c.thr, c.block_size,
        ),
        cfg,
        cfg.reduce,
        pbar,
    )


def make(cfg, pbar=None):
    return autoload(
        lambda c, pbar: make_segment_batch(
            data(cfg, pbar), reduce(cfg, pbar), c.batch, pbar.session("make"),
        ),
        cfg,
        cfg.make,
        pbar,
    )


def spike_prepare(cfg, stage, pbar=None):
    def prepare(c, pbar):
        return gen_factor(
            "temporal",
            make(cfg, pbar) if stage == 0 else footprint_clean(cfg, stage, pbar)[0],
            data(cfg, pbar),
            dynamics(cfg),
            penalty(cfg, cfg.temporal.penalty),
            c.batch,
            pbar.session(f"{stage}spike prepare"),
        )

    return autoload(prepare, cfg, cfg.temporal.prepare, pbar, stage)


def spike_optimize(cfg, stage, pbar=None):
    def optimize(c, pbar):
        optimizer = gen_optimizer(
            "temporal",
            spike_prepare(cfg, stage, pbar),
            dynamics(cfg),
            penalty(cfg, cfg.temporal.penalty),
            c.lr,
            c.scale,
            c.loss.num_devices,
        )
        optimizer.fit(
            c.n_epoch,
            c.n_step,
            c.early_stop.tol,
            pbar.session(f"{stage}spike optimize"),
            c.prox.num_devices,
        )
        spike = optimizer.val[0]
        normalize = spike / spike.max(axis=1, keepdims=True)
        tifffile.imwrite(f"spike{stage}.tif", normalize)
        return spike

    return autoload(optimize, cfg, cfg.temporal.optimize, pbar, stage)


def footprint_prepare(cfg, stage, pbar=None):
    def prepare(c, pbar):
        spk = spike_optimize(cfg, stage - 1, pbar)
        spk /= spk.max(axis=1, keepdims=True)
        return gen_factor(
            "spatial",
            spk,
            data(cfg, pbar),
            dynamics(cfg),
            penalty(cfg, cfg.spatial.penalty),
            c.batch,
            pbar.session(f"{stage}footprint prepare"),
        )

    return autoload(prepare, cfg, cfg.spatial.prepare, pbar, stage)


def footprint_optimize(cfg, stage, pbar=None):
    def optimize(c, pbar):
        optimizer = gen_optimizer(
            "spatial",
            footprint_prepare(cfg, stage, pbar),
            dynamics(cfg),
            penalty(cfg, cfg.spatial.penalty),
            c.lr,
            c.scale,
            c.loss.num_devices,
        )
        optimizer.fit(
            c.n_epoch,
            c.n_step,
            c.early_stop.tol,
            pbar.session(f"{stage}footprint optimize"),
            c.prox.num_devices,
        )
        return optimizer.val[0]

    return autoload(optimize, cfg, cfg.spatial.optimize, pbar, stage)


def footprint_clean(cfg, stage, pbar=None):
    def clean(c, pbar):
        imgs, mask = load_imgs(cfg.data)
        val = footprint_optimize(cfg, stage, pbar)
        val /= val.max(axis=1, keepdims=True)
        nk, h, w = val.shape[0], *imgs.shape[1:]
        if mask is None:
            fimgs = val.reshape(nk, h, w)
        else:
            fimgs = np.empty((nk, h, w), np.float32)
            fimgs[:, mask] = val
        cimgs, peaks = clean_segment_batch(
            fimgs,
            mask,
            radius(c),
            c.batch,
            pbar.session(f"{stage}footprint clean"),
        )
        cimgs = cimgs[peaks.r > 0]
        return cimgs, pd.DataFrame(peaks._asdict())

    return autoload(clean, cfg, cfg.clean, pbar, stage)
