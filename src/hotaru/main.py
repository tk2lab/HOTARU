from collections import namedtuple
from pathlib import Path

import hydra
import jax
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from .io.logger import logger
from .io.saver import (
    load,
    save,
)
from .io.image import load_imgs
from .filter.stats import (
    Stats,
    calc_stats,
)
from .footprint.find import (
    PeakVal,
    find_peaks_batch,
)
from .footprint.reduce import reduce_peaks_block
from .footprint.make import make_segment_batch
from .footprint.clean import clean_segment_batch
from .train.dynamics import SpikeToCalcium
from .train.loss import (
    gen_factor,
    gen_optimizer,
)
from .proxmodel.regularizer import MaxNormNonNegativeL1


Data = namedtuple("Data", "imgs mask avgx avgt std0")
Stats = namedtuple("Stats", "imin imax istd icor")
Penalty = namedtuple("Penalty", "la lu bx bt")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data(cfg, tqdm)
    find(cfg, tqdm)
    make(cfg, tqdm)
    nk = spike(cfg, 0, tqdm).shape[0]
    for stage in range(1, cfg.stage + 1):
        footprint(cfg, stage, tqdm)
        old_nk, nk = nk, spike(cfg, stage, tqdm).shape[0]
        if old_nk == nk:
            break


def autoload(func, cfg, c, stage=None):
    cfg.stage, stage_saved = stage, cfg.stage
    file = Path(cfg.outdir) / c.outfile
    force = c.force
    if stage is not None:
        force = force or (c.done is not None)
        if force:
            if c.done is None:
                c.done = -1
            force = stage > c.done
    if not force and file.exists():
        obj = load(file)
    else:
        obj = func(c)
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


def data(cfg, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    stats = autoload(
        lambda c: calc_stats(imgs, mask, c.batch, pbar()),
        cfg, cfg.stats,
    )
    return Data(imgs, mask, stats.avgx, stats.avgt, stats.std0)


def stats(cfg, pbar=None):
    def func(c):
        imgs, mask = load_imgs(cfg.data)
        stats = calc_stats(imgs, mask, c.batch, pbar()),
        return Stats(stats.imin, stats.imax, stats.istd, stats.icor)
    return autoload(func, cfg, cfg.stats)


def find(cfg, pbar=None):
    return autoload(
        lambda c: find_peaks_batch(data(cfg), radius(c), c.batch, pbar()),
        cfg, cfg.find,
    )


def reduce(cfg, pbar=None):
    return autoload(
        lambda c: reduce_peaks_block(find(cfg), c.rmin, c.rmax, c.thr, c.block_size),
        cfg, cfg.reduce,
    )


def make(cfg, pbar=None):
    return autoload(
        lambda c: make_segment_batch(data(cfg), reduce(cfg), c.batch, pbar()),
        cfg, cfg.make,
    )


def spike(cfg, stage, pbar=None):
    def prepare(c):
        return gen_factor(
            "temporal",
            make(cfg) if stage == 0 else footprint(cfg, stage),
            data(cfg),
            dynamics(cfg),
            penalty(cfg, cfg.temporal.penalty),
            c.batch,
            pbar and pbar(desc=f"{stage}spike prepare"),
        )
    def optimize(c):
        optimizer = gen_optimizer(
            "temporal",
            factor,
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
            pbar and pbar(desc=f"{stage}spike optimize"),
            c.prox.num_devices,
        )
        spike = optimizer.val[0]
        normalize = spike / spike.max(axis=1, keepdims=True)
        tifffile.imwrite(f"spike{stage}.tif", normalize)
        return spike
    factor = autoload(prepare, cfg, cfg.temporal.prepare, stage)
    return autoload(optimize, cfg, cfg.temporal.optimize, stage)


def footprint(cfg, stage, pbar=None):
    def prepare(c):
        spk = spike(cfg, stage - 1)
        spk /= spk.max(axis=1, keepdims=True)
        return gen_factor(
            "spatial",
            spk,
            data(cfg),
            dynamics(cfg),
            penalty(cfg, cfg.spatial.penalty),
            c.batch,
            pbar and pbar(desc=f"{stage}footprint prepare"),
        )
    def optimize(c):
        optimizer = gen_optimizer(
            "spatial",
            factor,
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
            pbar and pbar(desc=f"{stage}footprint optimize"),
            c.prox.num_devices,
        )
        return optimizer.val[0]
    def clean(c):
        imgs, mask = load_imgs(cfg.data)
        val = footprint
        val /= val.max(axis=1, keepdims=True)
        nk, h, w = val.shape[0], *imgs.shape[1:]
        _radius = radius(c)
        if mask is None:
            fimgs = val.reshape(nk, h, w)
        else:
            fimgs = np.empty((nk, h, w), np.float32)
            fimgs[:, mask] = val
        cimgs, peaks = clean_segment_batch(
            fimgs,
            mask,
            _radius,
            c.batch,
            pbar and pbar(desc=f"{stage}footprint clean"),
        )
        tifffile.imwrite(f"footprint{stage}.tif", cimgs.max(axis=0))
        pd.DataFrame(peaks._asdict()).to_csv(f"peaks{stage}.csv")
        cimgs = cimgs[peaks.r > 0]
        if mask is None:
            cimgs = cimgs.reshape(-1, h * w)
        else:
            cimgs = cimgs[:, mask]
        return cimgs
    factor = autoload(prepare, cfg, cfg.spatial.prepare, stage)
    footprint = autoload(optimize, cfg, cfg.spatial.optimize, stage)
    return autoload(clean, cfg, cfg.clean, stage)
