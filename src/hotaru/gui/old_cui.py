from collections import namedtuple
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from ..filter.stats import calc_stats
from ..footprint.clean import clean_segment_batch
from ..footprint.find import find_peaks_batch
from ..footprint.make import make_segment_batch
from ..footprint.reduce import reduce_peaks_block
from ..io.image import load_imgs
from ..io.saver import (
    load,
    save,
)
from ..proxmodel.regularizer import MaxNormNonNegativeL1
from ..train.dynamics import SpikeToCalcium
from ..train.loss import (
    gen_factor,
    gen_optimizer,
)
from .progress import (
    SilentProgress,
    TQDMProgress,
)

Data = namedtuple("Data", "imgs mask avgx avgt std0")
Stats = namedtuple("Stats", "imin imax istd icor")
Penalty = namedtuple("Penalty", "la lu bx bt")


def HagetakaCUI(cfg):
    pbar = TQDMProgress()
    nk = -1
    for stage in range(cfg.total_stage + 1):
        old_nk, nk = nk, temporal_optimize(cfg, stage, pbar).shape[0]
        if old_nk == nk:
            break


def autoload(func, cfg, stage=None, pbar=None):
    @wraps(func)
    def wrap(cfg, stage=None, pbar=None):
        if pbar is None:
            pbar = SilentProgress()
        cfg.stage, stage_saved = stage, cfg.stage
        c = cfg.solver
        for k in func.__name__.split("_"):
            c = c.get(k)
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
            save_objs = func(cfg, c, stage, pbar)
            cfg.stage = stage_saved
            if not isinstance(save_objs, tuple):
                save_objs = (save_objs,)
            for obj in save_objs:
                save(file, obj)
            if stage is None:
                c.force = False
            else:
                c.done = stage
        cfg.stage = stage_saved
        return save_objs
    return wrap


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


@autoload
def stats(cfg, c, stage, pbar):
    imgs, mask = load_imgs(cfg.data)
    return calc_stats(imgs, mask, c.batch, pbar.session("stats"))


def imgs_stats(cfg, pbar=None):
    s = stats(cfg, pbar=pbar)
    return Stats(s.imin, s.imax, s.istd, s.icor)


def get_frame(cfg, t, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    s = stats(cfg, pbar=pbar)
    return (imgs[t] - s.avgx - s.avgt[t]) / s.std0


def data(cfg, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    s = stats(cfg, pbar=pbar)
    return Data(imgs, mask, s.avgx, s.avgt, s.std0)


@autoload
def find(cfg, c, stage, pbar):
    return find_peaks_batch(
        data(cfg, pbar=pbar),
        radius(c),
        c.batch,
        pbar.session("find"),
    )


@autoload
def reduce(cfg, c, stage, pbar):
    return reduce_peaks_block(
        find(cfg, pbar=pbar),
        c.rmin,
        c.rmax,
        c.thr,
        c.block_size,
    )


@autoload
def make(cfg, c, stage, pbar):
    return make_segment_batch(
        data(cfg, pbar=pbar),
        reduce(cfg, pbar=pbar),
        c.batch,
        pbar.session("make"),
    )


@autoload
def clean(cfg, c, stage, pbar):
    imgs, mask = load_imgs(cfg.data)
    val = spatial_optimize(cfg, stage, pbar)
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


def temporal_prepare(cfg, c, stage, pbar):
    return gen_factor(
        "temporal",
        make(cfg, pbar=pbar) if stage == 0 else clean(cfg, stage, pbar)[0],
        data(cfg, pbar=pbar),
        dynamics(cfg),
        penalty(cfg, cfg.temporal.penalty),
        c.batch,
        pbar.session(f"{stage}spike prepare"),
    )


@autoload
def temporal_optimize(cfg, c, stage, pbar):
    optimizer = gen_optimizer(
        "temporal",
        temporal_prepare(cfg, stage, pbar),
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
    return optimizer.val[0]


@autoload
def spatial_prepare(cfg, c, stage, pbar):
    spk = temporal_optimize(cfg, stage - 1, pbar)
    spk /= spk.max(axis=1, keepdims=True)
    return gen_factor(
        "spatial",
        spk,
        data(cfg, pbar=pbar),
        dynamics(cfg),
        penalty(cfg, cfg.penalty),
        c.batch,
        pbar.session(f"{stage}footprint prepare"),
    )


@autoload
def spatial_optimize(cfg, c, stage, pbar):
    optimizer = gen_optimizer(
        "spatial",
        spatial_prepare(cfg, stage, pbar),
        dynamics(cfg),
        penalty(cfg, cfg.penalty),
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
