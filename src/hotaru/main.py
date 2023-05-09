from collections import namedtuple
from pathlib import Path

import hydra
import jax
import numpy as np
import pandas as pd
from tqdm import tqdm

from .io.logger import logger
from .io.saver import (
    load,
    save,
)
from .io.image import load_imgs
from .jax.filter.stats import (
    Stats,
    calc_stats,
)
from .jax.footprint.find import (
    PeakVal,
    find_peaks_batch,
)
from .jax.footprint.reduce import reduce_peaks_block
from .jax.footprint.make import make_segment_batch
from .jax.footprint.clean import clean_segment_batch
from .jax.train.dynamics import SpikeToCalcium
from .jax.train.temporal import temporal_optimizer
from .jax.train.spatial import spatial_optimizer
from .jax.proxmodel.regularizer import MaxNormNonNegativeL1


Data = namedtuple("Data", "imgs mask avgx avgt std0")
Stats = namedtuple("Stats", "imin imax istd icor")
Penalty = namedtuple("Penalty", "la lu bx bt")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data(cfg, tqdm)
    find(cfg, tqdm)
    reduce(cfg)
    make(cfg, tqdm)
    stage = cfg.stage
    print("spike 0")
    spike(cfg, 0, tqdm)
    for stage in range(1, stage + 1):
        print(f"footprint {stage}")
        footprint(cfg, stage, tqdm)
        #print(f"clean {stage}")
        #clean(cfg, stage, tqdm)
        print(f"spike {stage}")
        spike(cfg, stage, tqdm)


def autoload(cfg, label, func):
    c = cfg[label]
    file = Path(cfg.outdir) / c.outfile
    if not c.force and file.exists():
        obj = load(file)
    else:
        obj = func(c)
        save(file, obj)
        if not c.loop:
            c.force = False
    return obj


def radius(c):
    return np.geomspace(c.rmin, c.rmax, c.rnum)


def dynamics(cfg):
    c = cfg.dynamics
    if c.kind == "double_exp":
        return SpikeToCalcium.double_exp(c.tau1, c.tau2, c.duration, cfg.data.hz)


def penalty(cfg):
    c = cfg.penalty
    la = MaxNormNonNegativeL1(c.la)
    lu = MaxNormNonNegativeL1(c.lu)
    return Penalty(la, lu, c.bx, c.bt)


def data(cfg, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    stats = autoload(
        cfg, "stats",
        lambda c: calc_stats(imgs, mask, c.batch, pbar),
    )
    return Data(imgs, mask, stats.avgx, stats.avgt, stats.std0)


def stats(cfg, pbar=None):
    path = Path(cfg.data.dir)
    imgs = load_imgs(path / cfg.data.imgs)
    mask = load_mask(path / cfg.data.mask, imgs)
    stats = autoload(
        cfg, "stats",
        lambda c: calc_stats(imgs, mask, c.batch, pbar),
    )
    return Stats(stats.imin, stats.imax, stats.istd, stats.icor)


def find(cfg, pbar=None):
    return autoload(
        cfg, "find",
        lambda c: find_peaks_batch(data(cfg), radius(c), c.batch, pbar),
    )


def reduce(cfg, pbar=None):
    return autoload(
        cfg, "reduce",
        lambda c: reduce_peaks_block(find(cfg), c.rmin, c.rmax, c.thr, c.block_size),
    )


def make(cfg, pbar=None):
    return autoload(
        cfg, "make",
        lambda c: make_segment_batch(data(cfg), reduce(cfg), c.batch, pbar)
    )


def clean(cfg, stage, pbar=None):
    return footprint(cfg, stage)
    cfg.stage = stage
    return autoload(
        cfg, "clean",
        lambda c: clean_segment_batch(footprint(cfg, stage), radius(c), c.batch, pbar),
    )


def spike(cfg, stage, pbar=None):
    def func(c):
        footprint = make(cfg) if stage == 0 else clean(cfg, stage)
        optimizer = temporal_optimizer(
            footprint,
            data(cfg),
            dynamics(cfg),
            penalty(cfg),
            c.batch,
            pbar,
        )
        optimizer.set_params(c.lr, c.scale, c.reset)
        optimizer.fit(c.n_iter, pbar)
        return optimizer.val[0]
    cfg.stage = stage
    return autoload(cfg, "temporal", func)


def footprint(cfg, stage, pbar=None):
    def func(c):
        optimizer = spatial_optimizer(
            spike(cfg, stage - 1),
            data(cfg),
            dynamics(cfg),
            penalty(cfg),
            c.batch,
            pbar,
        )
        optimizer.set_params(c.lr, c.scale, c.reset)
        optimizer.fit(c.n_iter, pbar)
        return optimizer.val[0]
    cfg.stage = stage
    return autoload(cfg, "spatial", func)
