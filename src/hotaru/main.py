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
from .train.temporal import temporal_optimizer
from .train.spatial import spatial_optimizer
from .proxmodel.regularizer import MaxNormNonNegativeL1


Data = namedtuple("Data", "imgs mask avgx avgt std0")
Stats = namedtuple("Stats", "imin imax istd icor")
Penalty = namedtuple("Penalty", "la lu bx bt")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data(cfg, tqdm)
    find(cfg, tqdm)
    reduce(cfg)
    make(cfg, tqdm)
    spike(cfg, 0, tqdm)
    for stage in range(1, cfg.stage + 1):
        footprint(cfg, stage, tqdm)
        #print(f"clean {stage}")
        #clean(cfg, stage, tqdm)
        spike(cfg, stage, tqdm)


def autoload(func, cfg, label, stage=None):
    c = cfg[label]
    file = Path(cfg.outdir) / c.outfile
    if (not c.force or (stage and stage <= c.done)) and file.exists():
        obj = load(file)
    else:
        stage_saved = cfg.stage
        cfg.stage = stage
        obj = func(c)
        cfg.stage = stage_saved
        save(file, obj)
        if stage:
            c.done = stage
        else:
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
        lambda c: calc_stats(imgs, mask, c.batch, pbar),
        cfg, "stats",
    )
    return Data(imgs, mask, stats.avgx, stats.avgt, stats.std0)


def stats(cfg, pbar=None):
    imgs, mask = load_imgs(cfg.data)
    stats = autoload(
        lambda c: calc_stats(imgs, mask, c.batch, pbar),
        cfg, "stats",
    )
    return Stats(stats.imin, stats.imax, stats.istd, stats.icor)


def find(cfg, pbar=None):
    return autoload(
        lambda c: find_peaks_batch(data(cfg), radius(c), c.batch, pbar),
        cfg, "find",
    )


def reduce(cfg, pbar=None):
    return autoload(
        lambda c: reduce_peaks_block(find(cfg), c.rmin, c.rmax, c.thr, c.block_size),
        cfg, "reduce",
    )


def make(cfg, pbar=None):
    return autoload(
        lambda c: make_segment_batch(data(cfg), reduce(cfg), c.batch, pbar),
        cfg, "make",
    )


def clean(cfg, stage, pbar=None):
    return footprint(cfg, stage)
    return autoload(
        lambda c: clean_segment_batch(footprint(cfg, stage), radius(c), c.batch, pbar),
        cfg, "clean", stage,
    )


def spike(cfg, stage, pbar=None):
    def func(c):
        print(f"spike {stage}")
        optimizer = temporal_optimizer(
            make(cfg) if stage == 0 else clean(cfg, stage),
            data(cfg),
            dynamics(cfg),
            penalty(cfg),
            c.batch,
            pbar,
        )
        optimizer.set_params(c.lr, c.scale, c.reset)
        optimizer.fit(c.n_iter, pbar)
        return optimizer.val[0]
    return autoload(func, cfg, "temporal", stage)


def footprint(cfg, stage, pbar=None):
    def func(c):
        print(f"footprint {stage}")
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
    return autoload(func, cfg, "spatial", stage)
