import os
from logging import getLogger
from pathlib import Path

import numpy as np
from hydra.core.hydra_config import HydraConfig

from ..io import (
    apply_mask,
    load_imgs,
    try_load,
)
from .data import Data

logger = getLogger(__name__)


def set_env(cfg):
    hcfg = HydraConfig.get()
    for k, v in hcfg.job.env_set.items():
        os.environ[k] = v


def get_force(cfg, name, stage):
    force_dict = dict(
        normalize=[
            "normalize",
            "find",
            "reduce",
            "make",
            "init",
            "temporal",
            "evaluate",
        ],
        find=["find", "reduce", "make", "init", "temporal", "evaluate"],
        reduce=["reduce", "make", "init", "temporal", "evaluate"],
        make=["make", "init", "temporal", "evaluate"],
        init=["init", "temporal", "evaluate"],
        spatial=["spatial", "clean", "temporal", "evaluate"],
        clean=["clean", "temporal", "evaluate"],
        temporal=["temporal", "evaluate"],
        evaluate=["evaluate"],
    )
    force = (stage > cfg.force_from[0]) or (
        (stage == cfg.force_from[0]) and (name in force_dict[cfg.force_from[1]])
    )
    print("test force: ", name, stage)
    return force


def all_stats(cfg):
    out = []
    for stage in range(1000):
        stats, _ = load(cfg, "evaluate", stage)
        if stats is None:
            break
        out.append(stats)
    return out


def get_files(cfg, name, stage):
    odir = Path(cfg.outputs.dir)
    path = cfg.outputs[name]
    if (stage == 0) or ((stage == 1) and (name == "spatial")):
        fdir = odir / path.dir0
    else:
        fdir = odir / path.dir
    files = [fdir / file.format(stage=stage) for file in path.files]
    return files


def load(cfg, name, stage):
    return try_load(get_files(cfg, name, stage))


def get_data(cfg):
    imgs, hz = load_imgs(**cfg.data.imgs)
    imgs, mask = apply_mask(imgs, **cfg.data.mask)
    stats = try_load(get_files(cfg, "normalize", 0)[0])
    data = Data(imgs, mask, hz, *stats)
    return data


def rev_index(index):
    rev_index = np.full_like(index, index.size)
    for i, j in enumerate(index):
        rev_index[j] = i
    return rev_index


def reduce_log(cfg, stage):
    if stage == 0:
        stats = try_load(get_files(cfg, "reduce", stage))
    else:
        _, stats = try_load(get_files(cfg, "clean", stage))
    cell = stats[stats.kind == "cell"]
    bg = stats[stats.kind == "background"]
    removed = stats[stats.kind == "remove"]

    radius = np.sort(np.unique(stats.radius))
    for r in radius:
        logger.info(
            "radius=%f cell=%d bg=%d remove=%d",
            r,
            (cell.radius == r).sum(),
            (bg.radius == r).sum(),
            (removed.radius == r).sum(),
        )


def finish(cfg, stage):
    stats, _ = try_load(get_files(cfg, "evaluate", stage))
    cell = stats[stats.kind == "cell"]
    bg = stats[stats.kind == "background"]
    removed = stats[stats.kind == "remove"]

    reduce_log(cfg, stage)

    firmness = "intensity" if stage == 0 else "firmness"
    labels = ["uid", "y", "x", "radius", firmness, "signal", "udense"]
    logger.info("cell: %d\n%s", cell.shape[0], cell.head()[labels])
    if bg.shape[0] > 0:
        labels = [
            "uid",
            "y",
            "x",
            "radius",
            firmness,
            "bmax",
            "bsparse",
        ]
        if stage > 0:
            labels += ["old_radius", "old_udense", "old_bsparse"]
        logger.info("background: %d\n%s", bg.shape[0], bg.head()[labels])
    if removed.shape[0] > 0:
        logger.info("removed: %d\n%s", removed.shape[0], removed.head())

    return (stage > 0) and removed.shape[0] == 0
