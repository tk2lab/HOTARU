from logging import getLogger
from pathlib import Path

import numpy as np

from ..io import (
    apply_mask,
    load_imgs,
    try_load,
)
from ..utils import Data

logger = getLogger(__name__)


def get_force(cfg, name, stage):
    print(name, stage)
    force_dict = dict(
        normalize=["normalize", "init", "temporal", "spatial"],
        init=["init", "temporal", "spatial"],
        temporal=["temporal", "spatial"],
        spatial=["spatial"],
    )
    force = (stage > cfg.force_from[0]) or (
        (stage == cfg.force_from[0]) and (name in force_dict[cfg.force_from[1]])
    )
    return force


def get_files(cfg, name, stage):
    odir = Path(cfg.outputs.dir)
    path = cfg.outputs[name]
    fdir = odir / path.dir
    files = [fdir / file.format(stage=stage) for file in path.files]
    return files


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


def finish(cfg, stage):
    stats, _ = try_load(get_files(cfg, "evaluate", stage))
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

    logger.info("cell: %d\n%s", cell.shape[0], cell.head())
    if bg.shape[0] > 0:
        logger.info("background: %d\n%s", bg.shape[0], bg.head())
    if removed.shape[0] > 0:
        logger.info("removed: %d\n%s", removed.shape[0], removed.head())

    return (stage > 0) and removed.shape[0] == 0
