import os
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from ..io import apply_mask
from ..io import load_imgs
from ..io import try_load
from .data import Data

logger = getLogger(__name__)


def set_env(cfg):
    hcfg = HydraConfig.get()
    for k, v in hcfg.job.env_set.items():
        os.environ[k] = v


def get_force(cfg, name, stage):
    force_dict = {
        'normalize': [
            'normalize',
            'find',
            'reduce',
            'make',
            'init',
            'temporal',
        ],
        'find': ['find', 'reduce', 'make', 'init', 'temporal'],
        'reduce': ['reduce', 'make', 'init', 'temporal'],
        'make': ['make', 'init', 'temporal'],
        'init': ['init', 'temporal'],
        'spatial': ['spatial', 'clean', 'temporal'],
        'clean': ['clean', 'temporal'],
        'temporal': ['temporal'],
    }
    force = (stage > cfg.force_from[0]) or (
        (stage == cfg.force_from[0]) and (name in force_dict[cfg.force_from[1]])
    )
    logger.debug('test force: %s %d', name, stage)
    return force


def all_stats(cfg):
    out = []
    for stage in range(1000):
        (_, _, _, stats), flag = load(cfg, 'temporal', stage)
        if flag is None:
            break
        out.append(stats)
    return out


def get_files(cfg, name, stage):
    odir = Path(cfg.outputs.dir)
    path = cfg.outputs[name]
    fdir = odir / path.dir0 if stage == 0 or stage == 1 and name == 'spatial' else odir / path.dir
    files = [fdir / file.format(stage=stage) for file in path.files]
    return files


def load(cfg, name, stage):
    return try_load(get_files(cfg, name, stage))


def get_data(cfg):
    imgs, hz = load_imgs(**cfg.data.imgs)
    imgs, mask = apply_mask(imgs, **cfg.data.mask)
    stats = try_load(get_files(cfg, 'normalize', 0)[0])
    data = Data(imgs, mask, hz, *stats)
    return data


def rev_index(index):
    rev_index = np.full_like(index, index.size)
    for i, j in enumerate(index):
        rev_index[j] = i
    return rev_index


def reduce_log(cfg, stage):
    if stage == 0:
        stats = try_load(get_files(cfg, 'reduce', stage))
    else:
        stats, _ = try_load(get_files(cfg, 'clean', stage))
    cell = stats[stats.kind == 'cell']
    bg = stats[stats.kind == 'background']
    removed = stats[stats.kind == 'remove']

    radius = np.sort(np.unique(stats.radius))
    for r in radius:
        logger.info(
            'radius=%f cell=%d bg=%d remove=%d',
            r,
            (cell.radius == r).sum(),
            (bg.radius == r).sum(),
            (removed.radius == r).sum(),
        )


def print_stats(cfg, stage):
    _, _, _, stats = try_load(get_files(cfg, 'temporal', stage))

    cell = stats[stats.kind == 'cell']
    rsn = cell.rsn
    med = np.median(rsn)
    std = 1.4826 * np.median(np.abs(rsn - med))
    thr = med + 2 * std
    logger.info('rsn: med=%.3f std=%.3f thr=%.3f', med, std, thr)
    labels = [
        'y',
        'x',
        'radius',
        'firmness',
        'signal',
        # "udense",
        'unz',
        'rsn',
        'zrsn',
    ]
    if stage > 0:
        labels += [
            'min_dist_id',
            'min_dist',
            'max_dup_id',
            'max_dup',
        ]
    with pd.option_context('display.max_rows', None):
        logger.info(
            'cell: %d\n%s',
            cell.shape[0],
            cell[labels].sort_values('signal', ascending=False),
        )

    bg = stats[stats.kind == 'background']
    if bg.shape[0] > 0:
        labels = [
            'y',
            'x',
            'radius',
            'firmness',
            'bmax',
            'bsparse',
        ]
        # if stage > 0:
        #    labels += ["old_radius", "old_udense", "old_bsparse"]
        with pd.option_context('display.max_rows', None):
            logger.info(
                'background: %d\n%s',
                bg.shape[0],
                bg[labels].sort_values('bmax', ascending=False),
            )

    removed = stats[stats.kind == 'remove']
    if removed.shape[0] == 0:
        logger.info('removed: 0')
    else:
        labels = [
            'y',
            'x',
            'radius',
            'firmness',
        ]
        if stage > 0:
            labels += [
                'pos_move',
            ]
        with pd.option_context('display.max_rows', None):
            logger.info(
                'removed: %d\n%s',
                removed.shape[0],
                removed[labels].sort_values('firmness', ascending=False),
            )


def finish(cfg, stage):
    _, _, _, stats = try_load(get_files(cfg, 'temporal', stage))
    removed = stats[stats.kind == 'remove']

    # reduce_log(cfg, stage)
    return (stage > 0) and (removed.shape[0] <= cfg.early_stop)
