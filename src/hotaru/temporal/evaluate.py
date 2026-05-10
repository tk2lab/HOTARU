from logging import getLogger

import numpy as np
import pandas as pd

from ..utils.math import calc_mah
from ..utils.math import calc_sn
from ..utils.math import robust_zscore
from .dynamics import get_dynamics
from .dynamics import get_rdynamics

logger = getLogger(__name__)


def evaluate(stats, spikes, bg):
    cond_cell = stats.kind == 'cell'
    cond_bg = stats.kind == 'background'
    cmax = spikes.max(axis=1)

    active_cell = cmax > 0
    cond_remove_cell = np.zeros_like(cond_cell)
    cond_remove_cell[cond_cell] = ~active_cell
    cond_active_cell = np.zeros_like(cond_cell)
    cond_active_cell[cond_cell] = active_cell

    spikes = spikes[active_cell]
    cmax = cmax[active_cell]

    csd = spikes.mean(axis=1) / cmax
    cnonzero = np.count_nonzero(spikes > 0, axis=1)
    csn = calc_sn(spikes)
    crsn = 1 / csn
    czrsn = robust_zscore(crsn)
    x = stats.loc[cond_active_cell, 'firmness'].to_numpy()
    y = crsn
    chypot = np.hypot(robust_zscore(x), robust_zscore(y))
    cmah = calc_mah(x, y)

    bmax = np.abs(bg).max(axis=1)
    bsn = np.array([bi.max() / (1.4826 * np.median(np.abs(bi[bi != 0]))) for bi in bg])

    stats.loc[cond_remove_cell, 'kind'] = 'remove'

    stats['spkid'] = -1
    stats.loc[cond_active_cell, 'spkid'] = np.arange(spikes.shape[0])
    stats['signal'] = None
    stats.loc[cond_active_cell, 'signal'] = cmax
    stats['udense'] = None
    stats.loc[cond_active_cell, 'udense'] = csd
    stats['unz'] = -1
    stats.loc[cond_active_cell, 'unz'] = cnonzero
    stats['snratio'] = None
    stats.loc[cond_active_cell, 'snratio'] = csn
    stats['rsn'] = None
    stats.loc[cond_active_cell, 'rsn'] = crsn
    stats['zrsn'] = None
    stats.loc[cond_active_cell, 'zrsn'] = czrsn
    stats['hypot'] = None
    stats.loc[cond_active_cell, 'hypot'] = chypot
    stats['mah'] = None
    stats.loc[cond_active_cell, 'mah'] = cmah

    stats['bgid'] = -1
    stats.loc[cond_bg, 'bgid'] = np.arange(bg.shape[0])
    stats['bmax'] = None
    stats.loc[cond_bg, 'bmax'] = bmax
    stats['bsparse'] = None
    stats.loc[cond_bg, 'bsparse'] = bsn

    labels = [
        'kind',
        'old_kind',
        'segid',
        'spkid',
        'bgid',
        'y',
        'x',
        'radius',
        'aratio',
        'firmness',
        'pos_move',
        'min_dist_id',
        'min_dist',
        'max_dup_id',
        'max_dup',
        'asum',
        'area',
        'signal',
        'udense',
        'unz',
        'rsn',
        'zrsn',
        'hypot',
        'mah',
        'bmax',
        'bsparse',
    ]
    return stats[[k for k in labels if k in stats.columns]], spikes, bg


def fix_kind(stats, spikes, bg, dynamics, bg_type='bg', thr_bg=None, thr_cell=None):
    if thr_bg is None:
        thr_bg = {}
    if thr_cell is None:
        thr_cell = {}

    fdynamics = get_dynamics(dynamics)
    rdynamics = get_rdynamics(dynamics)
    cell_df = stats.query("kind == 'cell'")
    spikes = spikes[cell_df.spkid.to_numpy()]
    bg_df = stats.query("kind == 'background'")
    logger.info('thr %s %s', thr_bg, thr_cell)
    with np.printoptions(precision=3, suppress=True):
        bins = [-np.inf, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf]
        hist, bins = np.histogram(cell_df.zrsn, bins=bins)
        for s, e, c in zip(bins[:-1], bins[1:], hist, strict=False):
            logger.info('[%f %f): %d', s, e, c)
        bins = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, np.inf]
        hist, bins = np.histogram(cell_df.firmness, bins=bins)
        for s, e, c in zip(bins[:-1], bins[1:], hist, strict=False):
            logger.info('[%f %f): %d', s, e, c)

    if bg_type == 'remove':
        remove_mask = np.zeros(cell_df.shape[0], bool)
        for k, thr in thr_bg.items():
            if k in ['rsn', 'zrsn', 'hypot', 'mah']:
                remove_mask |= cell_df[k] > thr
            else:
                remove_mask |= cell_df[k] < thr
        spikes = spikes[~remove_mask]
        cell_df = cell_df.loc[~remove_mask].copy()
        cell_df['spkid'] = np.arange(cell_df.shape[0])

        stats = cell_df
    elif bg_type == 'bg':
        to_bg_mask = np.zeros(cell_df.shape[0], bool)
        for k, thr in thr_bg.items():
            if k in ['rsn', 'zrsn', 'hypot', 'mah']:
                to_bg_mask |= cell_df[k] > thr
            else:
                to_bg_mask |= cell_df[k] < thr

        if len(thr_cell) == 0:
            to_cell_mask = np.zeros(bg_df.shape[0], bool)
        else:
            to_cell_mask = np.ones(bg_df.shape[0], bool)
            for k, thr in thr_cell.items():
                to_cell_mask &= bg_df[k] > thr

        spikes, to_bg = spikes[~to_bg_mask], spikes[to_bg_mask]
        cell_df, to_bg_df = (
            cell_df.loc[~to_bg_mask].copy(),
            cell_df.loc[to_bg_mask].copy(),
        )
        bg, to_cell = bg[~to_cell_mask], bg[to_cell_mask]
        bg_df, to_cell_df = (
            bg_df.loc[~to_cell_mask].copy(),
            bg_df.loc[to_cell_mask].copy(),
        )
        to_bg = np.array(fdynamics(to_bg))
        to_cell = np.array(rdynamics(to_cell))

        spikes = np.concatenate([spikes, to_cell], axis=0)
        bg = np.concatenate([bg, to_bg], axis=0)

        nc = cell_df.shape[0]
        nb = bg_df.shape[0]
        nn = to_bg_df.shape[0]
        nm = to_cell_df.shape[0]

        cell_df['spkid'] = np.arange(cell_df.shape[0])
        bg_df['bgid'] = np.arange(bg_df.shape[0])

        to_bg_df['kind'] = 'background'
        to_bg_df['spkid'] = -1
        to_bg_df['bgid'] = np.arange(nb, nb + nn, dtype=np.int32)

        to_cell_df['kind'] = 'cell'
        to_cell_df['spkid'] = np.arange(nc, nc + nm, dtype=np.int32)
        to_cell_df['bgid'] = -1

        stats = pd.concat([cell_df, to_cell_df, bg_df, to_bg_df], axis=0)
    else:
        raise ValueError()
    return stats, spikes, bg
