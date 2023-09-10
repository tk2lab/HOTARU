from logging import getLogger

import numpy as np
import pandas as pd

from .dynamics import get_dynamics

logger = getLogger(__name__)


def evaluate(stats, spikes, bg):
    print(
        np.count_nonzero(stats.kind == "cell"),
        np.count_nonzero(stats.kind == "background"),
    )
    print(spikes.shape, bg.shape)
    sm = spikes.max(axis=1)
    sd = spikes.mean(axis=1) / sm
    nonzero = np.count_nonzero(spikes > 0, axis=1)
    sn = np.array([si.max() / (1.4826 * np.median(si[si > 0])) for si in spikes])

    bmax = np.abs(bg).max(axis=1)
    bsn = np.array([bi.max() / (1.4826 * np.median(np.abs(bi[bi != 0]))) for bi in bg])

    ci = stats.kind == "cell"
    stats["spkid"] = -1
    stats.loc[ci, "spkid"] = list(range(np.count_nonzero(ci)))
    stats["signal"] = None
    stats.loc[ci, "signal"] = sm
    stats["udense"] = None
    stats.loc[ci, "udense"] = sd
    stats["unz"] = -1
    stats.loc[ci, "unz"] = nonzero
    stats["snratio"] = None
    stats.loc[ci, "snratio"] = sn
    stats["rsn"] = 1 / stats.snratio

    bi = stats.kind == "background"
    stats["bgid"] = -1
    stats.loc[bi, "bgid"] = list(range(np.count_nonzero(bi)))
    stats["bmax"] = None
    stats.loc[bi, "bmax"] = bmax
    stats["bsparse"] = None
    stats.loc[bi, "bsparse"] = bsn

    labels = [
        "kind",
        "segid",
        "spkid",
        "bgid",
        "y",
        "x",
        "radius",
        "firmness",
        "pos_move",
        "min_dist_id",
        "min_dist",
        "max_dup_id",
        "max_dup",
        "asum",
        "area",
        "signal",
        "udense",
        "unz",
        "rsn",
        "bmax",
        "bsparse",
    ]
    return stats[[k for k in labels if k in stats.columns]]


def fix_kind(stats, footprints, spikes, bg, dynamics, thr_bg_rsn):
    dynamics = get_dynamics(dynamics)
    cell_df = stats.query("kind == 'cell'")
    rsn = cell_df.rsn.to_numpy()
    med = np.median(rsn)
    std = 1.4826 * np.median(np.abs(rsn - med))
    zrsn = (rsn - med) / std
    logger.info("thr %f", thr_bg_rsn)
    with np.printoptions(precision=3, suppress=True):
        bins = [-np.inf, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf]
        hist, bins = np.histogram(zrsn, bins=bins)
        for s, e, c in zip(bins[:-1], bins[1:], hist):
            logger.info("[%f %f): %d", s, e, c)

    to_bg_mask = zrsn > thr_bg_rsn

    spikes, to_bg = spikes[~to_bg_mask], spikes[to_bg_mask]
    cell_df, to_bg_df = cell_df.loc[~to_bg_mask], cell_df.loc[to_bg_mask].copy()
    to_bg = np.array(dynamics(to_bg))
    bg = np.concatenate([bg, to_bg], axis=0)

    bg_df = stats.query("kind == 'background'")
    nb = bg_df.shape[0]
    nn = np.count_nonzero(to_bg_mask)
    cell_df["spkid"] = np.arange(cell_df.shape[0])
    to_bg_df["kind"] = "background"
    to_bg_df["spkid"] = -1
    to_bg_df["bgid"] = np.arange(nb, nb + nn, dtype=np.int32)
    stats = pd.concat([cell_df, bg_df, to_bg_df], axis=0)
    return stats, spikes, bg
