import numpy as np
import pandas as pd

from .dynamics import get_dynamics


def evaluate(stats, spikes, bg):
    sm = spikes.max(axis=1)
    sd = spikes.mean(axis=1) / sm
    sn = np.array([si.max() / (1.4826 * np.median(si[si > 0])) for si in spikes])

    bmax = np.abs(bg).max(axis=1)
    bsn = np.array(
        [bi.max() / (1.4826 * np.median(np.abs(bi[bi != 0]))) for bi in bg]
    )

    if "signal" in stats.columns:
        stats["old_signal"] = stats.signal
    if "rsn" in stats.columns:
        stats["old_rsn"] = stats.rsn

    ci = stats.query("kind == 'cell'").index
    stats["spkid"] = pd.Series(list(range(ci.shape[0])), index=ci)
    stats["signal"] = pd.Series(sm, index=ci)
    stats["udense"] = pd.Series(sd, index=ci)
    stats["snratio"] = pd.Series(sn, index=ci)
    stats["rsn"] = pd.Series(1 / sn, index=ci)

    bi = stats.query("kind == 'background'").index
    stats["bgid"] = pd.Series(list(range(bi.shape[0])), index=bi)
    stats["bmax"] = pd.Series(bmax, index=bi)
    stats["bsparse"] = pd.Series(bsn, index=bi)

    return stats


def fix_kind(stats, footprints, spikes, bg, dynamics, thr_bg_rsn):
    dynamics = get_dynamics(dynamics)
    med = np.median(stats.rsn)
    std = 1.4826 * np.median(np.abs(stats.rsn - med))
    zrsn = (stats.rsn - med) / std
    nb = stats.query("kind=='background'").shape[0]
    cell_mask = stats.kind == "cell"
    to_bg_mask = cell_mask & (zrsn > thr_bg_rsn)
    nn = np.count_nonzero(to_bg_mask)
    spikes, to_bg = spikes[~to_bg_mask[cell_mask]], spikes[to_bg_mask[cell_mask]]
    to_bg = np.array(dynamics(to_bg))
    bg = np.concatenate([bg, to_bg], axis=0)
    stats.loc[to_bg_mask, "kind"] = "background"
    stats.loc[to_bg_mask, "bgid"] = np.arange(nb, nb + nn, dtype=np.int32)
    return stats, spikes, bg
