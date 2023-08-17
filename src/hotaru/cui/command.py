from logging import getLogger

import numpy as np
import pandas as pd

from ..footprint import (
    clean,
    make_footprints,
    reduce_peaks,
)
from ..train import (
    SpatialModel,
    TemporalModel,
)
from ..utils import get_clip

logger = getLogger(__name__)


def make(data, findval, env, cfg):
    peaks = reduce_peaks(findval, cfg.select, **cfg.cmd.reduce)
    footprints = make_footprints(data, peaks, env, **cfg.cmd.make)
    peaks["sum"] = footprints.sum(axis=(1, 2))
    peaks["area"] = np.count_nonzero(footprints > 0, axis=(1, 2))
    nk = np.count_nonzero(peaks.kind == "cell")
    nb = np.count_nonzero(peaks.kind == "background")
    logger.info("num cell/bg: %d/%d", nk, nb)
    return footprints, peaks


def rev(index):
    rev_index = np.full_like(index, index.size)
    for i, j in enumerate(index):
        rev_index[j] = i
    return rev_index


def spatial(data, oldx, stats, y1, y2, model, env, clip, prepare, optimize, step):
    stats = stats.query("kind != 'remove'")
    logger.info(
        "spatial: data=%s cell=%d background=%d",
        data.imgs.shape,
        y1.shape[0],
        y2.shape[0],
    )
    target = SpatialModel(data, oldx, stats, y1, y2, **model, env=env)

    clip = get_clip(data.shape, clip)

    out = []
    for cl in clip:
        target.prepare(cl, **prepare)
        optimizer = target.optimizer(**optimize)
        x1, x2 = target.initial_data()
        x1, x2 = optimizer.fit((x1, x2), **step)
        out.append(target.finalize(x1, x2))
    index, x = (np.concatenate(v, axis=0) for v in zip(*out))
    logger.debug(
        "%d %d\n%s",
        index.size,
        np.count_nonzero(np.sort(index) != np.arange(index.size)),
        index,
    )
    return x[rev(index)]


def temporal(data, y, stats, model, env, clip, prepare, optimize, step):
    logger.info(
        "temporal: data=%s cell=%d background=%d",
        data.shape,
        np.count_nonzero(stats.kind == "cell"),
        np.count_nonzero(stats.kind == "background"),
    )
    stats = stats[stats.kind != "remove"]
    target = TemporalModel(data, y, stats, **model, env=env)

    clip = get_clip(data.shape, clip)
    out = []
    for cl in clip:
        target.prepare(cl, **prepare)
        optimizer = target.optimizer(**optimize)
        x1, x2 = target.initial_data()
        x1, x2 = optimizer.fit((x1, x2), **step)
        out.append(target.finalize(x1, x2))
    index1, index2, x1, x2 = (np.concatenate(v, axis=0) for v in zip(*out))
    return np.array(x1[rev(index1)]), np.array(x2[rev(index2)])


def temporal_eval(spikes, bg, peaks):
    cell = peaks.query("kind=='cell'").index
    sm = spikes.max(axis=1)
    sd = spikes.mean(axis=1) / sm
    peaks["umax"] = pd.Series(sm, index=cell)
    peaks["udense"] = pd.Series(sd, index=cell)

    background = peaks.query("kind=='background'").index
    bmax = np.abs(bg).max(axis=1)
    bgvar = bg - np.median(bg, axis=1, keepdims=True)
    bstd = np.maximum(np.median(np.abs(bgvar), axis=1), 1e-10)
    peaks["bmax"] = pd.Series(bmax, index=background)
    peaks["bsparse"] = pd.Series(bmax / bstd, index=background)
    return peaks
