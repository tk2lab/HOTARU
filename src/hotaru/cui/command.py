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
    peaks = reduce_peaks(findval, cfg.density, **cfg.cmd.reduce)
    footprints = make_footprints(data, peaks, env, **cfg.cmd.make)
    peaks["sum"] = footprints.sum(axis=(1, 2))
    peaks["area"] = np.count_nonzero(footprints > 0, axis=(1, 2))
    nk = np.count_nonzero(peaks.kind == "cell")
    nb = np.count_nonzero(peaks.kind == "background")
    logger.info("num cell/bg: %d/%d", nk, nb)
    return footprints, peaks


def spatial(data, oldx, stats, y1, y2, model, env, clip, prepare, optimize, step):
    logger.info("spatial:")
    target = SpatialModel(data, oldx, stats, y1, y2, **model, env=env)

    clip = get_clip(data.shape, clip)
    out = []
    for cl in clip:
        target.prepare(cl, **prepare)
        optimizer = target.optimizer(**optimize)
        x1, x2 = target.initial_data()
        x1, x2 = optimizer.fit((x1, x2), **step)
        index, x = target.finalize(x1, x2)
        out.append((index, x))
    index, x = zip(*out)
    index = np.concatenate(index, axis=0)
    x = np.concatenate(x, axis=0)

    rev_index = np.empty_like(index)
    for i, j in enumerate(index):
        rev_index[j] = i
    return x[rev_index]


def temporal(data, y, peaks, model, env, prepare, optimize, step):
    logger.info("temporal:")
    target = TemporalModel(data, y, peaks, **model, env=env)
    target.prepare(**prepare)
    optimizer = target.optimizer(**optimize)
    x1, x2 = target.initial_data()
    x1, x2 = optimizer.fit((x1, x2), **step)
    return np.array(x1), np.array(x2)


def spatial_and_clean(data, old_footprints, old_peaks, spikes, background, cfg):
    old_peaks = old_peaks[old_peaks.kind != "remove"]
    segments = spatial(
        data,
        old_footprints,
        old_peaks,
        spikes,
        background,
        cfg.model,
        cfg.env,
        **cfg.cmd.spatial,
    )
    uid = old_peaks.uid.to_numpy()
    footprints, peaks = clean(
        uid,
        segments,
        cfg.radius,
        cfg.density,
        cfg.env,
        **cfg.cmd.clean,
    )
    return footprints, peaks


def temporal_and_eval(data, footprints, peaks, cfg):
    spikes, background = temporal(
        data,
        footprints,
        peaks,
        cfg.model,
        cfg.env,
        **cfg.cmd.temporal,
    )
    peaks["umax"] = pd.Series(
        spikes.max(axis=1), index=peaks.query("kind=='cell'").index
    )
    peaks["unum"] = pd.Series(
        np.count_nonzero(spikes, axis=1), index=peaks.query("kind=='cell'").index
    )
    peaks["bmax"] = pd.Series(
        np.abs(background).max(axis=1), index=peaks.query("kind=='background'").index
    )
    peaks["bmean"] = pd.Series(
        background.mean(axis=1), index=peaks.query("kind=='background'").index
    )
    return spikes, background, peaks
