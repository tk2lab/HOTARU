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


def rev(index):
    rev_index = np.full_like(index, index.size)
    for i, j in enumerate(index):
        rev_index[j] = i
    return rev_index


def spatial(data, oldx, stats, y1, y2, model, env, clip, prepare, optimize, step):
    logger.info(
        "spatial: %s %d %s %s %s",
        data.shape,
        stats.shape[0],
        oldx.shape,
        y1.shape,
        y2.shape,
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
    logger.info(
        "%d %d\n%s",
        index.size,
        np.count_nonzero(np.sort(index) != np.arange(index.size)),
        index,
    )
    return x[rev(index)]


def temporal(data, y, stats, model, env, clip, prepare, optimize, step):
    logger.info(
        "temporal: %s %s %d %d",
        data.shape,
        y.shape,
        stats.shape[0],
        np.count_nonzero(stats.kind != "remove"),
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
