from logging import getLogger

import numpy as np
import pandas as pd

from ..footprint import (
    clean,
    make_footprints,
    reduce_peaks,
)
from ..train.model import (
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


def spatial(data, oldx, stats, y1, y2, cfg):
    logger.info(
        "spatial: data=%s cell=%d background=%d",
        data.imgs.shape,
        y1.shape[0],
        y2.shape[0],
    )
    stats = stats.query("kind != 'remove'")
    model = SpatialModel(data, oldx, stats, y1, y2, **cfg.model, env=cfg.env)

    clip = get_clip(data.shape, cfg.cmd.spatial.clip)
    out = []
    for cl in clip:
        model.prepare(cl, **cfg.cmd.spatial.prepare)
        model.fit(**cfg.cmd.spatial.step)
        out.append(model.get_x())
    index, x = (np.concatenate(v, axis=0) for v in zip(*out))
    logger.debug(
        "%d %d\n%s",
        index.size,
        np.count_nonzero(np.sort(index) != np.arange(index.size)),
        index,
    )
    segments = x[rev(index)]
    return clean(stats, segments, cfg.radius, cfg.select, cfg.env, **cfg.cmd.clean)


def temporal(data, y, stats, cfg):
    logger.info(
        "temporal: data=%s cell=%d background=%d",
        data.shape,
        np.count_nonzero(stats.kind == "cell"),
        np.count_nonzero(stats.kind == "background"),
    )
    stats = stats[stats.kind != "remove"]
    model = TemporalModel(data, y, stats, **cfg.model, env=cfg.env)

    clip = get_clip(data.shape, cfg.cmd.temporal.clip)
    out = []
    for cl in clip:
        model.prepare(cl, **cfg.cmd.temporal.prepare)
        model.fit(**cfg.cmd.temporal.step)
        out.append(model.get_x())
    index1, index2, x1, x2 = (np.concatenate(v, axis=0) for v in zip(*out))
    return np.array(x1[rev(index1)]), np.array(x2[rev(index2)])


def eval_spikes(spikes, bg, peaks):
    cell = peaks.query("kind=='cell'").index
    sm = spikes.max(axis=1)
    sd = spikes.mean(axis=1) / sm
    peaks["umax"] = pd.Series(sm, index=cell)
    peaks["udense"] = pd.Series(sd, index=cell)

    background = peaks.query("kind=='background'").index
    bmax = np.abs(bg).max(axis=1)
    bgvar = bg - np.median(bg, axis=1, keepdims=True)
    bstd = 1.4826 * np.maximum(np.median(np.abs(bgvar), axis=1), 1e-10)
    peaks["bmax"] = pd.Series(bmax, index=background)
    peaks["bsparse"] = pd.Series(bmax / bstd, index=background)
    return peaks
