from logging import getLogger

import numpy as np
import pandas as pd

from ..io import (
    save,
    try_load,
)
from ..train import TemporalModel
from ..utils import (
    get_clip,
    get_xla_stats,
)
from .common import (
    get_force,
    get_data,
    get_files,
    rev_index,
)

logger = getLogger(__name__)


def temporal(cfg, stage, force=False):
    force = get_force(cfg, "temporal", stage)
    statsfile, flagfile = get_files(cfg, "evaluate", stage)
    if force or not statsfile.exists() or not flagfile.exists():
        if stage == 0:
            footprints, stats = try_load(get_files(cfg, "make", stage))
        else:
            footprints, stats = try_load(get_files(cfg, "clean", stage))
        spikefile, bgfile = get_files(cfg, "temporal", stage)
        if force or not spikefile.exists() or not bgfile.exists():
            logger.info(f"exec temporal ({stage})")
            data = get_data(cfg)
            stats = stats.query("kind != 'remove'")
            logger.info("%s", get_xla_stats())
            model = TemporalModel(
                data,
                footprints,
                stats,
                cfg.model.dynamics,
                cfg.model.penalty,
            )
            clips = get_clip(data.shape, cfg.cmd.temporal.clip)
            out = []
            for clip in clips:
                model.prepare(clip, **cfg.cmd.temporal.prepare)
                model.fit(**cfg.cmd.temporal.step)
                logger.info("%s", get_xla_stats())
                out.append(model.get_x())
            index1, index2, x1, x2 = (np.concatenate(v, axis=0) for v in zip(*out))
            spikes = np.array(x1[rev_index(index1)])
            bg = np.array(x2[rev_index(index2)])
            save((spikefile, bgfile), (spikes, bg))
            logger.info(f"saved temporal ({stage})")
        else:
            logger.info(f"load temporal ({stage})")
            spikes, bg = try_load((spikefile, bgfile))
        logger.info(f"exec temporal stats ({stage})")
        cell = stats.query("kind=='cell'").index
        sm = spikes.max(axis=1)
        sd = spikes.mean(axis=1) / sm
        stats["umax"] = pd.Series(sm, index=cell)
        stats["udense"] = pd.Series(sd, index=cell)

        background = stats.query("kind=='background'").index
        bmax = np.abs(bg).max(axis=1)
        bgvar = bg - np.median(bg, axis=1, keepdims=True)
        bstd = 1.4826 * np.maximum(np.median(np.abs(bgvar), axis=1), 1e-10)
        stats["bmax"] = pd.Series(bmax, index=background)
        stats["bsparse"] = pd.Series(bmax / bstd, index=background)
        save((statsfile, flagfile), (stats, "updated!"))
        logger.info(f"saved temporal stats ({stage})")
