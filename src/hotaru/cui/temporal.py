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
    get_data,
    get_files,
    get_force,
    rev_index,
)

logger = getLogger(__name__)


def temporal(cfg, stage, force=False):
    statsfile, flagfile = get_files(cfg, "evaluate", stage)
    if (
        get_force(cfg, "evaluate", stage)
        or not statsfile.exists()
        or not flagfile.exists()
    ):
        if stage == 0:
            footprints = try_load(get_files(cfg, "make", stage))
            stats = try_load(get_files(cfg, "init", stage))
        else:
            footprints, stats = try_load(get_files(cfg, "clean", stage))
        spikefile, bgfile, lossfile = get_files(cfg, "temporal", stage)
        if (
            get_force(cfg, "temporal", stage)
            or not spikefile.exists()
            or not bgfile.exists()
        ):
            logger.info(f"exec temporal ({stage})")
            data = get_data(cfg)
            logger.debug("%s", get_xla_stats())
            model = TemporalModel(
                data,
                stats,
                footprints,
                cfg.dynamics,
                cfg.penalty,
            )
            clips = get_clip(data.shape, cfg.cmd.temporal.clip)
            out = []
            logdfs = []
            for i, clip in enumerate(clips):
                model.prepare(clip, **cfg.cmd.temporal.prepare)
                log = model.fit(**cfg.cmd.temporal.step)
                logger.debug("%s", get_xla_stats())
                out.append(model.get_x())

                loss, sigma = zip(*log)
                df = pd.DataFrame(
                    dict(
                        stage=stage,
                        kind="temporal",
                        div=i,
                        step=np.arange(len(log)),
                        loss=loss,
                        sigma=sigma,
                    ),
                )
                logdfs.append(df)
            logdf = pd.concat(logdfs, axis=0)
            index1, index2, x1, x2 = (np.concatenate(v, axis=0) for v in zip(*out))
            spikes = np.array(x1[rev_index(index1)])
            bg = np.array(x2[rev_index(index2)])
            save((spikefile, bgfile, lossfile), (spikes, bg, logdf))
            logger.info(f"saved temporal ({stage})")
        else:
            logger.info(f"load temporal ({stage})")
            spikes, bg, logdf = try_load((spikefile, bgfile, lossfile))

        logger.info(f"exec temporal stats ({stage})")

        sm = spikes.max(axis=1)
        sd = spikes.mean(axis=1) / sm
        sn = np.array([si.max() / np.median(si[si > 0]) for si in spikes])
        cell = stats.query("kind == 'cell'").index
        stats["spkid"] = pd.Series(np.arange(sm.size), index=cell)
        stats["signal"] = pd.Series(sm, index=cell)
        stats["udense"] = pd.Series(sd, index=cell)
        stats["snratio"] = pd.Series(sn, index=cell)

        bmax = np.abs(bg).max(axis=1)
        bgvar = bg - np.median(bg, axis=1, keepdims=True)
        bstd = 1.4826 * np.maximum(np.median(np.abs(bgvar), axis=1), 1e-10)
        background = stats.query("kind=='background'").index
        stats["bgid"] = pd.Series(np.arange(bmax.size), index=background)
        stats["bmax"] = pd.Series(bmax, index=background)
        stats["bsparse"] = pd.Series(bmax / bstd, index=background)

        save((statsfile, flagfile), (stats, "updated!"))
        logger.info(f"saved temporal stats ({stage})")
