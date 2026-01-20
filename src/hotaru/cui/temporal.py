from logging import getLogger

import numpy as np
import pandas as pd

from ..io import (
    save,
    try_load,
)
from ..train import (
    TemporalModel,
    get_penalty,
)
from ..utils import (
    get_clip,
    get_xla_stats,
)
from ..spike import (
    evaluate,
)
from .common import (
    get_data,
    get_files,
    get_force,
    rev_index,
)

logger = getLogger(__name__)


def temporal(cfg, stage, force=False):
    (statsfile,) = get_files(cfg, "evaluate", stage)
    if (
        get_force(cfg, "evaluate", stage)
        or not statsfile.exists()
    ):
        if stage == 0:
            footprints = try_load(get_files(cfg, "make", stage))
            stats = try_load(get_files(cfg, "init", stage))
        else:
            stats, footprints = try_load(get_files(cfg, "clean", stage))
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
                get_penalty(cfg.penalty, stage),
            )
            clips = get_clip(data.shape, cfg.cmd.temporal.clip)
            logdfs = []
            out = []
            for i, clip in enumerate(clips):
                model.prepare(clip, **cfg.cmd.temporal.prepare)
                log = model.fit(**cfg.cmd.temporal.step)
                logger.debug("%s", get_xla_stats())

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
                out.append(model.get_x())
            logdf = pd.concat(logdfs, axis=0)
            index1, index2, x1, x2 = (np.concatenate(v, axis=0) for v in zip(*out))
            spikes = np.array(x1[rev_index(index1)])
            bg = np.array(x2[rev_index(index2)])
            if cfg.fix_top:
                nk, nu = spikes.shape
                idx = np.argpartition(spikes, -2, axis=1)
                spikes[np.arange(nk), idx[:, -1]] = spikes[np.arange(nk), idx[:, -2]]
            save((spikefile, bgfile, lossfile), (spikes, bg, logdf))
            logger.info(f"saved temporal ({stage})")
        else:
            logger.info(f"load temporal ({stage})")
            spikes, bg, logdf = try_load((spikefile, bgfile, lossfile))

        logger.info(f"exec temporal stats ({stage})")
        stats = evaluate(stats, spikes, bg)
        save((statsfile,), (stats,))
        logger.info(f"saved temporal stats ({stage})")
