from logging import getLogger

import numpy as np
import pandas as pd

from ..footprint import clean
from ..io import (
    save,
    try_load,
)
from ..train import (
    SpatialModel,
    get_penalty,
)
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


def spatial(cfg, stage, force=False):
    footprintsfile, statsfile = get_files(cfg, "clean", stage)
    if (
        get_force(cfg, "clean", stage)
        or not footprintsfile.exists
        or not statsfile.exists()
    ):
        stats, flags = try_load(get_files(cfg, "evaluate", stage - 1))
        stats = stats.query("kind != 'remove'").copy()
        segsfile, lossfile = get_files(cfg, "spatial", stage)
        if get_force(cfg, "spatial", stage) or not segsfile.exists():
            logger.info(f"exec spatial ({stage})")
            data = get_data(cfg)
            if stage == 1:
                footprints = try_load(get_files(cfg, "make", stage - 1)[0])
            else:
                footprints = try_load(get_files(cfg, "clean", stage - 1)[0])
            spikes, bg, _ = try_load(get_files(cfg, "temporal", stage - 1))
            logger.debug("%s", get_xla_stats())
            model = SpatialModel(
                data,
                stats,
                footprints,
                spikes,
                bg,
                cfg.dynamics,
                get_penalty(cfg.penalty, stage),
            )
            clips = get_clip(data.shape, cfg.cmd.spatial.clip)
            logdfs = []
            out = []
            for i, clip in enumerate(clips):
                model.prepare(clip, **cfg.cmd.spatial.prepare)
                log = model.fit(**cfg.cmd.spatial.step)
                logger.debug("%s", get_xla_stats())

                loss, sigma = zip(*log)
                df = pd.DataFrame(
                    dict(
                        stage="stage",
                        kind="spatial",
                        div=i,
                        step=np.arange(len(log)),
                        loss=loss,
                        sigma=sigma,
                    )
                )
                logdfs.append(df)
                out.append(model.get_x())
            logdf = pd.concat(logdfs, axis=0)
            index, x = (np.concatenate(v, axis=0) for v in zip(*out))
            logger.debug(
                "%d %d\n%s",
                index.size,
                np.count_nonzero(np.sort(index) != np.arange(index.size)),
                index,
            )

            segments = x[rev_index(index)]
            save((segsfile, lossfile), (segments, logdf))
            logger.info(f"saved spatial ({stage})")
        else:
            logger.info(f"load spatial ({stage})")
            segments = try_load(segsfile)
        logger.info(f"exec clean ({stage})")
        n1 = np.count_nonzero(stats.kind == "cell")
        stats["segid"] = np.where(stats.kind == "cell", stats.spkid, stats.bgid + n1)
        footprints, stats = clean(
            stats,
            segments,
            cfg.radius.filter,
            **cfg.clean.args,
            **cfg.cmd.clean,
        )
        save((footprintsfile, statsfile), (footprints, stats))
        logger.info(f"saved clean ({stage})")
        if cfg.cmd.remove_segments:
            segsfile.unlink(missing_ok=True)
