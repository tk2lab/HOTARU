from logging import getLogger

import numpy as np
import pandas as pd

from ..footprint import clean
from ..spike import fix_kind
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
    cleanstatsfile, footprintsfile = get_files(cfg, "clean", stage)
    if (
        get_force(cfg, "clean", stage)
        or not cleanstatsfile.exists()
        or not footprintsfile.exists
    ):
        stats = try_load(get_files(cfg, "evaluate", stage - 1))
        stats = stats.query("kind != 'remove'").copy()
        segstatsfile, segsfile, lossfile = get_files(cfg, "spatial", stage)
        if (
                get_force(cfg, "spatial", stage)
                or not segstatsfile.exists()
                or not segsfile.exists()
        ):
            logger.info(f"exec spatial ({stage})")
            data = get_data(cfg)
            if stage == 1:
                footprints = try_load(get_files(cfg, "make", stage - 1))
            else:
                _, footprints = try_load(get_files(cfg, "clean", stage - 1))
            spikes, bg, _ = try_load(get_files(cfg, "temporal", stage - 1))
            logger.debug("%s", get_xla_stats())
            stats, spikes, bg = fix_kind(
                stats,
                spikes,
                bg,
                cfg.dynamics,
                bg_type=cfg.bg_type,
                **(cfg.clean.temporal if stage > 1 else {}),
            )
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
            if cfg.fix_top:
                nk, h, w = segments.shape
                rseg = segments.reshape(nk, h * w)
                idx = np.argpartition(rseg, -2, axis=1)
                k = np.arange(nk)
                rseg[k, idx[:, -1]] = rseg[k, idx[:, -2]]
                segments = rseg.reshape(nk, h, w)
            stats["segid"] = np.arange(stats.shape[0])
            save((segstatsfile, segsfile, lossfile), (stats, segments, logdf))
            logger.info(f"saved spatial ({stage})")
        else:
            logger.info(f"load spatial ({stage})")
            segments = try_load(segsfile)
        logger.info(f"exec clean ({stage})")
        stats, footprints = clean(
            stats,
            segments,
            cfg.radius.filter,
            bg_type=cfg.bg_type,
            **cfg.clean.spatial,
            **cfg.cmd.clean,
        )
        save((cleanstatsfile, footprintsfile), (stats, footprints))
        logger.info(f"saved clean ({stage})")
        if cfg.cmd.remove_segments:
            segsfile.unlink(missing_ok=True)
