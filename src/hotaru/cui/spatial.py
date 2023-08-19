from logging import getLogger

import numpy as np

from ..footprint import clean
from ..io import (
    save,
    try_load,
)
from ..train import SpatialModel
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


def spatial(cfg, stage, force=False):
    force = get_force(cfg, "spatial", stage)
    footprintsfile, statsfile = get_files(cfg, "clean", stage)
    if force or not footprintsfile.exists or not statsfile.exists():
        stats, flags = try_load(get_files(cfg, "evaluate", stage - 1))
        segsfile, = get_files(cfg, "spatial", stage)
        if force or not segsfile.exists():
            logger.info(f"exec spatial ({stage})")
            data = get_data(cfg)
            if stage == 1:
                footprints = try_load(get_files(cfg, "make", stage - 1)[0])
            else:
                footprints = try_load(get_files(cfg, "clean", stage - 1)[0])
            spikes, bg = try_load(get_files(cfg, "temporal", stage - 1))
            stats = stats.query("kind != 'remove'")
            logger.info("%s", get_xla_stats())
            model = SpatialModel(
                data,
                footprints,
                stats,
                spikes,
                bg,
                cfg.model.dynamics,
                cfg.model.penalty,
            )
            clips = get_clip(data.shape, cfg.cmd.spatial.clip)
            out = []
            for clip in clips:
                model.prepare(clip, **cfg.cmd.spatial.prepare)
                model.fit(**cfg.cmd.spatial.step)
                logger.info("%s", get_xla_stats())
                out.append(model.get_x())
            index, x = (np.concatenate(v, axis=0) for v in zip(*out))
            logger.debug(
                "%d %d\n%s",
                index.size,
                np.count_nonzero(np.sort(index) != np.arange(index.size)),
                index,
            )
            segments = x[rev_index(index)]
            save(segsfile, segments)
            logger.info(f"saved spatial ({stage})")
        else:
            logger.info(f"load spatial ({stage})")
            segments = try_load(segsfile)
        logger.info(f"exec clean ({stage})")
        print(segments.shape)
        footprints, stats = clean(
            stats,
            segments,
            cfg.model.clean.radius,
            **cfg.model.clean.reduce,
            **cfg.cmd.clean,
        )
        save((footprintsfile, statsfile), (footprints, stats))
        logger.info(f"saved clean ({stage})")
