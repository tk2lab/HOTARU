from logging import getLogger

import numpy as np

from ..footprint import (
    find_peaks,
    make_footprints,
    reduce_peaks,
)
from ..io import (
    save,
    try_load,
)
from ..utils import get_xla_stats
from .common import (
    get_force,
    get_data,
    get_files,
)

logger = getLogger(__name__)


def init(cfg):
    force = get_force(cfg, "init", 0)
    makefile, statsfile = get_files(cfg, "make", 0)
    if force or not statsfile.exists():
        if force or not makefile.exists():
            logger.info("%s", get_xla_stats())
            data = get_data(cfg)
            (reducefile,) = get_files(cfg, "reduce", 0)
            if force or not reducefile.exists():
                (findfile,) = get_files(cfg, "find", 0)
                if force or not findfile.exists():
                    logger.info("exec find")
                    findval = find_peaks(data, **cfg.init.find, **cfg.cmd.find)
                    logger.info("%s", get_xla_stats())
                    save(findfile, findval)
                    logger.info("saved find")
                else:
                    logger.info("load find")
                    findval = try_load(findfile)
                logger.info("exec reduce")
                peaks = reduce_peaks(findval, **cfg.init.reduce, **cfg.cmd.reduce)
                logger.info("%s", get_xla_stats())
                save(reducefile, peaks)
                findfile.unlink(missing_ok=True)
                logger.info("saved reduce")
            else:
                logger.info("load reduce")
                peaks = try_load(reducefile)
            logger.info("exec make")
            footprints = make_footprints(data, peaks, **cfg.cmd.make)
            logger.info("%s", get_xla_stats())
            save(makefile, footprints)
            logger.info("saved make")
        else:
            logger.info("load make")
            footprints = try_load(makefile)
        logger.info("exec init")
        peaks["sum"] = footprints.sum(axis=(1, 2))
        peaks["area"] = np.count_nonzero(footprints > 0, axis=(1, 2))
        save(statsfile, peaks)
        reducefile.unlink(missing_ok=True)
        logger.info("saved init")
