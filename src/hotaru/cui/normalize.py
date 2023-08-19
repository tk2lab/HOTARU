from logging import getLogger

from ..filter import movie_stats
from ..io import (
    apply_mask,
    load_imgs,
    save,
)
from ..utils import get_xla_stats
from .common import get_force, get_files

logger = getLogger(__name__)


def normalize(cfg):
    force = get_force(cfg, "normalize", 0)
    files = get_files(cfg, "normalize", 0)
    if force or not all(file.exists() for file in files):
        logger.info("exec normalize")
        imgs, hz = load_imgs(**cfg.data.imgs)
        imgs, mask = apply_mask(imgs, **cfg.data.mask)
        logger.info("%s", get_xla_stats())
        outputs = movie_stats(imgs, mask, **cfg.cmd.stats)
        logger.info("%s", get_xla_stats())
        save(files, outputs)
        logger.info("saved normalize")
