from contextlib import nullcontext
from logging import getLogger
from pathlib import Path

import numpy as np
from jax.profiler import trace

from ..utils import Data
from ..io import (
    apply_mask,
    load_imgs,
    save,
    try_load,
)
from .command import (
    normalize,
    find,
    make,
    spatial,
    temporal,
    evaluate,
)

logger = getLogger(__name__)


def get_force(cfg, name, stage):
    force_dict = dict(
        stats=["stats", "find", "make", "temporal", "evaluate"],
        find=["find", "make", "temporal", "evaluate"],
        make=["make", "temporal", "evaluate"],
        spatial=["spatial", "temporal", "evaluate"],
        temporal=["temporal", "evaluate"],
        evaluate=["evaluate"],
    )
    force = (stage > cfg.force_from[0]) or (
        (stage == cfg.force_from[0]) and (name in force_dict[cfg.force_from[1]])
    )
    return force


def get_files(cfg, name, stage):
    odir = Path(cfg.outputs.dir)
    path = cfg.outputs[name]
    fdir = odir / path.dir
    files = [fdir / file.format(stage=stage) for file in path.files]
    return files


def get_trace_ctx(cfg):
    if cfg.trace:
        odir = Path(cfg.outputs.dir)
        trace_dir = odir / cfg.outputs.trace.dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_ctx = trace(str(trace_dir))
    else:
        trace_ctx = nullcontext()
    return trace_ctx


def radius_stats(peaks):
    #logger.info("%s", peaks.groupby("kind").radius.count())

    rcell = peaks.query("kind=='cell'").radius
    rbg = peaks.query("kind=='background'").radius
    for r in np.sort(np.unique(rcell)):
        ncell = (rcell == r).sum()
        nbg = (rbg == r).sum()
        logger.info("radius: %f cell: %d bg: %d", r, ncell, nbg)


def cui_main(cfg):
    def load_or_exec(name, command, *args, force=False, **kwargs):
        files = get_files(cfg, name, stage)
        if force or get_force(cfg, name, stage):
            out = [None]
        else:
            out = try_load(files)
        if out is None or np.any([o is None for o in out]):
            logger.info("exec: %s", name)
            with get_trace_ctx(cfg):
                out = command(*args, **kwargs)
                if len(files) == 1:
                    save(files, [out])
                else:
                    save(files, out)
                logger.info("saved:" + " %s" * len(files), *files)
        else:
            logger.info("loaded:" + " %s" * len(files), *files)
        return out

    def finish():
        cell = peaks[peaks.kind == "cell"]
        bg = peaks[peaks.kind == "background"]
        removed = peaks[peaks.kind == "remove"]
        logger.info("cell: %d\n%s", cell.shape[0], cell.head())
        if bg.shape[0] > 0:
            logger.info("background: %d\n%s", bg.shape[0], bg.head())
        if removed.shape[0] > 0:
            logger.info("removed: %d\n%s", removed.shape[0], removed.head())
        nk_old = peaks.shape[0]
        nk = footprints.shape[0]
        return stage > 0 and cfg.early_stop and nk == nk_old

    imgs, hz = load_imgs(**cfg.data.imgs)
    imgs, mask = apply_mask(imgs, **cfg.data.mask)

    for stage in range(cfg.max_train_step + 1):
        logger.info("*** stage %s ***", stage)

        if stage == 0:
            data_stats, *_ = load_or_exec(
                "normalize",
                normalize,
                imgs,
                mask,
                hz,
                cfg,
            )
            data = Data(imgs, mask, hz, *data_stats)
            findval = load_or_exec("find", find, data, cfg)
            footprints, peaks = load_or_exec("make", make, data, findval, cfg)

        else:
            footprints, peaks = load_or_exec(
                "spatial",
                spatial,
                data,
                footprints,
                peaks,
                spikes,  # noqa
                background,  # noqa
                cfg,
            )

        radius_stats(peaks)

        if cfg.mode == "test":
            break

        spikes, background = load_or_exec(
            "temporal",
            temporal,
            data,
            footprints,
            peaks,
            cfg,
        )
        peaks = load_or_exec(
            "eval",
            evaluate,
            spikes,
            background,
            peaks,
            force=True,
        )

        if finish():
            break
