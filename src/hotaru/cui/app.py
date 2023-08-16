from contextlib import nullcontext
from logging import getLogger
from pathlib import Path
from psutil import Process

import jax
import numpy as np
import pandas as pd
from jax.profiler import trace

from ..filter import calc_stats
from ..footprint import (
    clean,
    find_peaks,
    make_footprints,
    reduce_peaks,
)
from ..io import (
    apply_mask,
    load_imgs,
    save,
    try_load,
)

# from hotaru.io.movie import gen_normalize_movie
from ..io.plot import (
    plot_peak_stats,
    plot_seg,
    plot_seg_max,
    plot_simgs,
    plot_spike,
)
from ..train import (
    spatial,
    temporal,
)
from ..utils import Data

logger = getLogger(__name__)


def make(data, findval, env, cfg):
    peaks = reduce_peaks(findval, cfg.density, **cfg.cmd.reduce)
    footprints = make_footprints(data, peaks, env, **cfg.cmd.make)
    peaks["sum"] = footprints.sum(axis=(1, 2))
    peaks["area"] = np.count_nonzero(footprints > 0, axis=(1, 2))
    nk = np.count_nonzero(peaks.kind == "cell")
    nb = np.count_nonzero(peaks.kind == "background")
    logger.info("num cell/bg: %d/%d", nk, nb)
    return footprints, peaks


def spatial_and_clean(data, old_footprints, old_peaks, spikes, background, cfg):
    old_peaks = old_peaks[old_peaks.kind!="remove"]
    segments = spatial(
        data,
        old_footprints,
        old_peaks,
        spikes,
        background,
        cfg.model,
        cfg.env,
        **cfg.cmd.spatial,
    )
    uid = old_peaks.uid.to_numpy()
    footprints, peaks = clean(
        uid,
        segments,
        cfg.radius,
        cfg.density,
        cfg.env,
        **cfg.cmd.clean,
    )
    return footprints, peaks

def temporal_and_eval(data, footprints, peaks, cfg):
    spikes, background = temporal(
        data,
        footprints,
        peaks,
        cfg.model,
        cfg.env,
        **cfg.cmd.temporal,
    )
    peaks["umax"] = pd.Series(
        spikes.max(axis=1), index=peaks.query("kind=='cell'").index
    )
    peaks["unum"] = pd.Series(
        np.count_nonzero(spikes, axis=1), index=peaks.query("kind=='cell'").index
    )
    peaks["bmax"] = pd.Series(
        np.abs(background).max(axis=1), index=peaks.query("kind=='background'").index
    )
    peaks["bmean"] = pd.Series(
        background.mean(axis=1), index=peaks.query("kind=='background'").index
    )
    return spikes, background, peaks


def cui_main(cfg):
    def load_or_exec(name, command, *args, **kwargs):
        force_dict = dict(
            stats=["stats", "find", "make", "temporal"],
            find=["find", "make", "temporal"],
            make=["make", "temporal"],
            spatial=["spatial", "temporal"],
            temporal=["temporal"],
        )
        force = (stage > cfg.force_from[0]) or (
            (stage == cfg.force_from[0]) and (name in force_dict[cfg.force_from[1]])
        )
        odir = Path(cfg.outputs.dir)
        path = cfg.outputs[name]
        fdir = odir / path.dir
        files = [fdir / file.format(stage=stage) for file in path.files]
        if force:
            out = [None]
        else:
            out = try_load(files)
        if out is None or np.any([o is None for o in out]):
            logger.info("exec: %s", name)
            logger.info(
                "memory: %f MB", Process().memory_info().rss / 1014 / 1014
            )

            les = jax.lib.xla_bridge.get_backend().live_executables()
            lbs = jax.lib.xla_bridge.get_backend().live_buffers()
            logger.info(
                "live_buffer: %d" + " %s" * len(lbs),
                len(les),
                *(lb.shape for lb in lbs),
            )

            if cfg.trace:
                trace_dir = odir / cfg.outputs.trace.dir
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_ctx = trace(str(trace_dir))
            else:
                trace_ctx = nullcontext()
            with trace_ctx:
                out = command(*args, **kwargs)
                if len(files) == 1:
                    save(files, [out])
                else:
                    save(files, out)
                logger.info("saved:" + " %s" * len(files), *files)

            les = jax.lib.xla_bridge.get_backend().live_executables()
            lbs = jax.lib.xla_bridge.get_backend().live_buffers()
            logger.info(
                "live_buffer: %d" + " %s" * len(lbs),
                len(les),
                *(lb.shape for lb in lbs),
            )
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
        return cfg.early_stop and nk == nk_old

    fig_dir = Path(cfg.outputs.dir) / cfg.outputs.figs.dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    stage = 0
    logger.info("*** stage %s ***", stage)
    imgs, hz = load_imgs(**cfg.data.imgs)
    imgs, mask = apply_mask(imgs, **cfg.data.mask)
    stats, *simgs = load_or_exec(
        "stats",
        calc_stats,
        imgs,
        mask,
        cfg.env,
        **cfg.cmd.stats,
    )
    plot_simgs(simgs).write_image(fig_dir / "stats.pdf")

    data = Data(imgs, mask, hz, *stats)
    # gen_normalize_movie("test.mp4", data)
    # plot_gl(data, radius, [100, 200, 300], scale=0.3).write_image(fig_dir / "gl.pdf")

    findval = load_or_exec(
        "find",
        find_peaks,
        data,
        cfg.radius,
        cfg.env,
        **cfg.cmd.find,
    )
    footprints, peaks = load_or_exec("make", make, data, findval, cfg.env, cfg)
    plot_peak_stats(peaks, findval).write_image(fig_dir / f"{stage:03d}peaks.pdf")
    plot_seg_max(footprints, peaks).write_image(fig_dir / f"{stage:03d}max.pdf")
    plot_seg(footprints, peaks, 10).write_image(fig_dir / f"{stage:03d}seg.pdf")

    spikes, background, peaks = load_or_exec(
        "temporal",
        temporal_and_eval,
        data,
        footprints,
        peaks,
        cfg,
    )
    plot_spike(spikes[:, :1000], hz).write_image(fig_dir / f"{stage:03d}spike.pdf")

    finish()

    for stage in range(1, cfg.max_train_step + 1):
        logger.info("*** stage %s ***", stage)

        footprints, peaks = load_or_exec(
            "spatial",
            spatial_and_clean,
            data,
            footprints,
            peaks,
            spikes,
            background,
            cfg,
        )
        plot_peak_stats(peaks).write_image(fig_dir / f"{stage:03d}peaks.pdf")
        plot_seg_max(footprints, peaks).write_image(fig_dir / f"{stage:03d}max.pdf")
        plot_seg(footprints, peaks, 10).write_image(fig_dir / f"{stage:03d}seg.pdf")

        spikes, background, peaks = load_or_exec(
            "temporal",
            temporal_and_eval,
            data,
            footprints,
            peaks,
            cfg,
        )
        plot_spike(spikes[:, :1000], hz).write_image(fig_dir / f"{stage:03d}spike.pdf")

        if finish():
            break
