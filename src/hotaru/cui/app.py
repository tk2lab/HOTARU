from contextlib import nullcontext
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from jax.profiler import trace

from ..filter import calc_stats
from ..footprint import (
    clean,
    find_peaks,
    get_radius,
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
    plot_simgs,
    plot_gl,
    #plot_calcium,
    plot_seg,
    plot_seg_max,
    plot_spike,
    plot_peak_stats,
)
from ..train import (
    get_dynamics,
    get_penalty,
    spatial,
    temporal,
)
from ..utils import (
    Data,
    get_gpu_env,
)

logger = getLogger(__name__)


def make(data, findval, env, cfg):
    peaks = reduce_peaks(findval, cfg.density, **cfg.cmd.reduce)
    segments = make_footprints(data, peaks, env, **cfg.cmd.make)
    peaks["sum"] = segments.sum(axis=(1, 2))
    peaks["area"] = np.count_nonzero(segments > 0, axis=(1, 2))
    nk = np.count_nonzero(peaks.kind == "cell")
    nb = np.count_nonzero(peaks.kind == "background")
    logger.info("num cell/bg: %d/%d", nk, nb)
    return segments[:nk], segments[nk:], peaks


def cui_main(cfg):
    def load_or_exec(name, command, *args, **kwargs):
        force_dict = dict(
            stats=["stats"],
            find=["find", "make", "temporal"],
            make=["make", "temporal"],
            spatial=["spatial", "clean", "temporal"],
            clean=["clean", "temporal"],
            temporal=["temporal"],
        )
        force = (stage > cfg.force_from[0]) or (
            (stage == cfg.force_from[0]) and (name in force_dict[cfg.force_from[1]])
        )
        odir = Path(cfg.outputs.dir)
        path = cfg.outputs[name]
        fdir = odir / path.dir
        files = [fdir / file.format(stage=stage) for file in path.files]
        out = try_load(files)
        logger.debug("try_load: %s", out)
        if force or np.any([o is None for o in out]):
            if cfg.trace:
                trace_dir = odir / cfg.outputs.trace
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_ctx = trace(str(trace_dir))
            else:
                trace_ctx = nullcontext()
            with trace_ctx:
                out = command(*args, **kwargs)
            logger.info("files:" + " %s" * len(files), *files)
            if len(files) == 1:
                files = files[0]
            save(files, out)
        else:
            if len(files) == 1:
                out = out[0]
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

    radius = get_radius(cfg.radius)
    dynamics = get_dynamics(cfg.model.dynamics)
    penalty = get_penalty(cfg.model.penalty)
    env = get_gpu_env(cfg.gpu)

    stage = -1
    logger.info("*** prepare ***")
    imgs, hz = load_imgs(**cfg.data.imgs)
    imgs, mask = apply_mask(imgs, **cfg.data.mask)
    stats, simgs = load_or_exec(
        "stats",
        calc_stats,
        imgs,
        mask,
        env,
        **cfg.cmd.stats,
    )
    plot_simgs(simgs).write_image(fig_dir / "stats.pdf")

    data = Data(imgs, mask, hz, *stats)
    #gen_normalize_movie("test.mp4", data)

    stage = 0
    logger.info("*** stage %s ***", stage)
    #plot_gl(data, radius, [100, 200, 300], scale=0.3).write_image(fig_dir / "gl.pdf")
    findval = load_or_exec(
        "find",
        find_peaks,
        data,
        radius,
        env,
        **cfg.cmd.find,
    )
    footprints, background, peaks = load_or_exec(
        "make",
        make,
        data,
        findval,
        env,
        cfg,
    )
    plot_peak_stats(peaks, findval).write_image(fig_dir / f"{stage:03d}peaks.pdf")
    plot_seg_max(footprints, background).write_image(fig_dir / f"{stage:03d}max.pdf")
    plot_seg(peaks, footprints, 10).write_image(fig_dir / f"{stage:03d}seg.pdf")
    spikes, background = load_or_exec(
        "temporal",
        temporal,
        data,
        (footprints, background),
        dynamics,
        penalty,
        env,
        **cfg.cmd.temporal,
    )
    plot_spike(spikes[:, :1000], hz).write_image(fig_dir / f"{stage:03d}spike.pdf")
    peaks["usum"] = pd.Series(spikes.sum(axis=1), index=peaks.query("kind=='cell'").index)
    peaks["unum"] = pd.Series(np.count_nonzero(spikes, axis=1), index=peaks.query("kind=='cell'").index)
    finish()

    for stage in range(1, cfg.max_train_step + 1):
        logger.info("*** stage %s ***", stage)
        segments = load_or_exec(
            "spatial",
            spatial,
            data,
            (spikes, background),
            dynamics,
            penalty,
            env,
            **cfg.cmd.spatial,
        )
        footprints, background, peaks = load_or_exec(
            "clean",
            clean,
            peaks,
            segments,
            data.shape,
            mask,
            radius,
            cfg.density,
            env,
            **cfg.cmd.clean,
        )
        plot_peak_stats(peaks).write_image(fig_dir / f"{stage:03d}peaks.pdf")
        plot_seg_max(footprints, background).write_image(fig_dir / f"{stage:03d}max.pdf")
        plot_seg(peaks, footprints, 10).write_image(fig_dir / f"{stage:03d}seg.pdf")
        spikes, background = load_or_exec(
            "temporal",
            temporal,
            data,
            (footprints, background),
            dynamics,
            penalty,
            env,
            **cfg.cmd.temporal,
        )
        plot_spike(spikes[:, :1000], hz).write_image(fig_dir / f"{stage:03d}spike.pdf")
        peaks["usum"] = pd.Series(spikes.sum(axis=1), index=peaks.query("kind=='cell'").index)
        peaks["unum"] = pd.Series(np.count_nonzero(spikes, axis=1), index=peaks.query("kind=='cell'").index)
        if finish():
            break
