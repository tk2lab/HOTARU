from logging import getLogger
from pathlib import Path

import numpy as np

from ..io import (
    apply_mask,
    load_imgs,
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

logger = getLogger(__name__)


def plotter(cfg):
    def load(name):
        odir = Path(cfg.outputs.dir)
        path = cfg.outputs[name]
        fdir = odir / path.dir
        files = [fdir / file.format(stage=stage) for file in path.files]
        out = try_load(files)
        if out is None or np.any([o is None for o in out]):
            raise RuntimeError(f"load failed: {files}")
        else:
            logger.info("loaded:" + " %s" * len(files), *files)
        return out

    fig_dir = Path(cfg.outputs.dir) / cfg.outputs.figs.dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    time = cfg.plot.time

    imgs, hz = load_imgs(**cfg.data.imgs)
    imgs, mask = apply_mask(imgs, **cfg.data.mask)

    stage = cfg.plot.stage

    if stage == 0:
        stats, *simgs = load("stats")
        findval = load("find")
        footprints, peaks = load("make")
    else:
        footprints, peaks = load("spatial")

    # gen_normalize_movie("test.mp4", data)
    # plot_gl(data, radius, [100, 200, 300], scale=0.3).write_image(fig_dir / "gl.pdf")
    if stage == 0:
        plot_simgs(simgs).write_image(fig_dir / "stats.pdf")
        plot_peak_stats(peaks, findval).write_image(fig_dir / f"{stage:03d}peaks.pdf")
    else:
        plot_peak_stats(peaks).write_image(fig_dir / f"{stage:03d}peaks.pdf")
    plot_seg_max(footprints, peaks, **cfg.plot.seg_max).write_image(fig_dir / f"{stage:03d}max.pdf")
    plot_seg(footprints, peaks, 10).write_image(fig_dir / f"{stage:03d}seg.pdf")

    spikes, background = load("temporal")
    #stats = load("spike")
    diff = spikes.shape[1] - imgs.shape[0]

    plot_spike(spikes, hz, diff, time).write_image(fig_dir / f"{stage:03d}spike.pdf")
