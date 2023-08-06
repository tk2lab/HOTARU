import hydra

from hotaru.filter import calc_stats
from hotaru.footprint import (
    clean_segments,
    find_peaks,
    make_segments,
    reduce_peaks,
)
from hotaru.io import (
    apply_mask,
    load_imgs,
    save,
    try_load,
)

# from hotaru.io.movie import gen_normalize_movie
from hotaru.io.plot import (
    plot_calcium,
    plot_peak_stats,
    plot_seg,
    plot_seg_max,
    plot_simgs,
    plot_spike,
)
from hotaru.train import (
    optimize_spatial,
    optimize_temporal,
    prepare_spatial,
    prepare_temporal,
)
from hotaru.utils import Data


def load_or_exec(path, command, cfg, *args):
    filename = f"{path}/{cfg.outfile}"
    out = try_load(filename)
    if cfg.force or out is None:
        out = command(*args, **cfg.args)
        save(filename, out)
    return out


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):
    labels = hydra.core.hydra_config.HydraConfig.get().runtime.choices

    imgs, hz = load_imgs(**cfg.imgs)
    imgs, mask = apply_mask(imgs, **cfg.mask)
    nt, h, w = imgs.shape

    path = labels.data
    stats = load_or_exec(path, calc_stats, cfg.stats, imgs, mask)

    data = Data(imgs, mask, hz, *stats[:7])
    simgs = stats[7:]

    path += f"/{labels.radius}"
    peakval = load_or_exec(path, find_peaks, cfg.find, data)

    path += f"/{labels.density}"
    peaks = load_or_exec(path, reduce_peaks, cfg.reduce, peakval)
    footprint = load_or_exec(path, make_segments, cfg.make, data, peaks)

    path += f"/{labels.dynamics}/{labels.penalty}"
    cfgx = cfg.spatial
    cfgt = cfg.temporal
    prepare = load_or_exec(path, prepare_temporal, cfgt.prepare, footprint, data)
    spike = load_or_exec(path, optimize_temporal, cfgt.optimize, prepare)
    for i in range(cfg.max_train_step):
        pathi = f"{path}/{i}"
        prepare = load_or_exec(pathi, prepare_spatial, cfgx.prepare, spike, data)
        segment = load_or_exec(pathi, optimize_spatial, cfgx.optimize, prepare)
        footprint, *_ = load_or_exec(
            pathi, clean_segments, cfgx.clean, segment, (h, w), mask
        )
        prepare = load_or_exec(pathi, prepare_temporal, cfgt.prepare, footprint, data)
        spike = load_or_exec(pathi, optimize_temporal, cfgt.optimize, prepare)

    # gen_normalize_movie("test.mp4", data)
    plot_peak_stats(peakval, peaks).write_image("figs/peaks.pdf")
    plot_simgs(simgs).write_image("figs/stats.pdf")
    # plot_gl(data, [100, 200, 300], scale=0.3, **cfg.radius).write_image("figs/gl.pdf")
    plot_seg_max(footprint).write_image("figs/max.pdf")
    plot_seg(peaks, footprint).write_image("figs/seg.pdf")
    plot_calcium(prepare.a, footprint, hz).write_image("figs/calcium.pdf")
    plot_spike(spike, hz).write_image("figs/spike.pdf")


main()
