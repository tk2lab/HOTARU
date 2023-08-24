from pathlib import Path

import hydra

from hotaru.cui.common import all_stats, get_data

from hotaru.plot.seg import seg_max_image, bg_sum_image, segs_image, seg_max_fig
from hotaru.plot.spike import spike_image

from hotaru.plot.num import cell_num_fig
from hotaru.plot.spike import spike_stats_fig
from hotaru.plot.seg import footprint_stats_fig


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):

    name = cfg.data.label

    final = len(all_stats(cfg)) - 1
    h, w = get_data(cfg).shape

    fig_dir = Path("figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    """
    footprint_stats_fig(cfg, [0], usefind=True, width=200, height=200).write_image(
        fig_dir / f"{name}0peak.pdf",
    )
    seg_max_image(cfg, 0, base=0.5).save(
        fig_dir / f"{name}0segmax.png",
        dpi=(w / 3, w / 3),
    )
    bg_sum_image(cfg, 0).save(
        fig_dir / f"{name}0bgsum.png",
        dpi=(w / 3, w / 3),
    )

    segs_image(cfg, 0, range(0, 100), mx=10, hsize=25, pad=2).save(
        fig_dir / f"{name}0segtop.png",
        dpi=(255, 255),
    )
    segs_image(cfg, 0, range(-100, 0), mx=10, hsize=25, pad=2).save(
        fig_dir / f"{name}0segbottom.png",
        dpi=(255, 255),
    )
    img, nt, nk = spike_image(
        cfg, 0, tsel=slice(t0:=400, t1:=800),
        lines=((0, 99, (0, 0, 0), 3), (-100, -1, (0, 0, 0), 3)),
    )
    img.save(
        fig_dir / f"{name}0spike.png",
        dpi=(int(0.3 * nt), int(0.3 * nk)),
    )
    """

    cell_num_fig(cfg, width=400, height=200).write_image(
        fig_dir / f"{name}run_num.pdf",
    )
    spike_stats_fig(cfg, [0, 1, final], width=600, height=200).write_image(
        fig_dir / f"{name}run_spike.pdf",
    )
    footprint_stats_fig(cfg, [0, 1, final], width=600, height=200).write_image(
        fig_dir / f"{name}run_footprint.pdf",
    )

    segs_image(cfg, final, mx=10, hsize=25, pad=2).save(
        fig_dir / f"{name}finalseg.png",
        dpi=(255, 255),
    )
    seg_max_image(cfg, final, base=0.5).save(
        fig_dir / f"{name}finalsegmax.png",
        dpi=(w / 3, w / 3),
    )
    seg_max_fig(cfg, final, base=0.5).write_image(
        fig_dir / f"{name}finalsegmax.pdf",
    )
    bg_sum_image(cfg, final).save(
        fig_dir / f"{name}finalbgsum.png",
        dpi=(w / 3, w / 3),
    )
    img, nt, nk = spike_image(
        cfg, final, tsel=slice(t0:=400, t1:=800),
    )
    img.save(
        fig_dir / f"{name}finalspike.png",
        dpi=(int(0.3 * nt), int(0.3 * nk)),
    )


main()
