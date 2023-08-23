from pathlib import Path

import hydra

from hotaru.plot.seg import seg_max_image, segs_image
from hotaru.plot.spike import spike_image

from hotaru.plot.num import cell_num_fig
from hotaru.plot.spike import spike_stats_fig
from hotaru.plot.seg import footprint_stats_fig


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):

    fig_dir = Path("../figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    w = cfg.data.imgs.width

    footprint_stats_fig(cfg, [0], usefind=True, width=200, height=200).write_image(
        fig_dir / f"{cfg.plot.name}peak0.pdf",
    )
    seg_max_image(cfg, 0).save(
        fig_dir / f"{cfg.plot.name}segmax0.png",
        dpi=(w / 3, w / 3),
    )

    segs_image(cfg, 0, range(100), mx=10, hsize=25, pad=2).save(
        fig_dir / f"{cfg.plot.name}seg0top.png",
        dpi=(255, 255),
    )
    segs_image(cfg, 0, range(-100, 0), mx=10, hsize=25, pad=2).save(
        fig_dir / f"{cfg.plot.name}seg0bottom.png",
        dpi=(255, 255),
    )
    img, nt, nk = spike_image(
        cfg, 0, tsel=range(t0:=400, t1:=800),
        lines=((0, 99, (0, 0, 0), 3), (-100, -1, (0, 0, 0), 3)),
    )
    img.save(
        fig_dir / f"{cfg.plot.name}spike0.png",
        dpi=(int(0.3 * nt), int(0.3 * nk)),
    )

    cell_num_fig(cfg, width=400, height=200).write_image(
        fig_dir / f"{cfg.plot.name}num.pdf",
    )
    spike_stats_fig(cfg, [0, 1, cfg.plot.final], width=600, height=200).write_image(
        fig_dir / f"{cfg.plot.name}spike_stats.pdf",
    )
    footprint_stats_fig(cfg, [0, 1, cfg.plot.final], width=600, height=200).write_image(
        fig_dir / f"{cfg.plot.name}footprint_stats.pdf",
    )

    segs_image(cfg, cfg.plot.final, mx=10, hsize=25, pad=2).save(
        fig_dir / f"{cfg.plot.name}segfinal.png",
        dpi=(255, 255),
    )
    seg_max_image(cfg, cfg.plot.final).save(
        fig_dir / f"{cfg.plot.name}segmaxfinal.png",
        dpi=(w / 3, w / 3),
    )
    img, nt, nk = spike_image(
        cfg, cfg.plot.final, tsel=range(t0:=400, t1:=800),
    )
    img.save(
        fig_dir / f"{cfg.plot.name}segfinal.png",
        dpi=(int(0.3 * nt), int(0.3 * nk)),
    )


main()
