from pathlib import Path

import hydra

from hotaru.cui.common import (
    all_stats,
    get_data,
)
from hotaru.plot.densesig import dense_sig_fig
from hotaru.plot.seg import (
    bg_sum_image,
    seg_max_fig,
    seg_max_image,
    segs_image,
)
from hotaru.plot.spike import spike_image
from hotaru.plot.num import cell_num_fig


@hydra.main(version_base=None, config_path="pkg://hotaru.conf", config_name="config")
def main(cfg):
    name = f"{cfg.data.label}-{cfg.penalty.ulabel}-{cfg.clean.label}"

    final = len(all_stats(cfg)) - 1
    # final = 15
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

    thr_udense = 0.18

    """
    cell_num_fig(cfg, width=400, height=200).write_image(
        fig_dir / f"{name}run_num.pdf",
    )
    dense_sig_fig(
        cfg, [0, 1, final], thr_sig=cfg.clean.args.thr_bg_signal, label="epoch="
    ).write_image(
        fig_dir / f"{name}densesig.pdf",
    )
    """

    seg_max_fig(cfg, final, base=0.2, width=300, thr_udense=thr_udense).write_image(
        fig_dir / f"{name}finalsegmax.pdf",
    )
    seg_max_image(cfg, final, base=0.2).save(
        fig_dir / f"{name}finalsegmax.png",
        dpi=(w / 3, w / 3),
    )
    bg_sum_image(cfg, final).save(
        fig_dir / f"{name}finalbgsum.png",
        dpi=(w / 3, w / 3),
    )
    segs_image(cfg, final, mx=10, hsize=25, pad=2, thr_udense=thr_udense).save(
        fig_dir / f"{name}finalseg.png",
        dpi=(255, 255),
    )

    img, nt, nk = spike_image(
        cfg,
        final,
        #tsel=slice(t0 := 400, t1 := 800),
        thr_udense=thr_udense,
    )
    img.save(
        fig_dir / f"{name}finalspike.png",
        dpi=(int(0.3 * nt), int(0.3 * nk)),
    )

    """
    run_fig(
        cfg,
        [1, final],
        thr_f=cfg.clean.args.thr_bg_firmness,
        thr_d=cfg.clean.args.thr_bg_udense,
        min_val=(0.2, 0),
        max_val=(0.58, 0.2),
        min_r=1,
        max_r=1,
    ).write_image(
        fig_dir / f"{name}run.pdf",
    )
    """
    """
    spike_stats_fig(cfg, [0, 1, final], width=600, height=200).write_image(
        fig_dir / f"{name}run_spike.pdf",
    )
    footprint_stats_fig(cfg, [0, 1, final], width=600, height=200).write_image(
        fig_dir / f"{name}run_footprint.pdf",
    )
    """


main()
