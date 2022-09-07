import click
import matplotlib.pyplot as plt
import numpy as np

from ...evaluate.footprint import plot_maximum
from ..base import configure


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--stage", type=int)
@click.pass_obj
def figure(obj, tag, stage):
    """Output Figure"""

    prev_log = obj.log("1temporal", tag, stage)
    data_tag = prev_log["data_tag"]
    segment_tag = prev_log["segment_tag"]
    segment_stage = prev_log["segment_stage"]

    mask = obj.mask(data_tag)
    hz = obj.hz(data_tag)
    h, w = mask.shape
    tau = obj.used_tau(tag, stage)

    segment = obj.segment(segment_tag, segment_stage)
    nk = segment.shape[0]
    imgs = np.zeros((nk, h, w))
    imgs[:, mask] = segment

    spike = obj.spike(tag, stage)

    fig = plt.figure(figsize=(1, 1))

    ax = fig.add_axes([0, 0, 7, 7 * h / w])
    plot_maximum(ax, imgs, 0.0, 1.0)

    ax = fig.add_axes([0, -7 * h / w - 1, 7, 5])
    spike /= spike.max(axis=1, keepdims=True)
    ax.imshow(spike, aspect="auto", cmap="Reds", vmin=0.0, vmax=0.5)
    ax.set_xlabel("time (frame)")
    ax.set_ylabel("cells")
    ax.set_yticks([])

    path = obj.out_path("figure", tag, stage)
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.1)
