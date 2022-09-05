import click
import numpy as np
import pandas as pd
import tensorflow as tf

from ...train.dynamics import SpikeToCalciumDoubleExp
from ..base import configure


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--stage", type=int)
@click.pass_obj
def output(obj, tag, stage):
    """Output"""

    prev_log = obj.log("1temporal", tag, stage)
    data_tag = prev_log["data_tag"]
    segment_tag = prev_log["segment_tag"]
    segment_stage = prev_log["segment_stage"]

    mask = obj.mask(data_tag)
    hz = obj.hz(data_tag)
    h, w = mask.shape
    tau = obj.used_tau(tag, stage)

    segment = obj.segment(segment_tag, segment_stage)
    u = obj.spike(tag, stage)
    nk = u.shape[0]

    imgs = np.zeros((nk, h, w))
    imgs[:, mask] = segment
    obj.save_tiff(imgs, "output", f"footprint_{tag}", stage)

    with tf.device("CPU"):
        g = SpikeToCalciumDoubleExp(hz=hz, **tau)
        v = g(u).numpy()

    gap = u.shape[1] - v.shape[1]
    time = np.arange(-gap, v.shape[1]) / hz
    cols = [f"cell{i:03}" for i in range(u.shape[0])]
    u = pd.DataFrame(u.T, columns=cols)
    u.index = time
    u.index.name = "time"
    v = pd.DataFrame(v.T, columns=cols)
    v.index = time[gap:]
    v.index.name = "time"
    obj.save_csv(u, "output", f"spike_{tag}", stage)
    obj.save_csv(v, "output", f"calcium_{tag}", stage)
    if stage < 0:
        stage = "_curr"
    else:
        stage = f"_{stage:03}"
    click.echo(f"create: {obj.workdir}/output/TYPE_{tag}{stage}.EXT")
