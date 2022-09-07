import click
import numpy as np
import pandas as pd

from ...footprint.clean import check_accept
from ...footprint.clean import clean_footprint
from ...footprint.clean import modify_footprint
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--stage", type=int)
@click.option("--radius-type", type=click.Choice(["log", "linear"]))
@click.option("--radius-min", type=float)
@click.option("--radius-max", type=float)
@click.option("--radius-num", type=int)
@click.option("--distance", type=float)
@click.option("--thr-area-abs", type=click.FloatRange(0.0))
@click.option("--thr-area-rel", type=click.FloatRange(0.0))
@click.option("--thr-sim", type=click.FloatRange(0.0, 1.0))
@click.option("--gauss", type=float)
@click.option("--batch", type=click.IntRange(0))
@click.pass_obj
@command_wrap
def clean(obj, tag, stage, distance, gauss, batch, **args):
    """Clean Footprint and Make Segment."""

    footprint_tag = tag
    footprint_stage = stage

    prev_log = obj.log("2spatial", footprint_tag, footprint_stage)
    data_tag = prev_log["data_tag"]
    prev_tag = prev_log["segment_tag"]
    prev_stage = prev_log["segment_stage"]

    mask = obj.mask(data_tag)
    footprint = obj.footprint(footprint_tag, footprint_stage)
    index = obj.index(prev_tag, prev_stage)
    nk = footprint.shape[0]

    cond = modify_footprint(footprint)
    no_seg = pd.DataFrame(index=index[~cond])
    no_seg["accept"] = "no_seg"

    radius_opt = {k: v for k, v in args.items() if k[:6] == "radius"}
    radius = obj.get_radius(**radius_opt)
    thr_opt = {k: v for k, v in args.items() if k[:3] == "thr"}

    with Progress(length=nk, label="Clean", unit="cell") as prog:
        with obj.strategy.scope():
            segment, peaks = clean_footprint(
                footprint[cond],
                index[cond],
                mask,
                gauss,
                radius,
                batch,
                prog=prog,
            )

    idx = np.argsort(peaks["firmness"].values)[::-1]
    segment = segment[idx]
    peaks = peaks.iloc[idx].copy()

    check_accept(segment, peaks, radius, distance, **thr_opt)

    cond = peaks["accept"] == "yes"
    peaks = pd.concat([peaks, no_seg], axis=0)
    obj.save_numpy(segment[cond], "segment", tag, stage)
    obj.save_csv(peaks, "peak", tag, stage)

    peaks = peaks.loc[index]
    cond = peaks["accept"] == "yes"
    obj.save_numpy(footprint[~cond], "removed", tag, stage)

    old_nk = footprint.shape[0]
    nk = cond.sum()
    sim = peaks.loc[cond, "sim"].values
    peaks.sort_values("firmness", ascending=False, inplace=True)
    click.echo(peaks.loc[cond].head())
    click.echo(peaks.loc[cond].tail())
    click.echo(peaks.loc[~cond])
    peaks.sort_values("sim", ascending=False, inplace=True)
    click.echo(peaks.loc[cond].head())
    # click.echo(f"sim: {np.sort(sim[sim > 0])}")
    click.echo(f"ncell: {old_nk} -> {nk}")

    log = dict(
        footprint_tag=footprint_tag,
        footprint_stage=footprint_stage,
        data_tag=data_tag,
        num_cell=int(nk),
    )
    return log, "3segment", tag, stage
