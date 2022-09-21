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
@click.option("--footprint-tag", type=str)
@click.option("--footprint-stage", type=int)
@click.option("--storage-saving", is_flag=True)
@click.option("--radius-type", type=click.Choice(["log", "linear"]))
@click.option("--radius-min", type=float)
@click.option("--radius-max", type=float)
@click.option("--radius-num", type=int)
@click.option("--thr-area", type=click.FloatRange(0.0, 1.0))
@click.option("--thr-overwrap", type=click.FloatRange(0.0, 1.0))
@click.option("--batch", type=click.IntRange(0))
@click.pass_obj
@command_wrap
def clean(
    obj,
    tag,
    footprint_tag,
    footprint_stage,
    storage_saving,
    thr_area,
    thr_overwrap,
    batch,
    **args,
):
    """Clean Footprint and Make Segment."""

    if footprint_tag != tag:
        stage = 1
    else:
        stage = footprint_stage

    if storage_saving:
        stage = 999

    prev_log = obj.log("2spatial", footprint_tag, footprint_stage)
    data_tag = prev_log["data_tag"]
    prev_tag = prev_log["segment_tag"]
    prev_stage = prev_log["segment_stage"]

    mask = obj.mask(data_tag)
    footprint = obj.footprint(footprint_tag, footprint_stage)
    localx0 = obj.localx0(footprint_tag, footprint_stage)
    index = obj.index(prev_tag, prev_stage)
    old_nk = footprint.shape[0]

    cond = modify_footprint(footprint)
    num_seg = cond.sum()
    no_seg = pd.DataFrame(index=index[~cond])
    no_seg["segmentid"] = -1
    no_seg["x"] = -1
    no_seg["y"] = -1
    no_seg["next"] = -1
    no_seg["accept"] = "no"
    no_seg["reason"] = "no_seg"

    radius_opt = {k: v for k, v in args.items() if k[:6] == "radius"}
    radius = obj.get_radius(**radius_opt)
    radius_min = radius[0]
    radius_max = radius[-1]

    with Progress(length=num_seg, label="Clean", unit="cell") as prog:
        with obj.strategy.scope():
            segment, peaks_seg = clean_footprint(
                footprint[cond],
                index[cond],
                mask,
                radius,
                batch,
                prog=prog,
            )

    idx = np.argsort(peaks_seg.firmness.values)[::-1]
    segment = segment[idx]
    peaks_seg = peaks_seg.iloc[idx].copy()

    check_accept(
        segment, peaks_seg, radius_min, radius_max, thr_area, thr_overwrap,
    )
    peaks_seg["segmentid"] = np.arange(num_seg)
    peaks = pd.concat([peaks_seg, no_seg], axis=0)
    peaks.loc[index, "oldid"] = np.arange(old_nk)
    peaks_ordered = peaks.loc[index]

    cond_seg = peaks_seg["accept"] == "yes"
    obj.save_numpy(segment[cond_seg], "segment", tag, stage)

    cond_local = peaks_ordered["accept"] == "localx"
    localx = np.concatenate([localx0, footprint[cond_local]], axis=0)
    num_localx = localx.shape[0]
    obj.save_numpy(localx, "localx", tag, stage)

    cond_remove = peaks_ordered["accept"] == "no"
    obj.save_numpy(footprint[cond_remove], "removed", tag, stage)

    accept = peaks_seg["accept"] == "yes"
    nk = accept.sum()
    peaks = clean_peaks_df(peaks)
    obj.save_csv(peaks, "peak", tag, stage)

    click.echo(peaks.query("accept == 'yes'"))
    click.echo(peaks.query("reason == 'large_r'"))
    click.echo(peaks.query("accept == 'no'"))
    click.echo(f"ncell: {old_nk} -> {nk}, {num_localx}")

    log = dict(
        footprint_tag=footprint_tag,
        footprint_stage=footprint_stage,
        data_tag=data_tag,
        num_cell=int(nk),
    )
    return log, "3segment", tag, stage


def clean_peaks_df(peaks):
    peaks["oldid"] = peaks.oldid.astype(np.int32)
    peaks["x"] = peaks.x.astype(np.int32)
    peaks["y"] = peaks.y.astype(np.int32)
    peaks["next"] = peaks.next.astype(np.int32)
    return peaks[
        [
            "segmentid",
            "oldid",
            "x",
            "y",
            "radius",
            "firmness",
            "area",
            "next",
            "overwrap",
            "accept",
            "reason",
        ]
    ]
