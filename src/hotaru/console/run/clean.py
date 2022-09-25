import click
import numpy as np
import pandas as pd

from ...footprint.clean import check_accept
from ...footprint.clean import clean_footprint
from ...footprint.clean import modify_footprint
from ..base import command_wrap
from ..base import configure
from ..base import radius_options
from ..base import threshold_options
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--footprint-tag", type=str)
@click.option("--footprint-stage", type=int)
@radius_options
@threshold_options
@click.option("--batch", type=click.IntRange(0))
@click.option("--storage-saving", is_flag=True)
@click.pass_obj
@command_wrap
def clean(
    obj,
    tag,
    footprint_tag,
    footprint_stage,
    radius,
    threshold,
    batch,
    storage_saving,
):
    """Clean Footprint and Make Segment."""

    if footprint_tag != tag:
        stage = 1
    else:
        stage = footprint_stage

    if storage_saving:
        stage = 999

    data_tag = obj.data_tag("1spatial", footprint_tag, footprint_stage)
    mask = obj.mask(data_tag)

    old_nk, old_nl, old_peaks = obj.index("1spatial", footprint_tag, footprint_stage)
    footprint = obj.footprint(footprint_tag, footprint_stage)

    cond = modify_footprint(footprint)
    num_seg = cond.sum()
    no_seg = pd.DataFrame(index=old_peaks.index[~cond])
    no_seg["segid"] = -1
    no_seg["kind"] = "remove"
    no_seg["id"] = -1
    no_seg["x"] = -1
    no_seg["y"] = -1
    no_seg["sim_with"] = -1
    no_seg["wrap_with"] = -1

    with Progress(length=num_seg, label="Clean", unit="cell") as prog:
        with obj.strategy.scope():
            segment, peaks_seg = clean_footprint(
                footprint[cond],
                old_peaks.index[cond],
                mask,
                radius,
                batch,
                prog=prog,
            )
    cell, local, peaks_seg = check_accept(segment, peaks_seg, radius, **threshold)
    peaks = pd.concat([peaks_seg, no_seg], axis=0)
    peaks.insert(3, "old_kind", old_peaks["kind"])
    peaks.insert(4, "old_id", old_peaks["id"])
    obj.save_csv(peaks, "peak", tag, stage)
    obj.save_numpy(cell, "segment", tag, stage)
    obj.save_numpy(local, "localx", tag, stage)

    click.echo(peaks.query("kind == 'cell'"))
    click.echo(peaks.query("kind == 'local'"))
    click.echo(peaks.query("kind == 'remove'"))
    click.echo(peaks.sort_values("overwrap", ascending=False).head())
    click.echo(peaks.sort_values("similarity", ascending=False).head())
    click.echo(f"ncell: {old_nk}, {old_nl} -> {cell.shape[0]}, {local.shape[0]}")

    log = dict(data_tag=data_tag, num_cell=cell.shape[0], num_local=local.shape[0])
    return log, "2segment", tag, stage


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
