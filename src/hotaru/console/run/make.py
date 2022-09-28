import click
import numpy as np
import pandas as pd

from ...footprint.make import make_segment
from ...footprint.reduce import reduce_peak_idx_data
from ...footprint.reduce import reduce_peak_idx_finish
from ...evaluate.utils import calc_area
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--find-tag", type=str)
@click.option("--distance", type=float)
@click.option("--window", type=int)
@click.option("--batch", type=int)
@click.option("--threshold-region", type=float)
@click.option("--only-reduce", is_flag=True)
@click.pass_obj
@command_wrap
def make(obj, tag, find_tag, distance, window, batch, threshold_region, only_reduce):
    """Make Initial Segment."""

    if find_tag is None:
        find_tag = tag

    info = obj.info(find_tag, 0, "2find")
    radius = obj.used_radius(find_tag)

    cond_remove = np.round(info.radius, 3) == np.round(radius[0], 3)
    cond_local = np.round(info.radius, 3) == np.round(radius[-1], 3)
    cell_info = info.loc[~(cond_remove | cond_local)]
    local_info = info.loc[cond_local]

    idx_data = reduce_peak_idx_data(cell_info, distance, window)
    with Progress(iterable=idx_data, label="Reduce Cell", unit="block") as prog:
        idx = reduce_peak_idx_finish(prog)
    cell_info = cell_info.loc[idx]

    idx_data = reduce_peak_idx_data(local_info, distance, window)
    with Progress(iterable=idx_data, label="Reduce Local", unit="block") as prog:
        idx = reduce_peak_idx_finish(prog)
    local_info = local_info.loc[idx]

    nk = cell_info.shape[0]
    nl = local_info.shape[0]
    cell_info.insert(0, "kind", "cell")
    cell_info.insert(1, "id", np.arange(nk))
    local_info.insert(0, "kind", "local")
    local_info.insert(1, "id", np.arange(nl))
    info = pd.concat([cell_info, local_info], axis=0)
    info.insert(2, "old_kind", "-")
    info.insert(3, "old_id", -1)
    info["firmness"] = np.nan
    info["area"] = np.nan
    info["overwrap"] = np.nan
    info["scale"] = np.nan
    info["denseness"] = np.nan
    click.echo(f"num: {nk}, {nl}")

    if only_reduce:
        obj.save_csv(info, tag, 0, "3tune", "info")
        return {}, tag, 0, "3tune"

    data_tag = obj.data_tag(find_tag, 0, "2find")
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)

    with Progress(length=nt, label="Make", unit="frame") as prog:
        with obj.strategy.scope():
            segment = make_segment(data, mask, info, batch, prog=prog)

    info["area"] = calc_area(segment, threshold_region)
    footprint = segment[info.kind == "cell"]
    localx = segment[info.kind == "local"]

    obj.save_csv(info, tag, 1, "1spatial", "info")
    obj.save_numpy(footprint, tag, 1, "1spatial", "footprint")
    obj.save_numpy(localx, tag, 1, "1spatial", "localx")

    log = dict(data_tag=data_tag, find_tag=find_tag)
    return log, tag, 1, "1spatial"
