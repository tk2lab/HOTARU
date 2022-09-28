import click
import tensorflow as tf
import numpy as np
import pandas as pd

from ...evaluate.summary import write_spike_summary
from ...footprint.evaluate import normalize_footprint
from ..base import command_wrap
from ..base import configure
from ..base import dynamics_options
from ..base import penalty_options
from ..base import optimizer_options
from ..base import early_stop_options
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--spatial-tag", type=str)
@click.option("--spatial-stage", type=int)
@click.option("--threshold-region", type=float)
@click.option("--threshold-area", type=float)
@dynamics_options
@penalty_options
@optimizer_options
@early_stop_options
@click.option("--batch", type=int)
@click.option("--epochs", type=int)
@click.option("--storage-saving", is_flag=True)
@click.pass_obj
@command_wrap
def temporal(
    obj,
    tag,
    spatial_tag,
    spatial_stage,
    threshold_region,
    threshold_area,
    dynamics,
    penalty,
    optimizer,
    early_stop,
    epochs,
    batch,
    storage_saving,
):
    """Update Spike from Segment."""

    if spatial_tag is None:
        spatial_tag = tag

    if spatial_stage is None:
        spatial_stage = 1

    if spatial_tag != tag:
        stage = 1
    else:
        stage = spatial_stage

    if storage_saving:
        stage = 999

    prev = dict(tag=spatial_tag, stage=spatial_stage, kind="1spatial")
    curr = dict(tag=tag, stage=stage, kind="2temporal")

    data_tag = obj.data_tag(**prev)
    nt = obj.nt(data_tag)

    info = obj.info(**prev)
    footprint = obj.footprint(**prev)
    localx = obj.localx(**prev)
    old_nk = footprint.shape[0]
    old_nl = localx.shape[0]

    model = obj.model(data_tag, info.shape[0])
    model.set_double_exp(**dynamics)
    model.set_penalty(**penalty)
    model.temporal.optimizer.set(**optimizer)
    model.set_early_stop(**early_stop)

    no_seg = normalize_footprint(footprint)
    area = np.count_nonzero(footprint > threshold_region, axis=1)
    cell_to_local = area > threshold_area
    localx_new = footprint[cell_to_local]
    localx = np.concatenate([localx_new, localx], axis=0)
    footprint = footprint[~(no_seg | cell_to_local)]
    nk = footprint.shape[0]
    nl = localx.shape[0]

    info["old_kind"] = info.kind
    info["old_id"] = info.id
    cell_info = info.query("kind == 'cell'").copy()
    cell_info["area"] = area
    cell_info.loc[cell_to_local, "kind"] = "local"
    cell_info.loc[no_seg, "kind"] = "remove"
    local_info = info.query("kind == 'local'").copy()
    info = pd.concat([cell_info, local_info], axis=0)
    info.loc[info.kind == "cell", "id"] = np.arange(nk)
    info.loc[info.kind == "local", "id"] = np.arange(nl)

    obj.save_csv(info, **curr, name="info")
    click.echo(f"num: {old_nk}, {old_nl} -> {nk}, {nl}")

    with Progress(length=nt, label="InitT", unit="frame") as prog:
        model.footprint.set_val(footprint)
        model.localx.set_val(localx)
        model.prepare_temporal(batch, prog=prog)

    summary_dir = obj.summary_path("temporal", tag, stage)
    cb = obj.callbacks("TrainT", summary_dir)
    model_log = model.fit_temporal(callbacks=cb, epochs=epochs, verbose=0)

    obj.save_numpy(footprint, **curr, name="footprint")
    obj.save_numpy(localx, **curr, name="localx")
    obj.save_numpy(model.spike.get_val(), **curr, name="spike") 
    obj.save_numpy(model.localt.get_val(), **curr, name="localt")

    log = dict(
        data_tag=data_tag,
        spatial_tag=spatial_tag,
        spatial_stage=spatial_stage,
        history=model_log.history,
    )
    return log, tag, stage, "2temporal"
