import click
import numpy as np

from ...evaluate.utils import calc_denseness
from ..base import (
    command_wrap,
    configure,
    dynamics_options,
    early_stop_options,
    optimizer_options,
    penalty_options,
)
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--spatial-tag", type=str)
@click.option("--spatial-stage", type=int)
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
    threshold_area,
    dynamics,
    penalty,
    optimizer,
    early_stop,
    batch,
    epochs,
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
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)

    footprint = obj.footprint(**prev)
    localx = obj.localx(**prev)

    info = obj.info(**prev)
    info.query("(kind == 'cell') or (kind == 'local')", inplace=True)
    info["old_kind"] = info.kind
    info["old_id"] = info.id

    model = obj.model(data_tag, info.shape[0])
    model.set_double_exp(**dynamics)
    model.set_penalty(**penalty)
    model.temporal.optimizer.set(**optimizer)
    model.set_early_stop(**early_stop)

    cell = info.query("kind == 'cell'")
    local = info.query("kind == 'local'")
    nk, nl = cell.shape[0], local.shape[0]
    click.echo(f"num: {nk}, {nl}")

    cell_to_local = cell.area > threshold_area
    localx_new = footprint[cell_to_local]
    localx = np.concatenate([localx, localx_new], axis=0)
    footprint = footprint[~cell_to_local]
    info.loc[cell.index[cell_to_local], "kind"] = "local"
    click.echo(f"large area: {np.count_nonzero(cell_to_local)}")

    cell = info.query("kind == 'cell'")
    local = info.query("kind == 'local'")
    nk, nl = cell.shape[0], local.shape[0]
    click.echo(f"num: {nk}, {nl}")

    remove = cell.overwrap == 1.0
    footprint = footprint[~remove]
    info.loc[cell.index[remove], "kind"] = "overwrap"
    click.echo(f"overwrap: {np.count_nonzero(remove)}")

    cell = info.query("kind == 'cell'")
    local = info.query("kind == 'local'")
    nk, nl = cell.shape[0], local.shape[0]
    click.echo(f"num: {nk}, {nl}")

    footprint /= footprint.max(axis=1, keepdims=True)
    localx /= localx.max(axis=1, keepdims=True)

    with Progress(length=nt, label="Temporal Init", unit="frame") as prog:
        model.footprint.set_val(footprint)
        model.localx.set_val(localx)
        model.prepare_temporal(batch, prog=prog)

    summary_dir = obj.summary_path(**curr)
    cb = obj.callbacks("Temporal Train", summary_dir)
    model_log = model.fit_temporal(callbacks=cb, epochs=epochs, verbose=0)

    spike = model.spike.get_val()
    localt = model.localt.get_val()

    info["scale"] = np.nan
    scale = spike.max(axis=1, keepdims=True)
    info.loc[cell.index, "scale"] = scale
    scale = localt.std(axis=1, keepdims=True)
    info.loc[local.index, "scale"] = scale

    info["denseness"] = np.nan
    info.loc[cell.index, "denseness"] = calc_denseness(spike)
    info.loc[local.index, "denseness"] = calc_denseness(localt)

    info["id"] = -1
    info.loc[cell.index, "id"] = np.arange(cell.shape[0])
    info.loc[local.index, "id"] = np.arange(local.shape[0])

    obj.save_csv(info, **curr, name="info")
    obj.save_numpy(footprint, **curr, name="footprint")
    obj.save_numpy(localx, **curr, name="localx")
    obj.save_numpy(spike, **curr, name="spike")
    obj.save_numpy(localt, **curr, name="localt")

    log = dict(
        data_tag=data_tag,
        spatial_tag=spatial_tag,
        spatial_stage=spatial_stage,
        history=model_log.history,
    )
    return log, tag, stage, "2temporal"
