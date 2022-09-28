import click
import numpy as np
import tensorflow as tf
import pandas as pd

from ...evaluate.summary import write_footprint_summary
from ...evaluate.sparse import calc_sparseness
from ..base import command_wrap
from ..base import configure
from ..base import penalty_options
from ..base import optimizer_options
from ..base import early_stop_options
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--temporal-tag", type=str)
@click.option("--temporal-stage", type=int)
@click.option("--threshold-denseness", type=float)
@penalty_options
@optimizer_options
@early_stop_options
@click.option("--batch", type=int)
@click.option("--epochs", type=int)
@click.option("--storage-saving", is_flag=True)
@click.pass_obj
@command_wrap
def spatial(obj, tag, temporal_tag, temporal_stage, threshold_denseness, penalty, optimizer, early_stop, epochs, batch, storage_saving):
    """Update Footprint from Spike."""

    if temporal_tag is None:
        temporal_tag = tag

    if temporal_stage is None:
        temporal_stage = 1

    if temporal_tag != tag:
        stage = 1
    else:
        if temporal_stage == 999:
            stage = 999
        else:
            stage = temporal_stage + 1

    if storage_saving:
        stage = 999

    prev = dict(tag=temporal_tag, stage=temporal_stage, kind="2temporal")
    curr = dict(tag=tag, stage=stage, kind="1spatial")

    data_tag = obj.data_tag(**prev)
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)
    dynamics = obj.used_dynamics(**prev)

    info = obj.info(**prev)
    spike = obj.spike(**prev)
    localt = obj.localt(**prev)
    old_nk = spike.shape[0]
    old_nl = localt.shape[0]

    model = obj.model(data_tag, info.shape[0])
    model.set_double_exp(**dynamics)
    model.set_penalty(**penalty)
    model.spatial.optimizer.set(**optimizer)
    model.set_early_stop(**early_stop)

    denseness = np.array([calc_sparseness(spk) for spk in spike])
    cell_to_local = denseness > threshold_denseness
    localt_new = model.spike_to_calcium(spike[cell_to_local]).numpy()
    localt = np.concatenate([localt_new, localt], axis=0)
    spike = spike[~cell_to_local]
    nk = spike.shape[0]
    nl = localt.shape[0]

    info["old_kind"] = info.kind
    info["old_id"] = info.id
    cell_info = info.query("kind == 'cell'").copy()
    cell_info["denseness"] = denseness
    cell_info.loc[cell_to_local, "kind"] = "local"
    local_info = info.query("kind == 'local'").copy()
    info = pd.concat([cell_info, local_info], axis=0)
    info.loc[info.kind == "cell", "id"] = np.arange(nk)
    info.loc[info.kind == "local", "id"] = np.arange(nl)

    obj.save_csv(info, **curr, name="info")
    click.echo(f"num: {old_nk}, {old_nl} -> {nk}, {nl}")

    localt /= localt.std(axis=1, keepdims=True)
    with Progress(length=nt, label="InitS", unit="frame") as prog:
        model.spike.set_val(spike)
        model.localt.set_val(localt)
        model.prepare_spatial(batch, prog=prog)

    summary_dir = obj.summary_path("spatial", tag, stage)
    cb = obj.callbacks("TrainS", summary_dir)
    model_log = model.fit_spatial(callbacks=cb, epochs=epochs, verbose=0)

    footprint = model.footprint.get_val()
    scale = footprint.max(axis=1, keepdims=True)
    footprint /= scale
    spike *= scale

    localx = model.localx.get_val()
    scale = localx.std(axis=1, keepdims=True)
    localx /= scale
    localt *= scale

    obj.save_numpy(footprint, **curr, name="footprint")
    obj.save_numpy(localx, **curr, name="localx")
    obj.save_numpy(spike, **curr, name="spike") 
    obj.save_numpy(localt, **curr, name="localt")

    log = dict(
        data_tag=data_tag,
        temporal_tag=temporal_tag,
        temporal_stage=temporal_stage,
        history=model_log.history,
    )
    return log, tag, stage, "1spatial"
