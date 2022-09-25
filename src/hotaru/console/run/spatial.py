import click
import numpy as np
import tensorflow as tf

from ...evaluate.summary import write_footprint_summary
from ..base import command_wrap
from ..base import configure
from ..base import penalty_options
from ..base import optimizer_options
from ..base import early_stop_options
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--spike-tag", type=str)
@click.option("--spike-stage", type=int)
@penalty_options
@optimizer_options
@early_stop_options
@click.option("--batch", type=int)
@click.option("--epochs", type=int)
@click.option("--storage-saving", is_flag=True)
@click.pass_obj
@command_wrap
def spatial(obj, tag, spike_tag, spike_stage, penalty, optimizer, early_stop, epochs, batch, storage_saving):
    """Update Footprint from Spike."""

    if spike_tag != tag:
        stage = 1
    else:
        if spike_stage == 999:
            stage = 999
        else:
            stage = spike_stage + 1

    if storage_saving:
        stage = 999

    data_tag = obj.data_tag("3temporal", spike_tag, spike_stage)
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)

    dynamics = obj.used_dynamics(spike_tag, spike_stage)
    spike = obj.spike(spike_tag, spike_stage)
    localt = obj.localt(spike_tag, spike_stage)
    nk = spike.shape[0] + localt.shape[0]

    model = obj.model(data_tag, nk)
    model.set_double_exp(**dynamics)
    model.set_penalty(**penalty)
    model.spatial.optimizer.set(**optimizer)
    model.set_early_stop(**early_stop)

    with Progress(length=nt, label="InitS", unit="frame") as prog:
        model.spike.set_val(spike)
        model.localt.set_val(localt)
        model.prepare_spatial(batch, prog=prog)

    summary_dir = obj.summary_path("spatial", tag, stage)
    cb = obj.callbacks("TrainS", summary_dir)
    model_log = model.fit_spatial(callbacks=cb, epochs=epochs, verbose=0)

    footprint = model.footprint.get_val()
    localx = model.localx.get_val()
    footprint = np.concatenate([footprint, localx], axis=0)
    obj.save_numpy(footprint, "footprint", tag, stage)

    log = dict(
        data_tag=data_tag,
        index_tag=obj.index_tag("3temporal", spike_tag, spike_stage),
        index_stage=obj.index_stage("3temporal", spike_tag, spike_stage),
        history=model_log.history,
    )
    return log, "1spatial", tag, stage
