import click
import tensorflow as tf

from ...evaluate.summary import write_spike_summary
from ..base import command_wrap
from ..base import configure
from ..base import dynamics_options
from ..base import penalty_options
from ..base import optimizer_options
from ..base import early_stop_options
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--segment-tag", type=str)
@click.option("--segment-stage", type=int)
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
    segment_tag,
    segment_stage,
    dynamics,
    penalty,
    optimizer,
    early_stop,
    epochs,
    batch,
    storage_saving,
):
    """Update Spike from Segment."""

    if segment_tag != tag:
        stage = 1
    else:
        stage = segment_stage

    if storage_saving:
        stage = 999

    data_tag = obj.data_tag("2segment", segment_tag, segment_stage)
    nt = obj.nt(data_tag)

    segment = obj.segment(segment_tag, segment_stage)
    localx = obj.localx(segment_tag, segment_stage)
    nk = segment.shape[0] + localx.shape[0]

    model = obj.model(data_tag, nk)
    model.set_double_exp(**dynamics)
    model.set_penalty(**penalty)
    model.temporal.optimizer.set(**optimizer)
    model.set_early_stop(**early_stop)

    with Progress(length=nt, label="InitT", unit="frame") as prog:
        model.footprint.set_val(segment)
        model.localx.set_val(localx)
        model.prepare_temporal(batch, prog=prog)

    summary_dir = obj.summary_path("temporal", tag, stage)
    cb = obj.callbacks("TrainT", summary_dir)
    model_log = model.fit_temporal(callbacks=cb, epochs=epochs, verbose=0)

    obj.save_numpy(model.spike.get_val(), "spike", tag, stage)
    obj.save_numpy(model.localt.get_val(), "localt", tag, stage)

    log = dict(
        data_tag=data_tag,
        index_tag=segment_tag,
        index_stage=segment_stage,
        history=model_log.history,
    )
    return log, "3temporal", tag, stage
