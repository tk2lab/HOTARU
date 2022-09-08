import click
import tensorflow as tf

from ...evaluate.summary import write_spike_summary
from ...train.temporal import TemporalModel
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--segment-tag", type=str)
@click.option("--segment-stage", type=int)
@click.option("--storage-saving", is_flag=True)
@click.option("--tau-rise", type=float)
@click.option("--tau-fall", type=float)
@click.option("--tau-scale", type=float)
@click.option("--penalty-la", type=float)
@click.option("--penalty-lu", type=float)
@click.option("--penalty-bx", type=float)
@click.option("--penalty-bt", type=float)
@click.option("--learning-rate", type=float)
@click.option("--nesterov-scale", type=float)
@click.option("--steps-per-epoch", type=int)
@click.option("--min-delta", type=float)
@click.option("--patience", type=int)
@click.option("--epochs", type=int)
@click.option("--batch", type=int)
@click.pass_obj
@command_wrap
def temporal(
    obj,
    tag,
    segment_tag,
    segment_stage,
    storage_saving,
    batch,
    **args,
):
    """Update Spike from Segment."""

    if segment_tag != tag:
        stage = 1
    else:
        if segment_stage == 999:
            stage = 999
        else:
            stage = segment_stage + 1

    if storage_saving:
        stage = 999

    data_tag = obj.log("3segment", segment_tag, segment_stage)["data_tag"]
    data = obj.data(data_tag)
    segment = obj.segment(segment_tag, segment_stage)

    nk = segment.shape[0]
    nx = obj.nx(data_tag)
    nt = obj.nt(data_tag)
    hz = obj.hz(data_tag)
    tau_opt = {k[4:]: v for k, v in args.items() if k[:3] == "tau"}

    if not hasattr(obj, "temporal_model"):
        with obj.strategy.scope():
            model = TemporalModel(
                data,
                nk,
                nx,
                nt,
                hz,
                **tau_opt,
                **obj.penalty_opt(**args),
            )
            model.compile(**obj.compile_opt(**args))
        obj.temporal_model = model
    model = obj.temporal_model

    summary_dir = obj.summary_path("temporal", tag, stage)
    writer = tf.summary.create_file_writer(summary_dir)

    with Progress(length=nt, label="InitT", unit="frame") as prog:
        model.prepare_fit(segment, batch, prog=prog)

    cb = obj.callbacks("TrainT", summary_dir)
    model_log = model.fit(callbacks=cb, verbose=0, **obj.fit_opt(**args))

    val = model.spike.get_val()
    obj.save_numpy(val, "spike", tag, stage)
    write_spike_summary(writer, val)

    log = dict(
        segment_tag=segment_tag,
        segment_stage=segment_stage,
        data_tag=data_tag,
        history=model_log.history,
    )
    return log, "1temporal", tag, stage
