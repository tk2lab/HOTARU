import click
import tensorflow as tf

from ...evaluate.summary import write_spike_summary
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--segment-tag", type=str)
@click.option("--segment-stage", type=int)
@click.option("--storage-saving", is_flag=True)
@click.option("--tau1", type=float)
@click.option("--tau2", type=float)
@click.option("--penalty-footprint", type=float)
@click.option("--penalty-spike", type=float)
@click.option("--penalty-localx", type=float)
@click.option("--penalty-localt", type=float)
@click.option("--penalty-spatial", type=float)
@click.option("--penalty-temporal", type=float)
@click.option("--optimizer-learning-rate", type=float)
@click.option("--optimizer-nesterov-scale", type=float)
@click.option("--optimizer-reset-interval", type=int)
@click.option("--early-stop-min-delta", type=float)
@click.option("--early-stop-patience", type=int)
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

    tau_args = [v for k, v in args.items() if k.startswith("tau")]
    penalty_args = [v for k, v in args.items() if k.startswith("penalty")]
    opt_args = [v for k, v in args.items() if k.startswith("opt")]
    early_stop_args = [v for k, v in args.items() if k.startswith("early")]

    summary_dir = obj.summary_path("temporal", tag, stage)
    writer = tf.summary.create_file_writer(summary_dir)

    prev_log = obj.log("3segment", segment_tag, segment_stage)
    segment = obj.segment(segment_tag, segment_stage)
    localx = obj.localx(segment_tag, segment_stage)
    nk = segment.shape[0] + localx.shape[0]
    data_tag = prev_log["data_tag"]
    data_log = obj.log("1data", data_tag, 0)
    nt = data_log["nt"]

    model = obj.model(data_tag, nk)
    model.set_double_exp(*tau_args)
    model.set_penalty(*penalty_args)
    model.set_optimizer(*opt_args)
    model.set_early_stop(*early_stop_args)

    with Progress(length=nt, label="InitT", unit="frame") as prog:
        model.footprint.set_val(segment)
        model.localx.set_val(localx)
        model.prepare_temporal(batch, prog=prog)

    cb = obj.callbacks("TrainT", summary_dir)
    model_log = model.fit_temporal(callbacks=cb, verbose=0)

    spike = model.spike.get_val()
    localt = model.localt.get_val()
    obj.save_numpy(spike, "spike", tag, stage)
    obj.save_numpy(localt, "localt", tag, stage)
    write_spike_summary(writer, spike)

    log = dict(
        segment_tag=segment_tag,
        segment_stage=segment_stage,
        data_tag=data_tag,
        history=model_log.history,
    )
    return log, "1temporal", tag, stage
