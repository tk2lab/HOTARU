import click
import tensorflow as tf

from ...evaluate.summary import write_footprint_summary
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--spike-tag", type=str)
@click.option("--spike-stage", type=int)
@click.option("--storage-saving", is_flag=True)
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
def spatial(obj, tag, spike_tag, spike_stage, storage_saving, batch, **args):
    """Update Footprint from Spike."""

    if spike_tag != tag:
        stage = 1
    else:
        stage = spike_stage

    if storage_saving:
        stage = 999

    penalty_args = [v for k, v in args.items() if k.startswith("penalty")]
    opt_args = [v for k, v in args.items() if k.startswith("opt")]
    early_stop_args = [v for k, v in args.items() if k.startswith("early")]

    prev_log = obj.log("1temporal", spike_tag, spike_stage)
    data_tag = prev_log["data_tag"]
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    spike = obj.spike(spike_tag, spike_stage)
    localt = obj.localt(spike_tag, spike_stage)
    nk = spike.shape[0] + localt.shape[0]
    tau_args = obj.used_tau(spike_tag, spike_stage).values()
    data_log = obj.log("1data", data_tag, 0)
    nt = data_log["nt"]

    summary_dir = obj.summary_path("spatial", tag, stage)
    writer = tf.summary.create_file_writer(summary_dir)

    model = obj.model(data_tag, nk)
    model.set_double_exp(*tau_args)
    model.set_penalty(*penalty_args)
    model.set_optimizer(*opt_args)
    model.set_early_stop(*early_stop_args)

    with Progress(length=nt, label="InitS", unit="frame") as prog:
        model.spike.set_val(spike)
        model.localt.set_val(localt)
        model.prepare_spatial(batch, prog=prog)

    cb = obj.callbacks("TrainS", summary_dir)
    model_log = model.fit_spatial(callbacks=cb, verbose=0)

    footprint = model.footprint.get_val()
    localx = model.localx.get_val()
    obj.save_numpy(footprint, "footprint", tag, stage)
    obj.save_numpy(localx, "localx0", tag, stage)
    write_footprint_summary(writer, footprint, mask)

    log = dict(
        data_tag=data_tag,
        segment_tag=prev_log["segment_tag"],
        segment_stage=prev_log["segment_stage"],
        history=model_log.history,
    )
    return log, "2spatial", tag, stage
