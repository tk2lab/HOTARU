import click
import tensorflow as tf

from ...evaluate.summary import write_footprint_summary
from ...train.spatial import SpatialModel
from ..base import command_wrap
from ..base import configure


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--stage", type=int)
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
def spatial(obj, tag, stage, batch, **args):
    """Update footprint"""

    if stage <= 0:
        stage = "_curr"

    spike_tag = tag
    spike_stage = stage

    prev_log = obj.log("1temporal", spike_tag, spike_stage)
    data_tag = prev_log["data_tag"]
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    spike = obj.spike(spike_tag, spike_stage)

    nk = spike.shape[0]
    nx = obj.nx(data_tag)
    nt = obj.nt(data_tag)
    hz = obj.hz(data_tag)
    tau_opt = obj.used_tau(spike_tag, spike_stage)

    if not hasattr(obj, "spatial_model"):
        with obj.strategy.scope():
            model = SpatialModel(
                data,
                nk,
                nx,
                nt,
                hz,
                **tau_opt,
                **obj.penalty_opt(**args),
            )
            model.compile(**obj.compile_opt(**args))
        obj.spatial_model = model
    model = obj.spatial_model

    summary_dir = obj.summary_path("spatial", tag, stage)
    writer = tf.summary.create_file_writer(summary_dir)

    with click.progressbar(length=nt, label="InitS") as prog:
        model.prepare_fit(spike, batch, prog=prog)

    cb = obj.callbacks("TrainS", summary_dir)
    model_log = model.fit(callbacks=cb, verbose=0, **obj.fit_opt(**args))

    val = model.footprint.get_val()
    obj.save_numpy(val, "footprint", tag, stage)
    write_footprint_summary(writer, val, mask)

    log = dict(
        spike_tag=spike_tag,
        spike_stage=spike_stage,
        segment_tag=prev_log["segment_tag"],
        segment_stage=prev_log["segment_stage"],
        data_tag=data_tag,
        history=model_log.history,
    )
    return log, "2spatial", tag, stage
