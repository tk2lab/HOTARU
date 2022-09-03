import click
import tensorflow as tf

from ...evaluate.summary import write_spike_summary
from ...train.temporal import TemporalModel
from ..base import run_command
from ..progress import ProgressCallback
from .options import model_options


@run_command(
    click.Option(["--stage", "-s"], type=int),
    click.Option(["--segment-tag", "-T"]),
    click.Option(["--segment-stage", "-S"], type=int),
    *model_options(),
)
def temporal(obj):
    """Update spike"""

    if obj.stage == -1:
        obj["stage"] = "_curr"

    if obj.segment_tag == "":
        if obj.stage == 0:
            obj["segment_tag"] = obj.init_tag
        else:
            obj["segment_tag"] = obj.tag

    if obj.segment_stage == -1:
        obj["segment_stage"] = obj.stage

    if "temporal_model" not in obj:
        with obj.strategy.scope():
            obj.temporal_model = TemporalModel(
                obj.data,
                obj.segment.shape[0],
                obj.nx,
                obj.nt,
                obj.tau,
                **obj.reg,
            )
            obj.temporal_model.compile(**obj.compile_opt)
    model = obj.temporal_model

    log_dir = obj.summary_path()
    writer = tf.summary.create_file_writer(log_dir)

    with click.progressbar(length=obj.nt, label="InitT") as prog:
        model.prepare_fit(obj.segment, obj.batch, prog=prog)

    cb = [
        ProgressCallback("TrainT", obj.epoch),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq="batch",
            write_graph=False,
        ),
    ]
    log = model.fit(callbacks=cb, verbose=0, **obj.fit_opt)

    val = model.spike.get_val()
    obj.save_numpy(val, "spike")
    write_spike_summary(writer, val)
    return dict(history=log.history)
