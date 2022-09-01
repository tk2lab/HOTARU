import click
import tensorflow as tf

from ...evaluate.summary import write_spike_summary
from ...train.spike import SpikeModel
from ...util.distribute import MirroredStrategy
from ...util.progress import ProgressCallback
from ..base import run_command
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

    log_dir = obj.summary_path()
    writer = tf.summary.create_file_writer(log_dir)

    cb = [
        ProgressCallback("temporal", obj.epoch),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq="batch",
            write_graph=False,
        ),
    ]

    strategy = MirroredStrategy()
    with strategy.scope():
        model = SpikeModel(
            obj.data,
            obj.segment.shape[0],
            obj.nx,
            obj.nt,
            obj.tau,
            **obj.reg,
        )
        model.compile(**obj.compile_opt)
    log = model.fit(obj.segment, callbacks=cb, verbose=0, **obj.fit_opt)
    strategy.close()

    val = model.spike.get_val()
    obj.save_numpy(val, "spike")
    write_spike_summary(writer, val)
    return dict(history=log.history)
