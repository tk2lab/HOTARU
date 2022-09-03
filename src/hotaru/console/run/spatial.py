import click
import tensorflow as tf

from ...evaluate.summary import write_footprint_summary
from ...train.spatial import SpatialModel
from ..base import run_command
from ..progress import ProgressCallback
from .options import model_options


@run_command(
    click.Option(["--stage", "-s"], type=int),
    click.Option(["--spike-tag", "-T"]),
    click.Option(["--spike-stage", "-S"], type=int),
    *model_options(),
    click.Option(["--batch"], type=int, default=100, show_default=True),
)
def spatial(obj):
    """Update footprint"""

    if obj.stage == -1:
        obj["stage"] = "_curr"

    if obj.spike_tag == "":
        obj["spike_tag"] = obj.tag

    if obj.spike_stage == -1:
        obj["spike_stage"] = obj.stage

    if "spatial_model" not in obj:
        with obj.strategy.scope():
            obj.spatial_model = SpatialModel(
                obj.data,
                obj.spike.shape[0],
                obj.nx,
                obj.nt,
                obj.tau,
                **obj.reg,
            )
            obj.spatial_model.compile(**obj.compile_opt)
    model = obj.spatial_model

    log_dir = obj.summary_path()
    writer = tf.summary.create_file_writer(log_dir)

    with click.progressbar(length=obj.nt, label="InitS") as prog:
        model.prepare_fit(obj.spike, obj.batch, prog=prog)

    cb = [
        ProgressCallback("TrainS", obj.epoch),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq="batch",
            write_graph=False,
        ),
    ]
    log = model.fit(callbacks=cb, verbose=0, **obj.fit_opt)

    val = model.footprint.get_val()
    obj.save_numpy(val, "footprint")
    write_footprint_summary(writer, val, obj.mask)
    return dict(history=log.history)
