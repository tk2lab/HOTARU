import click

from ...train.spike import SpikeModel
from ...util.distribute import MirroredStrategy
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
    log = model.fit(obj.segment, **obj.fit_opt)
    strategy.close()

    obj.save_numpy(model.spike.get_val(), "spike")
    return dict(history=log.history)
