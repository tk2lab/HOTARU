import click

from ...train.footprint import FootprintModel
from ...util.distribute import MirroredStrategy
from ..base import run_command
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

    strategy = MirroredStrategy()
    with strategy.scope():
        model = FootprintModel(
            obj.data,
            obj.spike.shape[0],
            obj.nx,
            obj.nt,
            obj.tau,
            **obj.reg,
        )
        model.compile(**obj.compile_opt)
    log = model.fit(obj.spike, **obj.fit_opt)
    strategy.close()

    obj.save_numpy(model.footprint.get_val(), "footprint")
    return dict(history=log.history)
