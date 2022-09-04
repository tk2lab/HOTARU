import click

from ..base import run_command
from ..evaluate.figure import figure
from ..run.clean import clean
from ..run.data import data
from ..run.find import find
from ..run.init import init
from ..run.output import output
from ..run.spatial import spatial
from ..run.temporal import temporal


@run_command(
    click.Option(
        ["--max-iteration"],
        type=int,
    ),
    click.Option(
        ["--storing-intermidiate-results"],
        is_flag=True,
        default=None,
    ),
    click.Option(
        ["--non-stop"],
        is_flag=True,
        default=None,
    ),
    pass_context=True,
)
def auto(ctx, **args):
    """Auto"""

    num_cell = -1
    for stage in range(ctx.obj.max_iteration):
        if ctx.obj.storing_intermidiate_results:
            _stage = stage
        else:
            _stage = "_curr"
        if stage == 0:
            ctx.invoke(data)
            ctx.invoke(find)
            ctx.invoke(init)
            num_cell, old_cell = ctx.obj.num_cell(-2), num_cell
        else:
            ctx.invoke(clean, stage=_stage)
            num_cell, old_cell = ctx.obj.num_cell(_stage), num_cell
        ctx.invoke(temporal, stage=_stage)
        if not ctx.obj.non_stop and (num_cell == old_cell):
            break
        ctx.invoke(spatial, stage=_stage)
    ctx.invoke(output, stage=_stage)
    ctx.invoke(figure, stage=_stage)
