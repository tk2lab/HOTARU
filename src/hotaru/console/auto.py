import click

from .run.data import data
from .run.find import find
from .run.init import init
from .run.temporal import temporal
from .run.spatial import spatial
from .run.clean import clean
from .run.output import output
from .figure import figure


@click.command()
@click.option(
    '--max-iteration',
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    '--storing-intermidiate-results',
    is_flag=True,
)
@click.option(
    '--non-stop',
    is_flag=True,
)
@click.pass_context
def auto(ctx, max_iteration, storing_intermidiate_results, non_stop):
    '''Auto'''

    num_cell = -1
    for stage in range(max_iteration):
        if storing_intermidiate_results:
            _stage = stage
        else:
            _stage = '_curr'
        if stage == 0:
            ctx.invoke(data)
            ctx.invoke(find)
            ctx.invoke(init)
            num_cell, old_cell = ctx.obj.num_cell(-2), num_cell
        else:
            ctx.invoke(clean, stage=_stage)
            num_cell, old_cell = ctx.obj.num_cell(_stage), num_cell
        ctx.invoke(temporal, stage=_stage)
        if not non_stop and (num_cell == old_cell):
            break
        ctx.invoke(spatial, stage=_stage)
    ctx.invoke(output, stage=_stage)
    ctx.invoke(figure, stage=_stage)
