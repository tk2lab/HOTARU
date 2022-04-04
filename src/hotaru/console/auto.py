import click

from .run.data import data
from .run.find import find
from .run.init import init
from .run.temporal import temporal
from .run.spatial import spatial
from .run.clean import clean
from .run.output import output


@click.command()
@click.option(
    '--max-iteration',
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    '--overwrite',
    is_flag=True,
)
@click.pass_context
def auto(ctx, max_iteration, overwrite):
    '''Auto'''

    num_cell = -1
    for stage in range(max_iteration):
        if overwrite:
            ctx.obj['stage'] = -1
        else:
            ctx.obj['stage'] = stage 
        if stage == 0:
            ctx.invoke(data)
            ctx.invoke(find)
            ctx.invoke(init)
        else:
            ctx.invoke(clean, stage=stage)
        num_cell, old_cell = ctx.obj.num_cell, num_cell
        ctx.invoke(temporal, stage=stage)
        if num_cell == old_cell:
            break
        ctx.invoke(spatial, stage=stage)
    ctx.invoke(output, stage=stage)
