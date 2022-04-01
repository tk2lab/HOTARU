import click

from .data import data
from .find import find
from .init import init
from .temporal import temporal
from .spatial import spatial
from .clean import clean
from .output import output


@click.command()
@click.option(
    '--start',
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    '--end',
    type=int,
    default=10,
    show_default=True,
)
@click.pass_context
def auto(ctx, start, end):
    '''Auto'''

    ctx.obj.prev_stage = None
    for i in range(start, end):
        print(i)
        if i == 0:
            ctx.obj.stage = None
            ctx.invoke(data)
            ctx.invoke(find)
            ctx.invoke(init)
            ctx.obj.stage = 0
        else:
            ctx.obj.stage = i
            ctx.invoke(clean)
        ctx.invoke(temporal)
        ctx.invoke(spatial)
    ctx.obj.stage = end
    ctx.invoke(clean)
    ctx.invoke(temporal)
    ctx.invoke(output)
