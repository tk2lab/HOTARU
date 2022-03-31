import click

from .data import data
from .find import find
from .init import init
from .temporal import temporal
from .spatial import spatial
from .clean import clean


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

    for i in range(start, end + 1):
        print(i)
        if i == 0:
            ctx.obj.stage = None
            ctx.invoke(data)
            ctx.invoke(find)
            ctx.invoke(init)
            ctx.obj.stage = 0
        else:
            ctx.obj.stage = i
            ctx.obj.prev_stage = i - 1
            ctx.invoke(clean)
            ctx.obj.prev_stage = i
        ctx.invoke(temporal)
        ctx.invoke(spatial)
