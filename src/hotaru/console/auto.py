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
def auto(ctx, start, end, force):
    '''Auto'''

    for i in range(start, end + 1):
        ctx.obj['stage'] = i
        if i == 0:
            ctx.invoke(data)
            ctx.invoke(find)
            ctx.invoke(init)
        else:
            ctx.invoke(clean)
        ctx.invoke(temporal)
        if i != end:
            ctx.invoke(spatial)
        else:
            ctx.invoke(output)
