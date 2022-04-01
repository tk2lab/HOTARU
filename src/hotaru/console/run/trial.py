import click
import matplotlib.pyplot as plot

from hotaru.eval.radius import plot_radius

from .data import data
from .find import find
from .init import init


@click.command()
@click.pass_context
def trial(ctx):
    '''Trial'''

    obj = ctx.obj
    obj.stage = None
    obj.prev_tag = None
    obj.prev_stage = None

    ctx.invoke(data)
    ctx.invoke(find)
    ctx.invoke(init)

    obj.prev_tag = obj.tag
    peak0 = ctx.obj.peak(initial=True)
    obj.prev_stage = '_init'
    peak1 = ctx.obj.peak()

    fig = plt.figure(figsize=(1, 1))

    ax = fig.add_exes([0, 0, 3, 2])
    peak0['color'] = 'b'
    cmap = dict(r='r', g='g', b='b')
    plot_radius(
        ax, peak0, 'intensity', 'color', obj.radius, palette=cmap,
        edgecolor='none', alpha=0.2, size=2,
        legend=False, rasterized=True,
    )

    ax = fig.add_exes([4, 0, 3, 2])
    plot_radius(
        ax, peak1, 'intensity', 'accept', obj.radius, #palette=cmap,
        edgecolor='none', alpha=0.2, size=2,
        legend=False, rasterized=True, ylabel=False,
    )

    fig.savefig('trial.pdf', bbox_inches='tight', pad_inches=0.1)
