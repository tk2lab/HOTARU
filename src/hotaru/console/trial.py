import matplotlib.pyplot as plt
import click

from hotaru.eval.radius import plot_radius
from hotaru.eval.circle import plot_circle

from .run.data import data
from .run.find import find
from .run.test import test


@click.command()
@click.pass_context
def trial(ctx):
    '''Trial'''

    obj = ctx.obj

    ctx.invoke(data)
    ctx.invoke(find)
    obj['force'] = True
    ctx.invoke(test)

    radius_min = obj.used_radius_min
    radius_max = obj.used_radius_max
    distance = obj.used_distance
    h, w = obj.mask.shape
    peak0 = obj.peak
    peak1 = obj.peak_trial

    fig = plt.figure(figsize=(1, 1))
    cmap = dict(r='r', g='g', b='b')

    ax = fig.add_axes([0, 0, 3, 2])
    peak0['color'] = 'b'
    plot_radius(
        ax, peak0, 'intensity', 'color', obj.radius, palette=cmap,
        edgecolor='none', alpha=0.2, size=2,
        legend=False, rasterized=True,
    )

    ax = fig.add_axes([4, 0, 3, 2])
    plot_radius(
        ax, peak1, 'intensity', 'accept', obj.radius, #palette=cmap,
        edgecolor='none', alpha=0.2, size=2,
        legend=False, rasterized=True, ylabel=False,
    )

    ax = fig.add_axes([0, 3, 7, 7 * h / w])
    cond = (peak1['radius'] > radius_min) & (peak1['radius'] < radius_max)
    ax.scatter(
        peak1.loc[cond, 'x'].values, peak1.loc[cond, 'y'].values,
        s=5, c='r',
    )
    plot_circle(
        ax, peak1[cond].copy(), h, w, distance, color='g',
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    path = obj.out_path('figure', obj.tag, '_trial')
    fig.savefig(f'{path}.pdf', bbox_inches='tight', pad_inches=0.1)
