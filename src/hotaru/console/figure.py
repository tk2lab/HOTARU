import numpy as np
import matplotlib.pyplot as plt
import click

from hotaru.eval.footprint import plot_maximum

from .run.data import data
from .run.find import find
from .run.test import test


@click.command()
@click.option('--stage', '-s', type=int)
@click.pass_obj
def figure(obj, stage):
    '''Figure'''

    obj['stage'] = stage
    hz = obj.hz
    mask = obj.mask
    h, w = mask.shape

    val = obj.segment
    seg = np.zeros((val.shape[0], h, w))
    seg[:, mask] = val
    spike = obj.spike

    fig = plt.figure(figsize=(1, 1))

    ax = fig.add_axes([0, 0, 7, 7 * h / w])
    plot_maximum(ax, seg, 0.0, 1.0)

    ax = fig.add_axes([0, - 7 * h / w - 1, 7, 5])
    spike /= spike.max(axis=1, keepdims=True)
    ax.imshow(spike, aspect='auto', cmap='Reds', vmin=0.0, vmax=0.5)
    ax.set_xlabel('time (frame)')
    ax.set_ylabel('cells')
    ax.set_yticks([])

    path = obj.out_path('figure', obj.tag, obj.stage)
    fig.savefig(f'{path}.pdf', bbox_inches='tight', pad_inches=0.1)
