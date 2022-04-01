import click
import numpy as np
import matplotlib.pyplot as plt


@click.command()
@click.pass_obj
def figure(obj):
    '''Figure'''

    obj.prev_tag = obj.tag
    obj.prev_stage = obj.stage

    segment = obj.segment()
    mask = obj.mask()
    nk = segment.shape[0]
    h, w = mask.shape
    imgs = np.zeros((nk, h, w))
    imgs[:, mask] = segment

    spike = obj.spike()
    nk, nt = spike.shape
    hz = obj.tau['hz']

    fig = plt.figure(figsize=(1, 1))

    ax = fig.add_exes([0, 0, 10, 10 / w * h])
    plot_maximum(ax, imgs, 0.0, 1.0)

    ax = fig.add_exes([0, 0, 10, 10])
    spike /= spike.max(axis=1, keepdims=True)
    ax.imshow(spike, aspect='auto', cmap='Reds', vmin=0.0, vmax=0.5)

    fig.savefig('radius_intensity.pdf', bbox_inches='tight', pad_inches=0.1)
