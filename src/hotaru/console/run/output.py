import click
import tensorflow as tf
import numpy as np
import tifffile

from hotaru.train.dynamics import SpikeToCalcium
from hotaru.util.csv import save_csv


@click.command()
@click.pass_obj
def output(obj):
    '''Output'''

    segment = obj.segment()
    mask = obj.mask()
    nk = segment.shape[0]
    h, w = mask.shape
    
    imgs = np.zeros((nk, h, w))
    imgs[:, mask] = segment
    tifffile.imsave('footprints.tif')

    u = obj.spike()
    with tf.device('CPU'):
        g = SpikeToCalcium()
        g.set_double_exp(**obj.tau())
        v = g(u).numpy()
    save_csv('spikes.csv', u)
    save_csv('trace.csv', v)
