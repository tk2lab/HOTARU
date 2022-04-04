import tensorflow as tf
import numpy as np
import pandas as pd
import tifffile
import click

from hotaru.train.dynamics import SpikeToCalcium
from hotaru.util.csv import save_csv

from .base import run_command


@run_command(
    click.Option(['--stage', '-s'], type=int),
)
def output(obj):
    '''Output'''

    obj['segment_tag'] = obj.tag
    obj['segment_stage'] = obj.stage
    obj['spike_tag'] = obj.tag
    obj['spike_stage'] = obj.stage

    segment = obj.segment
    mask = obj.mask
    nk = segment.shape[0]
    h, w = mask.shape
    
    imgs = np.zeros((nk, h, w))
    imgs[:, mask] = segment
    tifffile.imsave('footprints.tif', imgs)

    u = obj.spike
    with tf.device('CPU'):
        g = SpikeToCalcium()
        g.set_double_exp(**obj.used_tau)
        v = g(u).numpy()

    gap = u.shape[1] - v.shape[1]
    time = np.arange(-gap, v.shape[1]) / obj.hz
    cols = [f'cell{i:03}' for i in range(u.shape[0])]
    u = pd.DataFrame(u.T, columns=cols)
    u.index = time
    u.index.name = 'time'
    v = pd.DataFrame(v.T, columns=cols)
    v.index = time[gap:]
    v.index.name = 'time'
    save_csv('spikes.csv', u)
    save_csv('trace.csv', v)
