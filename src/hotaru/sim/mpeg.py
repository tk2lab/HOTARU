import tensorflow as tf
import numpy as np
from matplotlib.pyplot import get_cmap

from hotaru.util.pickle import load_pickle
from hotaru.util.numpy import load_numpy
from hotaru.util.tfrecord import load_tfrecord
from hotaru.util.dataset import unmasked
from hotaru.train.dynamics import SpikeToCalcium
from hotaru.util.mpeg import MpegStream


def make_mpeg(data_tag, tag, stage, has_truth=False):
    outfile = f'hotaru/figure/{tag}_{stage}.mp4'
    cmap = get_cmap('Greens')
    p = load_pickle(f'hotaru/log/{data_tag}_data.pickle')
    mask = p['mask']
    avgt = p['avgt']
    avgx = p['avgx']
    smin = p['smin']
    smax = p['smax']
    sstd = p['sstd']
    hz = p['hz']
    h, w = mask.shape

    data = load_tfrecord(f'hotaru/data/{data_tag}.tfrecord')
    data = unmasked(data, mask)

    if has_truth:
        a0 = load_numpy('truth/a0.npy')
        a0 = a0 > 0.5
        v0 = load_numpy('truth/v0.npy')
        v0 -= v0.min(axis=1, keepdims=True)
        v0 /= (smax - smin)
        v0 *= 2.0

    a2 = load_numpy(f'hotaru/segment/{tag}_{stage}.npy')
    a2 = a2.reshape(-1, h, w)
    a2 = a2 > 0.5
    u2 = load_numpy(f'hotaru/spike/{tag}_{stage}.npy')
    with tf.device('CPU'):
        g = SpikeToCalcium()
        g.set_double_exp(hz, 0.08, 0.16, 6.0)
        v2 = g(u2).numpy()
    v2 *= sstd / (smax - smin)
    v2 -= v2.min(axis=1, keepdims=True)
    v2 *= 2.0

    if has_truth:
        filters = [
            ('drawtext', dict(x=10, y=10, text='orig')),
            ('drawtext', dict(x=10 + w, y=10, text='true')),
            ('drawtext', dict(x=10 + 2 * w, y=10, text='HOTARU')),
        ]
        with MpegStream(3 * w, h, hz, outfile, filters) as mpeg:
            for t, d in enumerate(data.as_numpy_iterator()):
                imgo = d * sstd + avgt[t] + avgx
                imgo = (imgo - smin) / (smax - smin)
                img0 = np.clip((a0 * v0[:, t, None, None]).sum(axis=0), 0.0, 1.0)
                img2 = np.clip((a2 * v2[:, t, None, None]).sum(axis=0), 0.0, 1.0)
                img = np.concatenate([imgo, img0, img2], axis=1)
                img = (255 * cmap(img)).astype(np.uint8)
                mpeg.write(img)
    else:
        filters = [
            ('drawtext', dict(x=10, y=10, text='orig')),
            ('drawtext', dict(x=10 + w, y=10, text='HOTARU')),
        ]
        with MpegStream(2 * w, h, hz, outfile, filters) as mpeg:
            for t, d in enumerate(data.as_numpy_iterator()):
                imgo = d * sstd + avgt[t] + avgx
                imgo = (imgo - smin) / (smax - smin)
                img2 = np.clip((a2 * v2[:, t, None, None]).sum(axis=0), 0.0, 1.0)
                img = np.concatenate([imgo, img2], axis=1)
                img = (255 * cmap(img)).astype(np.uint8)
                mpeg.write(img)
