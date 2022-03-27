import numpy as np

from cleo import Command
from cleo import option

from matplotlib.pyplot import get_cmap

from ..options import options
from ..options import option_type

from ...util.pickle import load_pickle
from ...util.tfrecord import load_tfrecord
from ...util.dataset import unmasked
from ...util.mpeg import MpegStream


class MpegCommand(Command):

    name = 'mpeg'
    description = 'Make mpeg'
    help = '''
'''

    options = [
        options['tag'],
        options['hz'],
        option('cmap', None, '', False, False, False, 'Greens'),
    ]

    def handle(self):
        tag = self.option('tag')
        hz = float(self.option('hz'))
        cmap = get_cmap(self.option('cmap'))

        p = load_pickle(f'hotaru/data/{tag}_log.pickle')
        mask = p['mask']
        avgt = p['avgt']
        avgx = p['avgx']
        mmin = p['mmin']
        mmax = p['mmax']
        mstd = p['mstd']
        h, w = mask.shape

        data = load_tfrecord(f'hotaru/data/{tag}.tfrecord')
        data = unmasked(data, mask)

        smin, smax = np.inf, -np.inf
        with MpegStream(w, h, hz, f'hotaru/fig/{tag}.mp4') as mpeg:
            for i, d in enumerate(data.as_numpy_iterator()):
                img = d * mstd + avgt[i] + avgx
                smin = min(smin, img.min())
                smax = max(smax, img.max())
                img = (img - mmin) / (mmax - mmin)
                img = (255 * cmap(img)).astype(np.uint8)
                mpeg.write(img)
