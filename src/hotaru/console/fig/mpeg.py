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
        smin = p['smin']
        smax = p['smax']
        sstd = p['sstd']
        h, w = mask.shape

        data = load_tfrecord(f'hotaru/data/{tag}.tfrecord')
        data = unmasked(data, mask)

        with MpegStream(w, h, hz, f'hotaru/fig/{tag}.mp4') as mpeg:
            for i, d in enumerate(data.as_numpy_iterator()):
                img = d * sstd + avgt[i] + avgx
                img = (img - smin) / (smax - smin)
                img = (255 * cmap(img)).astype(np.uint8)
                mpeg.write(img)
