import numpy as np
import pandas as pd

from .base import CommandBase
from .options import options

from ..image.filter.gaussian import gaussian
from ..train.dynamics import SpikeToCalcium
from ..util.numpy import load_numpy
from ..util.pickle import load_pickle
from ..util.tiff import save_tiff
from ..util.csv import save_csv


class OutputCommand(CommandBase):

    name = 'output'
    _type = 'output'
    description = 'Output tif and csv files'
    help = '''
'''

    options = CommandBase.base_options('work') + [
        options['data-tag'],
    ]

    def _handle(self, base):

        tag = self.option('tag')
        val = load_numpy(f'hotaru/footprint/{tag}.npy')
        param = load_pickle(f'hotaru/footprint/{tag}_log.pickle')
        mask = param['mask']
        h, w = mask.shape
        nk = val.shape[0]
        footprint = np.zeros((nk, h, w), np.float32)
        footprint[:, mask] = val
        save_tiff(f'{base}_cell.tiff', footprint)

        spike = load_numpy(f'hotaru/spike/{tag}.npy')
        param = load_pickle(f'hotaru/spike/{tag}_log.pickle')
        hz, *tau = param['hz'], param['tau1'], param['tau2'], param['tscale']
        nk, nu = spike.shape
        spike_to_calcium = SpikeToCalcium()
        spike_to_calcium.set_double_exp(hz, *tau)
        calcium = spike_to_calcium(spike).numpy()
        pad = spike_to_calcium.pad
        time = (np.arange(nu) - pad) / hz

        spike = pd.DataFrame(spike.T)
        spike.index = time
        spike.index.name = 'time'
        save_csv(f'{base}_spike.csv', spike)

        calcium = pd.DataFrame(calcium.T)
        calcium.index = time[pad:]
        calcium.index.name = 'time'
        save_csv(f'{base}_calcium.csv', calcium)

