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

    options = CommandBase.base_options() + [
        options['data-tag'],
        options['stage'],
    ]

    def _handle(self, p):
        tag = p['tag']
        stage = p['stage']
        curr = 'curr' if stage < 0 else f'{stage:03}'
        val = load_numpy(f'hotaru/footprint/{tag}_{curr}.npy')
        param = load_pickle(f'hotaru/footprint/{tag}_{curr}_log.pickle')
        mask = param['mask']
        h, w = mask.shape
        nk = val.shape[0]
        footprint = np.zeros((nk, h, w), np.float32)
        footprint[:, mask] = val
        save_tiff(f'hotaru/output/{tag}_{curr}_cell.tiff', footprint)

        spike = load_numpy(f'hotaru/spike/{tag}_{curr}.npy')
        param = load_pickle(f'hotaru/spike/{tag}_{curr}_log.pickle')
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
        save_csv(f'hotaru/output/{tag}_{curr}_spike.csv', spike)

        calcium = pd.DataFrame(calcium.T)
        calcium.index = time[pad:]
        calcium.index.name = 'time'
        save_csv(f'hotaru/output/{tag}_{curr}_calcium.csv', calcium)

